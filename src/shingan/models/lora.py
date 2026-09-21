"""Track B: QLoRA fine-tuning of the base language model.

This module is the one place in the package that imports the ``train`` extra, and it
does so **lazily and only inside functions**. That is not tidiness: ``shingan doctor``
and every CPU-only run import this module to ask what is installed, and on a clean
checkout torch is not. A module-level ``import torch`` would make the whole CLI
unimportable on the machine the core install targets.

Two invariants are enforced here rather than trusted to the config file, because both
failures are silent and both invalidate every number downstream:

**``packing`` must be false.** Packing concatenates short samples across their prompt
boundaries to fill a sequence. Every example here carries its own label, horizon and
ticker, so packing trains the model to continue one company's answer into the next
company's question — corrupting supervision while making throughput look better. The
flag is asserted at run time rather than read from YAML, so a hand-edited config
cannot switch it on.

**The training file must not contain test-fold rows.** ``shingan data sft`` writes a
manifest stating that ``test`` was deliberately not emitted, and the CLI cannot check
that the *file* agrees. This module reads every row's ``meta.split`` and refuses the
run if a test row is present: writing the evaluation sample into the training file is
precisely the leakage the purge margin exists to prevent, and it would show up as a
good score rather than as an error.
"""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shingan.config import LoraConfig
from shingan.logging_utils import get_logger

logger = get_logger(__name__)

#: Import names of the packages the training path needs, in the order a reader would
#: install them. ``bitsandbytes`` is required rather than optional because the config
#: defaults to a 4-bit base model and a 14B model does not fit a 32 GB card otherwise.
REQUIRED_MODULES: tuple[str, ...] = (
    "torch",
    "transformers",
    "peft",
    "trl",
    "datasets",
    "accelerate",
    "bitsandbytes",
)

#: Import names that improve the run or are needed for particular tokenisers. Reported
#: but not required.
OPTIONAL_MODULES: tuple[str, ...] = ("sentencepiece", "safetensors", "einops")

#: What the user is told when the extra is absent. Deliberately spells out the cu128
#: index: on Windows with an RTX 5090 (Blackwell, sm_120) a plain ``pip install torch``
#: resolves a wheel that has no kernels for the card, and the failure surfaces much
#: later as a CUDA "no kernel image is available" error mid-run.
INSTALL_HINT = (
    "the training stack is not installed.\n"
    "  Windows / RTX 5090 (Blackwell sm_120) — install torch from the cu128 index FIRST,\n"
    "  otherwise pip resolves the CPU wheel:\n"
    "    pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
    "    pip install -e \".[train]\"\n"
    "  See docs/06-windows-setup.md and docs/adr/0002-training-stack-windows.md."
)

#: Splits a training file is allowed to contain. ``test`` is absent on purpose — see the
#: module docstring.
PERMITTED_TRAIN_SPLITS: frozenset[str] = frozenset({"train", "valid"})

#: Minimum compute capability that has kernels in a cu128 build. Below this the wheel
#: predates the architecture and the run dies inside the first attention kernel.
_BLACKWELL_CAPABILITY = (12, 0)


class MissingTrainDependencies(RuntimeError):
    """The ``train`` extra is not importable.

    A condition rather than an error category: the CLI catches this single class and
    prints :data:`INSTALL_HINT`, and nothing else in the package catches it
    generically.
    """

    def __init__(self, missing: Iterable[str]) -> None:
        self.missing = tuple(missing)
        super().__init__(
            f"the training stack is incomplete: missing {', '.join(self.missing)}. {INSTALL_HINT}"
        )


class LoraLeakageError(RuntimeError):
    """A training run was asked to proceed in a state that would leak the test block.

    Raised for ``packing: true`` and for a training file containing test-fold rows.
    Named ``...Error`` because a caller catches it as a category —
    ``shingan train lora`` catches exactly this class and reports it as a refusal.
    """


# --------------------------------------------------------------------------- #
# Environment probes (must not import torch at module scope)
# --------------------------------------------------------------------------- #


def _importable(module: str) -> bool:
    """Whether ``module`` can be imported, without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        # A namespace package whose parent is missing raises ValueError on some
        # Python versions; either way it is not importable.
        return False


def dependency_report() -> dict[str, bool]:
    """``{module: importable}`` for the required and optional training packages."""
    names = (*REQUIRED_MODULES, *OPTIONAL_MODULES)
    return {name: _importable(name) for name in names}


def missing_required_modules() -> list[str]:
    """Required modules that are not importable, in :data:`REQUIRED_MODULES` order."""
    return [name for name in REQUIRED_MODULES if not _importable(name)]


def describe_train_environment() -> dict[str, Any]:
    """Interpreter, torch build and CUDA device facts, for the report and ``doctor``.

    Returns a partial mapping when torch is absent rather than raising: the caller
    prints whatever is known. Every value is JSON-serialisable so it can travel into
    ``run.json``.
    """
    facts: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    if not _importable("torch"):
        facts["torch"] = "not installed"
        return facts

    import torch

    facts["torch"] = torch.__version__
    facts["torch_cuda_build"] = getattr(torch.version, "cuda", None) or "cpu-only build"
    facts["cuda_available"] = bool(torch.cuda.is_available())
    if not torch.cuda.is_available():
        return facts

    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    total_bytes = int(properties.total_memory)
    facts.update(
        {
            "device_index": int(index),
            "device_name": str(properties.name),
            "capability": f"sm_{properties.major}{properties.minor}",
            "vram_total_gib": round(total_bytes / 1024**3, 2),
            "vram_free_gib": round(
                (total_bytes - int(torch.cuda.memory_reserved(index))) / 1024**3, 2
            ),
        }
    )
    for name in ("transformers", "peft", "trl", "datasets", "accelerate", "bitsandbytes"):
        if _importable(name):
            facts[name] = _module_version(name)
    return facts


def _module_version(module: str) -> str:
    """Best-effort ``module.__version__``, or a marker when it has none."""
    try:
        imported = __import__(module)
    except Exception:  # a probe must never take the caller down
        return "import failed"
    return str(getattr(imported, "__version__", "unknown"))


def check_device_compatibility() -> list[str]:
    """Human-readable warnings about the configured training device.

    Returns an empty list on a machine that would actually run. The Blackwell check is
    the one that matters: an sm_120 card with a pre-cu128 wheel imports fine, reports
    CUDA available, and then fails inside the first kernel with an error that does not
    mention the wheel at all.
    """
    warnings: list[str] = []
    missing = missing_required_modules()
    if missing:
        return [f"the training stack is incomplete: missing {', '.join(missing)}"]
    if not _importable("torch"):
        return ["torch is not installed"]

    import torch

    if not torch.cuda.is_available():
        warnings.append(
            "no CUDA device is visible to torch. A 14B 4-bit fine-tune on CPU is not a "
            "slower version of this run, it is a different and useless one — see "
            "docs/06-windows-setup.md."
        )
        return warnings

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    capability = (int(properties.major), int(properties.minor))
    cuda_build = str(getattr(torch.version, "cuda", "") or "")
    if capability >= _BLACKWELL_CAPABILITY and cuda_build:
        try:
            build_major, build_minor = (int(part) for part in cuda_build.split(".")[:2])
        except ValueError:
            build_major, build_minor = (0, 0)
        if (build_major, build_minor) < (12, 8):
            warnings.append(
                f"{properties.name} is sm_{capability[0]}{capability[1]} (Blackwell) but "
                f"this torch was built against CUDA {cuda_build}. Install the cu128 "
                "wheel or the first attention kernel will fail with 'no kernel image is "
                "available for execution on the device'."
            )

    total_gib = properties.total_memory / 1024**3
    if total_gib < 24:
        warnings.append(
            f"the device reports {total_gib:.1f} GiB of VRAM. A 14B 4-bit fine-tune at "
            "4096 tokens with gradient checkpointing needs roughly one 24 GB card; "
            "expect to reduce lora.max_seq_length or move to an 8B base model."
        )
    return warnings


# --------------------------------------------------------------------------- #
# Leakage guard
# --------------------------------------------------------------------------- #


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield each JSON object in a JSONL file, skipping blank lines.

    Raises:
        ValueError: If a line is not a JSON object. Reported with the line number,
            because a corrupted training file is otherwise indistinguishable from an
            empty one.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number} is not valid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{number} is not a JSON object")
            yield record


def assert_no_test_rows(path: Path) -> dict[str, int]:
    """Refuse a training file that contains rows from a split it must not use.

    Args:
        path: An SFT JSONL written by ``shingan data sft``.

    Returns:
        Counts per split actually seen, so the caller can record the composition.

    Raises:
        LoraLeakageError: If any row's ``meta.split`` is outside
            :data:`PERMITTED_TRAIN_SPLITS`, or if a row carries no split at all. A
            missing split is refused rather than assumed to be ``train``: the whole
            check rests on the label being present, so an absent one has to fail.
    """
    counts: dict[str, int] = {}
    offenders: dict[str, int] = {}
    for record in iter_jsonl(path):
        meta = record.get("meta")
        split = str(meta.get("split", "")) if isinstance(meta, dict) else ""
        if split not in PERMITTED_TRAIN_SPLITS:
            offenders[split or "<absent>"] = offenders.get(split or "<absent>", 0) + 1
        counts[split or "<absent>"] = counts.get(split or "<absent>", 0) + 1

    if offenders:
        raise LoraLeakageError(
            f"{path} contains rows from splits {sorted(offenders)} with counts "
            f"{offenders}; a training file may only contain "
            f"{sorted(PERMITTED_TRAIN_SPLITS)}. Rebuild it with `shingan data sft`, "
            "which does not emit the test block."
        )
    return counts


def assert_packing_disabled(config: LoraConfig) -> None:
    """Refuse ``packing: true``.

    Raises:
        LoraLeakageError: If packing is enabled.
    """
    if config.packing:
        raise LoraLeakageError(
            "lora.packing is true. Packing concatenates samples across prompt "
            "boundaries; every example here carries its own label and horizon, so "
            "packing corrupts supervision while making throughput look better. Set "
            "lora.packing: false (docs/04-training.md section 2.4)."
        )


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class LoraRunResult:
    """What one fine-tuning run produced."""

    output_dir: Path
    base_model: str
    n_train_examples: int
    n_eval_examples: int
    eval_metrics: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    split_counts: dict[str, int] = field(default_factory=dict)
    adapter_dir: Path | None = None
    run_json: Path | None = None


def train_lora(
    config: LoraConfig,
    train_file: Path | str,
    *,
    eval_file: Path | str | None = None,
    output_dir: Path | str | None = None,
    paths: Any = None,
    seed: int | None = None,
) -> LoraRunResult:
    """QLoRA fine-tune the text track.

    Args:
        config: The ``lora`` block of the project configuration.
        train_file: SFT JSONL of training-fold rows.
        eval_file: SFT JSONL of validation-fold rows, used for per-epoch evaluation.
            Optional: without it there is no epoch selection, which this function
            reports as a warning rather than silently picking the last epoch.
        output_dir: Where checkpoints and ``run.json`` are written. Defaults to
            ``config.output_dir``.
        paths: A :class:`~shingan.paths.ProjectPaths`, used only to resolve a relative
            ``output_dir`` against the project root.
        seed: Overrides ``project.seed``. Recorded in ``run.json`` so a run is
            reproducible from its own artifact.

    Returns:
        The :class:`LoraRunResult`.

    Raises:
        MissingTrainDependencies: If the ``train`` extra is not importable.
        LoraLeakageError: If ``config.packing`` is true, or the training file holds
            rows from a split outside :data:`PERMITTED_TRAIN_SPLITS`.
        FileNotFoundError: If ``train_file`` does not exist.
    """
    missing = missing_required_modules()
    if missing:
        raise MissingTrainDependencies(missing)

    assert_packing_disabled(config)

    source = Path(train_file)
    if not source.is_file():
        raise FileNotFoundError(f"training file not found: {source}")
    split_counts = assert_no_test_rows(source)

    destination = _resolve_output_dir(config, output_dir, paths)
    destination.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    evaluation = Path(eval_file) if eval_file is not None else None
    if evaluation is not None and not evaluation.is_file():
        warnings.append(f"the validation file {evaluation} does not exist; ignored")
        evaluation = None
    if evaluation is None:
        warnings.append(
            "no validation file was supplied, so no per-epoch evaluation ran and epoch "
            "selection is not possible. The last epoch is what ships — quote the "
            "held-out test block from `shingan eval run` rather than a validation number."
        )
    if evaluation is not None:
        eval_counts = assert_no_test_rows(evaluation)
        if eval_counts.get("test"):
            raise LoraLeakageError(f"{evaluation} contains test-fold rows")

    for warning in check_device_compatibility():
        logger.warning("%s", warning)

    import torch
    from datasets import Dataset
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    # LoraConfig carries no seed of its own: the project seed is the single source of
    # randomness in this package, and the CLI passes it in. Zero here means the caller
    # did not supply one, which is recorded in run.json rather than hidden.
    resolved_seed = int(seed) if seed is not None else 0
    torch.manual_seed(resolved_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved_seed)

    quantization = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
        )
        if config.load_in_4bit
        else None
    )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=quantization,
        device_map="auto" if config.device_map == "auto" else None,
        torch_dtype=getattr(torch, "bfloat16" if config.bf16 else "float32"),
        attn_implementation=config.attn_implementation,
    )
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.gradient_checkpointing
        )

    peft_config = PeftLoraConfig(
        r=int(config.lora_r),
        lora_alpha=int(config.lora_alpha),
        lora_dropout=float(config.lora_dropout),
        target_modules=list(config.target_modules),
        bias=config.bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    train_dataset = Dataset.from_list(list(iter_jsonl(source)))
    eval_dataset = Dataset.from_list(list(iter_jsonl(evaluation))) if evaluation else None

    training_arguments = SFTConfig(
        output_dir=str(destination),
        max_length=int(config.max_seq_length),
        packing=False,
        learning_rate=float(config.learning_rate),
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=float(config.warmup_ratio),
        num_train_epochs=float(config.num_train_epochs),
        per_device_train_batch_size=int(config.per_device_train_batch_size),
        per_device_eval_batch_size=int(config.per_device_eval_batch_size),
        gradient_accumulation_steps=int(config.gradient_accumulation_steps),
        gradient_checkpointing=bool(config.gradient_checkpointing),
        optim=config.optim,
        bf16=bool(config.bf16),
        fp16=bool(config.fp16),
        max_grad_norm=float(config.max_grad_norm),
        logging_steps=int(config.logging_steps),
        save_total_limit=int(config.save_total_limit),
        eval_strategy=config.eval_strategy if eval_dataset is not None else "no",
        report_to=config.report_to,
        dataloader_num_workers=0,
        seed=resolved_seed,
        data_seed=resolved_seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(destination / "adapter"))
    tokenizer.save_pretrained(str(destination / "adapter"))

    eval_metrics = {
        key: float(value)
        for key, value in (trainer.state.log_history or [{}])[-1].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }

    result = LoraRunResult(
        output_dir=destination,
        base_model=str(config.base_model),
        n_train_examples=len(train_dataset),
        n_eval_examples=0 if eval_dataset is None else len(eval_dataset),
        eval_metrics=eval_metrics,
        warnings=warnings,
        split_counts=split_counts,
        adapter_dir=destination / "adapter",
    )
    result.run_json = _write_run_json(result, config, resolved_seed)
    logger.info("LoRA run finished; adapter at %s", result.adapter_dir)
    return result


def _resolve_output_dir(
    config: LoraConfig, output_dir: Path | str | None, paths: Any
) -> Path:
    """Resolve where a run writes, anchoring a relative path at the project root."""
    if output_dir is not None:
        candidate = Path(output_dir)
    else:
        candidate = Path(config.output_dir)
    if candidate.is_absolute():
        return candidate
    root = getattr(paths, "root", None)
    return (Path(root) / candidate) if root is not None else candidate.resolve()


def _write_run_json(result: LoraRunResult, config: LoraConfig, seed: int) -> Path:
    """Write the run's own record next to the adapter.

    Follows ``shingan eval run``'s rule: an artifact without its configuration is
    unreadable a month later, so the config travels with the checkpoint.
    """
    payload = {
        "base_model": result.base_model,
        "seed": seed,
        "n_train_examples": result.n_train_examples,
        "n_eval_examples": result.n_eval_examples,
        "split_counts": result.split_counts,
        "eval_metrics": result.eval_metrics,
        "warnings": result.warnings,
        "environment": describe_train_environment(),
        "config": config.model_dump(mode="json"),
        "packing": False,
        "note": (
            "the reported epoch metrics are selection metrics evaluated on the same "
            "fold the epoch was chosen on; quote the held-out test block from "
            "`shingan eval run` instead"
        ),
    }
    path = result.output_dir / "run.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


__all__ = [
    "INSTALL_HINT",
    "OPTIONAL_MODULES",
    "PERMITTED_TRAIN_SPLITS",
    "REQUIRED_MODULES",
    "LoraLeakageError",
    "LoraRunResult",
    "MissingTrainDependencies",
    "assert_no_test_rows",
    "assert_packing_disabled",
    "check_device_compatibility",
    "dependency_report",
    "describe_train_environment",
    "iter_jsonl",
    "missing_required_modules",
    "train_lora",
]
