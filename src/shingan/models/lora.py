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

A third boundary is enforced here: **the mapping from configuration to trainer
arguments.** The trainer's configuration class is a separately versioned API and this
project's YAML is a description of intent, so the two can disagree — `transformers` v5
removed `warmup_ratio` outright, and the symptom was a `TypeError` raised after a 25 GB
base-model download. The mapping is a pure function (see "Trainer arguments") and the
keys it produces are checked against the installed class before the model is touched.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import inspect
import json
import math
import platform
import sys
from collections.abc import Iterable, Iterator, Mapping
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
    # The versions of the training stack, probed before the torch/CUDA branches below so
    # they are recorded on a machine with no GPU too. A keyset one major accepts another
    # rejects (see "Trainer arguments"), so a run without these cannot be replayed — nor
    # can a failure like `warmup_ratio` be explained after the fact. Kept as flat keys and
    # read from the imported module rather than from installed metadata, because the
    # module is what the trainer actually executes.
    for name in ("transformers", "peft", "trl", "datasets", "accelerate", "bitsandbytes"):
        if _importable(name):
            facts[name] = _module_version(name)

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
# Trainer arguments
# --------------------------------------------------------------------------- #
#
# The project configuration is a statement of intent; the trainer's configuration class
# is a versioned API. The two drift, and they drift in the direction that costs the most:
# a key the installed library no longer accepts raises `TypeError` when the trainer is
# built, which here is *after* the base model has been downloaded and quantised. One full
# run was lost to exactly that — `transformers` v5 removed `warmup_ratio` outright.
#
# Everything below exists to move that boundary somewhere cheap and testable: the
# mapping from config to arguments is a pure function, and the check against the
# installed API runs before anything expensive is imported.

#: Major version of ``transformers`` in which ``torch_dtype`` became ``dtype``. The old
#: spelling still works there but emits a deprecation warning, and a run that prints
#: warnings it can avoid trains the reader to ignore warnings.
_DTYPE_RENAMED_IN = 5


class TrainingStackMismatch(RuntimeError):
    """The installed training libraries do not accept this project's arguments.

    Deliberately not a :class:`MissingTrainDependencies`: that one means "install the
    extra", this one means "the extra is installed and is a build this package cannot
    drive". Reporting the second as the first would send the user to reinstall a stack
    that is already present and correct.
    """


def library_versions(*names: str) -> dict[str, str]:
    """Installed distribution versions for ``names``, without importing them.

    Uses ``importlib.metadata``, which reads installed metadata rather than executing the
    package, so this stays inside this module's promise of never importing the ``train``
    extra at import time. An absent distribution reports ``not installed`` instead of
    raising: the caller is usually already reporting an absence.

    Returns:
        ``{distribution: version}``, in the order asked for.
    """
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def parse_major(version: str) -> int | None:
    """Leading integer of a version string, or ``None`` if there is not one.

    ``None`` rather than 0, because "this version cannot be classified" has to branch
    differently from "this version is old": the callers pick the spelling that works on
    both sides of a rename when the version is unknown.
    """
    head = version.split("+", 1)[0].split(".", 1)[0].strip()
    return int(head) if head.isdigit() else None


def dtype_keyword(transformers_version: str) -> str:
    """The weight-dtype argument ``from_pretrained`` wants on this version.

    ``transformers`` renamed ``torch_dtype`` to ``dtype`` in v5 and kept the old name as a
    deprecated alias. Both spellings load the same weights on v5; only ``torch_dtype``
    works on v4. So the old name is what an unparseable version gets — it is the spelling
    that is valid on both sides.
    """
    major = parse_major(transformers_version)
    if major is not None and major >= _DTYPE_RENAMED_IN:
        return "dtype"
    return "torch_dtype"


def total_optimizer_steps(
    n_examples: int,
    *,
    batch_size: int,
    accumulate: int,
    epochs: int,
    world_size: int = 1,
) -> int:
    """Optimiser steps a run will take, by the trainer's own arithmetic.

    Spelled out as the two ceilings the trainer applies — the dataloader yields
    ``ceil(n / (batch * world))`` batches per epoch and a step is taken every
    ``accumulate`` batches — because that order is what a reader is checking this number
    against. The composition is provably equal to ``ceil(n / (batch * world * accumulate))``;
    it is written this way to be traceable, not because the rounding differs.
    """
    batches = math.ceil(n_examples / max(1, batch_size * world_size))
    return math.ceil(batches / max(1, accumulate)) * max(1, epochs)


def resolve_warmup_steps(config: LoraConfig, n_train_examples: int, *, world_size: int = 1) -> int:
    """``lora.warmup_ratio`` expressed as the absolute step count the trainer takes.

    ``transformers`` v5 removed ``warmup_ratio``, so the ratio has to be resolved against
    the length of the run. Resolving it here rather than writing a step count into the
    YAML keeps the configuration readable *and* keeps the intent correct when the batch
    size or the epoch count changes, which a frozen step count would silently not.

    Rounding is half-up, not the built-in ``round``: a 0.5-step warmup is warmup the
    config asked for, and banker's rounding would drop it to none.

    Returns:
        The step count, possibly 0 on a run short enough that the ratio is under half a
        step. 0 is the trainer's way of saying "no warmup", which is the truthful
        rendering of "3% of three steps".
    """
    steps = total_optimizer_steps(
        n_train_examples,
        batch_size=config.per_device_train_batch_size,
        accumulate=config.gradient_accumulation_steps,
        epochs=config.num_train_epochs,
        world_size=world_size,
    )
    return math.floor(float(config.warmup_ratio) * steps + 0.5)


def accepted_arguments(target: Any) -> set[str] | None:
    """Keyword names ``target`` accepts, or ``None`` if it accepts anything.

    ``None`` is the honest answer for a callable with ``**kwargs``: unknown keys are then
    not rejected, they are silently ignored, which is a different failure mode and not
    one this check can detect.
    """
    parameters = inspect.signature(target.__init__).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return None
    return {
        name
        for name, parameter in parameters.items()
        if name != "self"
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }


def unsupported_arguments(arguments: Mapping[str, Any], target: Any) -> list[str]:
    """Argument names in ``arguments`` that ``target`` would reject, sorted.

    Empty when ``target`` accepts arbitrary keywords, since nothing can be rejected.
    """
    accepted = accepted_arguments(target)
    if accepted is None:
        return []
    return sorted(set(arguments) - accepted)


def assert_trainer_arguments_supported(
    arguments: Mapping[str, Any], target: Any, *, label: str | None = None
) -> None:
    """Refuse to start when the installed libraries cannot take these arguments.

    This is the check that converts "25 GB of weights downloaded, then a ``TypeError``"
    into an immediate failure that names the offending keys and the versions in play.

    Args:
        arguments: The keyword arguments about to be passed.
        target: The class that will receive them.
        label: Name to report for ``target``. Defaults to its ``__name__``.

    Raises:
        TrainingStackMismatch: If ``target`` would reject any key.
    """
    unsupported = unsupported_arguments(arguments, target)
    if not unsupported:
        return
    name = label or getattr(target, "__name__", None) or str(target)
    versions = library_versions("transformers", "trl", "peft")
    installed = ", ".join(f"{key}={value}" for key, value in versions.items())
    raise TrainingStackMismatch(
        f"{name} does not accept {', '.join(unsupported)}. Installed: {installed}. "
        "The trainer API is versioned separately from this project's configuration, so a "
        "key can disappear in a major release; `pip install -e \".[train]\"` reinstalls "
        "the pinned stack. See docs/04-training.md section 2.6."
    )


def build_sft_config_kwargs(
    config: LoraConfig,
    *,
    output_dir: Path | str,
    n_train_examples: int,
    eval_enabled: bool,
    seed: int,
) -> dict[str, Any]:
    """Map the ``lora`` config block onto the trainer's configuration class.

    A pure function of the config and the run's size, so it can be tested on a machine
    with no training stack installed — which matters, because the assertion that this
    mapping stays inside the installed API surface must not itself depend on that API
    being importable.

    Two values are constants rather than config reads. ``packing`` is refused upstream by
    :func:`assert_packing_disabled`; passing it as a literal here is the second lock on
    the same door and closes the gap between the check and the call. ``dataloader_num_workers``
    is a Windows constraint (no fork), and allowing a hand-edited YAML to raise it would
    reintroduce exactly the failure the overlay's comment describes.
    """
    return {
        "output_dir": str(output_dir),
        "max_length": int(config.max_seq_length),
        "packing": False,
        # `warmup_ratio` does not exist on transformers >= 5; the ratio is resolved
        # against this run's actual length instead.
        "warmup_steps": resolve_warmup_steps(config, n_train_examples),
        "learning_rate": float(config.learning_rate),
        "lr_scheduler_type": config.lr_scheduler_type,
        "num_train_epochs": float(config.num_train_epochs),
        "per_device_train_batch_size": int(config.per_device_train_batch_size),
        "per_device_eval_batch_size": int(config.per_device_eval_batch_size),
        "gradient_accumulation_steps": int(config.gradient_accumulation_steps),
        "gradient_checkpointing": bool(config.gradient_checkpointing),
        "optim": config.optim,
        "bf16": bool(config.bf16),
        "fp16": bool(config.fp16),
        "max_grad_norm": float(config.max_grad_norm),
        "logging_steps": int(config.logging_steps),
        "save_total_limit": int(config.save_total_limit),
        "eval_strategy": config.eval_strategy if eval_enabled else "no",
        "report_to": config.report_to,
        "dataloader_num_workers": 0,
        "seed": seed,
        "data_seed": seed,
    }


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
    #: The keyword arguments the trainer was actually constructed with, after the config
    #: was mapped. Recorded because the config alone does not say what ran: a key can be
    #: translated (`warmup_ratio` -> `warmup_steps`) or dropped by an older library.
    trainer_arguments: dict[str, Any] = field(default_factory=dict)


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
    import transformers
    from datasets import Dataset
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    # LoraConfig carries no seed of its own: the project seed is the single source of
    # randomness in this package, and the CLI passes it in. Zero here means the caller
    # did not supply one, which is recorded in run.json rather than hidden.
    resolved_seed = int(seed) if seed is not None else 0

    # -- cheap first -------------------------------------------------------------
    # Both datasets are read, the arguments are mapped, and the mapped keys are checked
    # against the installed trainer API *before* a single weight is loaded. Reading two
    # JSONL files costs milliseconds; discovering a renamed keyword after
    # `from_pretrained` costs the download. That ordering is the fix for a run that died
    # at 2h05m on `warmup_ratio`.
    train_dataset = Dataset.from_list(list(iter_jsonl(source)))
    eval_dataset = Dataset.from_list(list(iter_jsonl(evaluation))) if evaluation else None

    trainer_arguments = build_sft_config_kwargs(
        config,
        output_dir=destination,
        n_train_examples=len(train_dataset),
        eval_enabled=eval_dataset is not None,
        seed=resolved_seed,
    )
    assert_trainer_arguments_supported(trainer_arguments, SFTConfig, label="SFTConfig")
    training_arguments = SFTConfig(**trainer_arguments)

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

    # The weight-dtype keyword was renamed in transformers v5 (`dtype`, with `torch_dtype`
    # kept as a deprecated alias). Both spellings work there; only the old one works on
    # v4. So the choice is made from the installed version rather than assumed — this is
    # the one argument that cannot be written once and left alone.
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=quantization,
        device_map="auto" if config.device_map == "auto" else None,
        attn_implementation=config.attn_implementation,
        **{
            dtype_keyword(transformers.__version__): getattr(
                torch, "bfloat16" if config.bf16 else "float32"
            )
        },
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
        trainer_arguments=trainer_arguments,
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
        "trainer_arguments": result.trainer_arguments,
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
    "TrainingStackMismatch",
    "assert_no_test_rows",
    "assert_packing_disabled",
    "assert_trainer_arguments_supported",
    "build_sft_config_kwargs",
    "check_device_compatibility",
    "dependency_report",
    "describe_train_environment",
    "dtype_keyword",
    "iter_jsonl",
    "library_versions",
    "missing_required_modules",
    "resolve_warmup_steps",
    "total_optimizer_steps",
    "train_lora",
]
