"""Scoring with a fine-tuned adapter: the missing half of the text track.

``models/lora.py`` trains an adapter, and ``prompts.py`` can parse the output that
adapter was trained to produce — but nothing connected the two. The consequence was
not a missing feature, it was a missing answer: the project's central question
(whether the text track adds measurable information beyond the structured signals)
could not be asked of any artifact in the repository, because no command ever scored
the adapter. ``shingan eval run`` fits structured, TF-IDF and fusion, and stops there.

This module is that connection. It is split so that everything testable without a GPU
is tested without one:

* :func:`adapter_facts` hashes what the adapter directory contains. Its identity has
  to travel with the numbers, or a score cannot be traced back to a file.
* :func:`score_generation` turns one raw generation into a score, through the same
  parser the data builder validates its targets with. No second parser exists.
* :func:`load_for_inference` and :func:`generate_texts` are the only functions that
  import torch, and therefore the only ones CI cannot exercise.

The tokenizer question deserves the explicit note it gets on
:data:`TOKENIZER_MARKERS`, because it is silent when it goes wrong.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shingan.data.schema import RiskAssessment

# `_importable` is private to `lora.py` and imported deliberately: there should be one
# implementation of "can this module be imported without importing it", because a second
# probe is a second answer. Still not re-exported.
from shingan.models.lora import INSTALL_HINT, _importable, dtype_keyword
from shingan.prompts import parse_assessment

logger = logging.getLogger(__name__)

#: Importable modules inference needs. Deliberately *not* ``REQUIRED_MODULES``: that
#: tuple includes ``trl``, ``datasets`` and ``accelerate``, which are training-only. A
#: machine that can score an adapter but not train one is a legitimate machine — asking
#: it for the whole training extra would be a wrong requirement stated confidently.
INFERENCE_MODULES: tuple[str, ...] = ("torch", "transformers", "peft", "bitsandbytes")

#: Files whose presence marks a directory as carrying its own tokenizer. The adapter
#: directory was written by the trainer and holds the tokenizer actually used, which is
#: not necessarily the one on the base model: the training log recorded
#: ``Updated tokens: {'bos_token_id': None}``, so the two disagree about BOS. Scoring
#: with the base model's tokenizer would silently change the token sequence from the one
#: the weights were fitted on.
TOKENIZER_MARKERS: tuple[str, ...] = ("tokenizer_config.json", "tokenizer.json")

#: Weight files an adapter may be stored in, in preference order.
ADAPTER_WEIGHT_FILES: tuple[str, ...] = ("adapter_model.safetensors", "adapter_model.bin")


class MissingInferenceDependencies(RuntimeError):
    """The stack needed to score an adapter is not importable.

    Separate from :class:`~shingan.models.lora.MissingTrainDependencies` so that a run
    that only scores does not send the user to install a trainer it will never call.
    """


def missing_inference_modules() -> list[str]:
    """Inference modules that are not importable, in :data:`INFERENCE_MODULES` order."""
    return [name for name in INFERENCE_MODULES if not _importable(name)]


@dataclass(slots=True)
class AdapterFacts:
    """What the adapter directory says about itself, plus what it hashes to."""

    directory: Path
    base_model: str
    weights_file: str
    weights_sha256: str
    weights_bytes: int
    config_sha256: str
    rank: int | None = None
    alpha: int | None = None
    target_modules: list[str] = field(default_factory=list)
    #: ``"adapter directory"`` or ``"base model"``. Recorded because the two tokenizers
    #: can differ, and a scored run that does not say which one it used cannot be
    #: reproduced.
    tokenizer_source: str = "unknown"

    def as_dict(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory),
            "base_model": self.base_model,
            "weights_file": self.weights_file,
            "weights_sha256": self.weights_sha256,
            "weights_bytes": self.weights_bytes,
            "adapter_config_sha256": self.config_sha256,
            "rank": self.rank,
            "alpha": self.alpha,
            "target_modules": list(self.target_modules),
            "tokenizer_source": self.tokenizer_source,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_facts(adapter_dir: Path) -> AdapterFacts:
    """Read and hash an adapter directory.

    Args:
        adapter_dir: Directory written by ``shingan train lora``.

    Returns:
        The facts, including the SHA-256 of the weights and of ``adapter_config.json``.

    Raises:
        FileNotFoundError: If the directory, its config, or its weights are absent. A
            missing directory is a typing mistake, and reporting it here beats letting
            the Hub client try to resolve the path as a repository id.
    """
    directory = Path(adapter_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"adapter directory not found: {directory}")
    config_path = directory / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"not an adapter directory (no adapter_config.json): {directory}")
    weights_path = next(
        (directory / name for name in ADAPTER_WEIGHT_FILES if (directory / name).is_file()),
        None,
    )
    if weights_path is None:
        raise FileNotFoundError(
            f"no weights in {directory}: expected one of {', '.join(ADAPTER_WEIGHT_FILES)}"
        )

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    base_model = str(payload.get("base_model_name_or_path") or "")
    if not base_model:
        raise ValueError(
            f"{config_path} carries no base_model_name_or_path, so the base weights "
            "cannot be located. Scoring needs both halves of the model."
        )
    has_tokenizer = any((directory / name).is_file() for name in TOKENIZER_MARKERS)

    return AdapterFacts(
        directory=directory,
        base_model=base_model,
        weights_file=weights_path.name,
        weights_sha256=_sha256(weights_path),
        weights_bytes=weights_path.stat().st_size,
        config_sha256=_sha256(config_path),
        rank=payload.get("r"),
        alpha=payload.get("lora_alpha"),
        target_modules=list(payload.get("target_modules") or []),
        tokenizer_source="adapter directory" if has_tokenizer else "base model",
    )


@dataclass(slots=True)
class ScoreAttempt:
    """One generation, reduced to what a metric can consume.

    ``score`` is None when the generation was rejected, and ``reason`` says why. The
    distinction matters at the report level: a run where 60% of generations were
    rejected produces numbers that are arithmetic on a biased subsample, and that has
    to be visible next to the numbers rather than discoverable later.
    """

    score: float | None
    assessment: RiskAssessment | None = None
    reason: str = ""

    @property
    def parsed(self) -> bool:
        return self.score is not None


def score_generation(
    text: str,
    *,
    expected_label: str | None = None,
    expected_horizon_days: int | None = None,
) -> ScoreAttempt:
    """Parse one raw generation into a score, or explain the rejection.

    Args:
        text: Raw model output.
        expected_label: The label that was asked for. A model that answers a different
            label has answered a different question, and its score is not comparable
            with the others in the same metric.
        expected_horizon_days: The horizon that was asked for.

    Returns:
        A :class:`ScoreAttempt`. Never raises: one malformed generation out of hundreds
        must not end the run.
    """
    reasons: list[str] = []
    assessment = parse_assessment(
        text,
        expected_label=expected_label,
        expected_horizon_days=expected_horizon_days,
        failure_reason=reasons,
    )
    if assessment is None:
        return ScoreAttempt(score=None, reason=reasons[-1] if reasons else "rejected")
    return ScoreAttempt(score=float(assessment.score), assessment=assessment)


def load_for_inference(
    adapter_dir: Path,
    *,
    device_map: str = "auto",
    attn_implementation: str = "sdpa",
    load_in_4bit: bool = True,
    compute_dtype: str = "bfloat16",
) -> tuple[Any, Any, AdapterFacts]:
    """Load base weights plus the adapter, and the tokenizer the adapter was trained with.

    The quantization settings mirror training (4-bit NF4 with double quantisation), so
    that the adapter is evaluated on the same numerical substrate it was fitted on.

    Args:
        adapter_dir: Directory written by ``shingan train lora``.
        device_map: Passed to ``from_pretrained``. ``"auto"`` places the 4-bit weights
            on the GPU when one is visible.
        attn_implementation: Attention kernel. ``sdpa`` matches the training config.
        load_in_4bit: Quantise the base weights. Off needs ~28 GB of VRAM for a 14B
            model in bf16.
        compute_dtype: Dtype for the quantized matmuls.

    Returns:
        ``(model, tokenizer, facts)``.

    Raises:
        MissingInferenceDependencies: If the stack is absent.
    """
    missing = missing_inference_modules()
    if missing:
        raise MissingInferenceDependencies(
            f"cannot score an adapter: missing {', '.join(missing)}.\n{INSTALL_HINT}"
        )

    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    facts = adapter_facts(adapter_dir)
    # The tokenizer comes from the adapter directory when it has one — see
    # TOKENIZER_MARKERS. `facts.tokenizer_source` records which happened.
    tokenizer_path = (
        str(facts.directory) if facts.tokenizer_source == "adapter directory" else facts.base_model
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    quantization = None
    if load_in_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=getattr(torch, compute_dtype),
        )

    model = AutoModelForCausalLM.from_pretrained(
        facts.base_model,
        quantization_config=quantization,
        device_map=device_map if device_map != "none" else None,
        attn_implementation=attn_implementation,
        **{dtype_keyword(transformers.__version__): getattr(torch, compute_dtype)},
    )
    model = PeftModel.from_pretrained(model, str(facts.directory))
    model.eval()
    logger.info(
        "loaded adapter %s (base %s, tokenizer from %s)",
        facts.directory,
        facts.base_model,
        facts.tokenizer_source,
    )
    return model, tokenizer, facts


def generate_texts(
    model: Any,
    tokenizer: Any,
    conversations: Sequence[list[dict[str, str]]],
    *,
    max_new_tokens: int = 512,
    batch_size: int = 4,
    temperature: float = 0.0,
    progress: bool = False,
) -> list[str]:
    """Generate one response per conversation, in order.

    Greedy by default. A sampled score is a different quantity from a greedy one and
    the two are not comparable across runs, so sampling has to be opted into.

    Args:
        model: A ``PeftModel`` on an eval-mode ``AutoModelForCausalLM``.
        tokenizer: The tokenizer those weights were fitted with.
        conversations: One ``messages`` list per row, as built by
            :func:`shingan.prompts.build_chat_messages`.
        max_new_tokens: Generation cap. The trained targets are single-line JSON
            around 200 tokens; the cap is loose so a verbose failure is visible as a
            parse error rather than as a truncated but plausible object.
        batch_size: Rows per forward pass. Lower it if the card runs out of memory.
        temperature: 0 for greedy. Anything else enables sampling.
        progress: Log a line per batch.

    Returns:
        The decoded continuations, with the prompt removed.
    """
    import torch

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left padding for decoder-only generation
    do_sample = temperature > 0
    if not do_sample:
        temperature = 1.0

    outputs: list[str] = []
    for start in range(0, len(conversations), batch_size):
        batch = conversations[start : start + batch_size]
        rendered = [
            tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True)
            for item in batch
        ]
        encoded = tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        prompt_length = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
            )
        outputs.extend(
            tokenizer.decode(row[prompt_length:], skip_special_tokens=True) for row in generated
        )
        if progress:
            done = min(start + batch_size, len(conversations))
            logger.info("generated %d/%d", done, len(conversations))
    return outputs
