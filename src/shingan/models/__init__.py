"""The three model layers, and the split of responsibility between them.

Layout:

* :mod:`shingan.models.structured` — Track A. A calibrated gradient-boosted model over
  the panel's numeric features.
* :mod:`shingan.models.text_baseline` — Track B0. TF-IDF over the same rendered prompt
  the LoRA sees, so the fine-tuned LLM has a baseline it must beat to justify its cost.
* :mod:`shingan.models.fusion` — Track C. The stacker that combines the two scores, and
  the ablation table that shows what each contributed.
* :mod:`shingan.models.lora` — Track B. QLoRA fine-tuning. Imported here because it is
  torch-free *at import time* — every torch import inside it is function-local, which is
  what lets ``shingan doctor`` report the training stack as missing instead of crashing.

The import graph is deliberately shallow. Nothing in this package imports
:mod:`shingan.pipeline`; the pipeline imports these modules and passes them the split
frames it built, so the one-fit/one-apply rule in
:mod:`shingan.pipeline` is enforceable in one place rather than three.
"""

from shingan.models.fusion import (
    STRUCTURED_SCORE,
    TEXT_SCORE,
    RiskFusion,
    cross_sectional_rank,
    fusion_ablation,
)
from shingan.models.lora import (
    INSTALL_HINT,
    LoraLeakageError,
    LoraRunResult,
    MissingTrainDependencies,
    TrainingStackMismatch,
    check_device_compatibility,
    dependency_report,
    describe_train_environment,
    missing_required_modules,
    train_lora,
)
from shingan.models.structured import StructuredRiskModel
from shingan.models.text_baseline import TextBaselineModel

__all__ = [
    "INSTALL_HINT",
    "STRUCTURED_SCORE",
    "TEXT_SCORE",
    "LoraLeakageError",
    "LoraRunResult",
    "MissingTrainDependencies",
    "RiskFusion",
    "StructuredRiskModel",
    "TextBaselineModel",
    "TrainingStackMismatch",
    "check_device_compatibility",
    "cross_sectional_rank",
    "dependency_report",
    "describe_train_environment",
    "fusion_ablation",
    "missing_required_modules",
    "train_lora",
]
