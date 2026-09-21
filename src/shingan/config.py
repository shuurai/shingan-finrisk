"""Typed configuration.

Configuration is YAML in ``configs/``, validated here with pydantic. Three
properties matter more than convenience:

1. **Unknown keys are errors.** Every model forbids extra fields, so a typo in a
   YAML key fails loudly at load time instead of silently training with a default.
2. **Layering is explicit.** ``configs/default.yaml`` is the base; the data, train
   and eval files are overlays that are deep-merged onto it. This is what lets the
   same pipeline run on ten companies or on the full index without forking code.
3. **The label horizons are the single source of truth for the split margin.**
   ``purge_days`` is not allowed to be smaller than the longest label horizon in
   use, because that would quietly re-introduce the leakage the split exists to
   prevent.
"""

from __future__ import annotations

import copy
import os
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from shingan.__about__ import DATA_SCHEMA_VERSION
from shingan.logging_utils import get_logger
from shingan.paths import ProjectPaths
from shingan.seed import DEFAULT_SEED

if TYPE_CHECKING:
    # Types only, never imported at runtime. `shingan.data.synthetic` imports this
    # module, and importing `shingan.data.schema` would run the `shingan.data`
    # package initialiser → synthetic → back here, while this module is still
    # half-built. The annotations below are strings (`from __future__ import
    # annotations`), so a type-checking-only import is both sufficient and the
    # accurate description of the dependency.
    from shingan.data.schema import RiskLabel

logger = get_logger(__name__)

RiskLabelName = Literal["default_risk", "fraud_risk", "tail_risk"]
DataSourceName = Literal[
    "synthetic",
    "edgar",
    "fnspid",
    "prices_yfinance",
    "prices_stooq",
]
FeatureGroupName = Literal["ratios", "technical", "text_counts"]

# Typed defaults. Spelled as module-level tuples rather than inline literals so
# that ``default_factory=list(_DEFAULT_TARGETS)`` keeps the element type: an
# inline ``lambda: ["default_risk", ...]`` is inferred as ``list[str]`` and fails
# validation of the declared ``list[RiskLabelName]``.
_DEFAULT_TARGETS: tuple[RiskLabelName, ...] = ("default_risk", "fraud_risk", "tail_risk")
_DEFAULT_SOURCES: tuple[DataSourceName, ...] = ("synthetic",)
_DEFAULT_FEATURE_GROUPS: tuple[FeatureGroupName, ...] = ("ratios", "technical", "text_counts")

#: Files that make up the default configuration, in merge order. Later files win.
DEFAULT_CONFIG_FILES: tuple[str, ...] = (
    "configs/default.yaml",
    "configs/data/poc_largecap10.yaml",
    "configs/eval/default.yaml",
)

#: Overlay that raises the synthetic event rates enough to exercise every label end to
#: end. Deliberately *not* part of the default stack: its base rates are not the ones
#: the label documentation specifies, and a report produced with it is a demonstration
#: of the pipeline rather than an estimate of anything.
DEMO_CONFIG_FILES: tuple[str, ...] = (
    "configs/default.yaml",
    "configs/data/demo.yaml",
    "configs/eval/default.yaml",
)

#: Trading days per calendar day, used to convert a trading-day horizon (the
#: ``tail_risk`` window) into the calendar-day margin the split needs.
#: 5 trading days per 7 calendar days is the convention, matching 30 trading
#: days ~= 42 calendar days.
_TRADING_TO_CALENDAR = 7 / 5


class _StrictModel(BaseModel):
    """Base class that rejects unknown keys.

    Inherited by every configuration model so that a misspelled key is a hard
    error. The alternative — silently ignoring it — produces runs that look
    correctly configured and are not.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=False)


class TimeWindow(_StrictModel):
    """Inclusive date range ``[start, end]``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: date
    end: date

    @model_validator(mode="after")
    def _check_order(self) -> TimeWindow:
        if self.end < self.start:
            raise ValueError(f"window end {self.end} precedes start {self.start}")
        return self

    def days(self) -> int:
        """Length of the window in calendar days, inclusive of both ends."""
        return (self.end - self.start).days + 1

    def render(self) -> str:
        """Human-readable form used in reports."""
        return f"{self.start.isoformat()} .. {self.end.isoformat()}"


class ProjectMeta(_StrictModel):
    """Project-level identity and global knobs."""

    name: str = "shingan"
    description: str = "Evidence-grounded financial risk modelling"
    seed: int = Field(default=DEFAULT_SEED, ge=0, le=2**32 - 1)
    data_version: str = DATA_SCHEMA_VERSION
    root: Path | None = Field(
        default=None,
        description="Override the auto-detected repository root. Useful for tests "
        "and for running a pipeline against a scratch directory.",
    )


class SyntheticConfig(_StrictModel):
    """Knobs for the deterministic offline dataset generator.

    The defaults describe a small panel: a handful of companies over a dozen
    years, small enough that the end-to-end demo finishes in seconds on a laptop
    CPU.
    """

    n_companies: int = Field(default=12, ge=2, le=2000)
    start: date = date(2010, 1, 1)
    end: date = date(2024, 12, 31)
    rows_per_year: int = Field(
        default=4,
        ge=1,
        le=252,
        description="Observation points per company per year. 4 mimics quarterly filings.",
    )
    text_component_strength: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Share of the latent risk factor that is visible ONLY in the text. "
        "This is what makes the fusion comparison meaningful: if it were 0, the "
        "structured track could recover everything and the text track would be noise.",
    )
    structure_information_loss: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Fraction of the latent factor discarded when projecting it onto the "
        "structured features, before noise is added.",
    )
    label_noise: float = Field(default=0.12, ge=0.0, le=1.0)
    base_rate_multiplier: float = Field(
        default=1.0,
        gt=0.0,
        description="Scales the per-label event rates. Raise it to get more positives "
        "in a small demo; the honest default leaves the rates low, as they are in "
        "reality for these labels.",
    )
    inject_edge_cases: bool = Field(
        default=True,
        description="Add the rows that real panels always contain and that most "
        "pipelines break on: a company with almost no history, a halted price "
        "series, a restatement, and a company that stops existing. Used by the "
        "leakage and split tests.",
    )
    missing_rate: float = Field(
        default=0.04,
        ge=0.0,
        le=0.5,
        description="Fraction of ratio observations dropped, so that imputation and the "
        "ratios_missing_frac feature are exercised rather than untested.",
    )


class DataConfig(_StrictModel):
    """Where the panel comes from and how strictly it is assembled."""

    universe: list[str] = Field(
        default_factory=lambda: ["JPM", "GS", "GE", "CAT", "BA", "XOM", "T", "F", "M", "WBA"]
    )
    start: date = date(2010, 1, 1)
    end: date = date(2024, 12, 31)
    sources: list[DataSourceName] = Field(default_factory=lambda: list(_DEFAULT_SOURCES))
    offline: bool = Field(
        default=True,
        description="When true, no adapter may touch the network. The demo and the whole "
        "test suite run with this left at its default.",
    )
    cache_dir: Path | None = None

    # SEC access policy: a descriptive User-Agent is required, and requests must
    # be rate limited. A missing User-Agent returns 403 with an unhelpful body.
    sec_user_agent: str = Field(
        default="shingan-research/0.1 (contact: set SHINGAN_SEC_USER_AGENT)",
        description="Sent as the User-Agent header on every EDGAR request. Override with "
        "the SHINGAN_SEC_USER_AGENT environment variable; include a real contact.",
    )
    sec_requests_per_second: float = Field(default=5.0, gt=0.0, le=10.0)
    sec_max_retries: int = Field(default=4, ge=0, le=10)
    sec_timeout_seconds: float = Field(default=30.0, gt=0.0)

    max_nan_run: int = Field(
        default=10,
        ge=1,
        description="Longest tolerated run of missing prices before the whole rolling "
        "feature window in that row is nulled instead of interpolated.",
    )
    min_history_days: int = Field(
        default=252,
        ge=1,
        description="A row with less than this much price history behind it is flagged "
        "insufficient_history and excluded from the feature matrix.",
    )
    news_dedup_similarity: float = Field(default=0.9, ge=0.0, le=1.0)
    news_dedup_prefix_chars: int = Field(default=256, ge=16)
    max_filings_per_company: int = Field(default=8, ge=1)
    synthetic: SyntheticConfig = Field(default_factory=SyntheticConfig)

    @model_validator(mode="after")
    def _check_window(self) -> DataConfig:
        if self.end <= self.start:
            raise ValueError(f"data.end ({self.end}) must be after data.start ({self.start})")
        if "synthetic" in self.sources and len(self.sources) > 1:
            raise ValueError(
                "synthetic data may not be mixed with real sources in one config; "
                "builder.py refuses to emit a panel that mixes is_synthetic values"
            )
        if not self.offline and self.sources == ["synthetic"]:
            logger.warning(
                "data.offline is False but the only source is synthetic; no network "
                "adapter will be exercised"
            )
        return self


class LabelConfig(_StrictModel):
    """Risk label definitions.

    Horizons are part of the label definition, not tuning parameters: changing one
    changes what "risk" means and invalidates previous results. The three labels
    therefore carry their horizons explicitly rather than sharing a global value.
    """

    targets: list[RiskLabelName] = Field(default_factory=lambda: list(_DEFAULT_TARGETS))

    # default_risk: a rating downgrade of at least N notches, or bankruptcy /
    # default proceedings, within the horizon.
    default_risk_horizon_days: int = Field(default=365, ge=1)
    default_risk_downgrade_notches: int = Field(default=2, ge=1)
    default_risk_use_bankruptcy: bool = True

    # fraud_risk: restatement, enforcement action or non-standard audit opinion.
    fraud_risk_horizon_days: int = Field(default=730, ge=1)

    # tail_risk: drawdown over a window measured in TRADING days. The definition
    # is the running peak-to-trough drawdown, not the drop from the window start:
    # a stock that rallies 40% and then falls 35% from its peak has had a 35%
    # drawdown even though its start-to-end return is positive.
    tail_risk_horizon_trading_days: int = Field(default=30, ge=2)
    tail_risk_drawdown_threshold: float = Field(default=-0.30, lt=0.0, gt=-1.0)

    positive_rate_warning_threshold: float = Field(
        default=0.02,
        ge=0.0,
        le=0.5,
        description="Below this observed positive rate, a label is reported as "
        "too rare for the POC to say anything about.",
    )
    min_positives_for_metrics: int = Field(
        default=20,
        ge=1,
        description="Fewer than this many positives in an evaluation slice and the "
        "report says 'insufficient evidence' instead of printing an AUC.",
    )
    negative_downsample_ratio: float | None = Field(
        default=None,
        gt=0.0,
        description="Negatives kept per positive. None keeps every negative and relies "
        "on sample_weight instead of throwing data away.",
    )

    def calendar_horizon_days(self, label: RiskLabel | RiskLabelName | str) -> int:
        """Horizon expressed in calendar days, for margin arithmetic."""
        return self.horizon_days(label, calendar=True)

    def horizon_days(
        self, label: RiskLabel | RiskLabelName | str, *, calendar: bool = False
    ) -> int:
        """Return the label horizon in trading days, or in calendar days.

        ``tail_risk`` is defined in trading days, so the calendar form is an
        approximation (30 trading days ~= 42 calendar days) used only for sizing
        the purge margin. Everything else is already in calendar days.

        ``label`` accepts either :class:`~shingan.data.schema.RiskLabel` or its
        string value: the two are interchangeable in the data dictionary, and
        requiring the caller to remember which form this particular function wants
        would be a trap for no benefit. The annotation says ``| str`` for exactly that
        reason — it was narrowed to the literal union, which is accurate for callers
        holding a config field but wrong for the many call sites that carry a label
        name as a plain string.
        """
        if label == "default_risk":
            return self.default_risk_horizon_days
        if label == "fraud_risk":
            return self.fraud_risk_horizon_days
        if label == "tail_risk":
            trading = self.tail_risk_horizon_trading_days
            return round(trading * _TRADING_TO_CALENDAR) if calendar else trading
        raise ValueError(f"unknown label: {label!r}")

    def max_calendar_horizon_days(self) -> int:
        """Longest calendar horizon across the configured targets.

        This is the smallest purge margin that can be used with all targets
        simultaneously without leaking.
        """
        if not self.targets:
            raise ValueError("labels.targets is empty; nothing to evaluate")
        return max(self.calendar_horizon_days(label) for label in self.targets)

    def drawdown_threshold_for(self, label: RiskLabelName | str) -> float | None:
        """The drawdown threshold, or None for labels not defined on prices."""
        return self.tail_risk_drawdown_threshold if label == "tail_risk" else None


class SplitConfig(_StrictModel):
    """Static train / valid / test windows plus the purge and embargo margins.

    The nominal windows below are *inputs*; the effective windows after purging are
    computed in :mod:`shingan.eval.splits` and printed at the top of every report,
    because they can be a year or more shorter than what is written here.

    **The windows are derived from the margin, not chosen freely.** Only two edges
    are adjusted by purging: ``train`` is pulled back to ``valid.start - margin`` and
    ``valid`` is pulled back to ``test.start - margin``. The *starts* are untouched.
    So the effective length of a block is decided by where the *next* block starts,
    not by where this block's nominal end is:

    =========================  ========================================
    block                      effective end
    =========================  ========================================
    ``train``                  ``min(train.end, valid.start - margin)``
    ``valid``                  ``min(valid.end, test.start - margin)``
    ``test``                   ``min(test.end, data_end - max_horizon)``
    =========================  ========================================

    With ``purge_days`` bound to the longest horizon in use, and the longest horizon
    here being ``fraud_risk``'s 730 days, ``margin`` is 772 days — just over two
    years. A 15-year panel therefore has to give up about 4.2 years to margins plus
    another 2 years of terminal truncation, leaving under 9 usable years to divide
    three ways. The defaults below divide it roughly 4.9 / 1.9 / 1.9 years.

    Two failure modes this arithmetic has already produced once, both silent:

    * ``valid`` was 324 days rather than the two years the evaluation chapter
      assumes, because ``test.start`` was set 3 years after ``valid.start`` while
      the margin is 2.1 years.
    * Every walk-forward fold was discarded, because a block of ``test_block_years``
      is both the validation and the test block and a 2-year block (730 days) is
      shorter than the 772-day margin, so purging always inverted the validation
      block. ``test_block_years`` must exceed ``margin_days / 365.25`` with room to
      spare; it is 3 here, and :func:`shingan.eval.splits.walk_forward_folds` logs a
      warning per rejected fold rather than reporting an empty rolling evaluation.
    """

    train: TimeWindow = Field(
        default_factory=lambda: TimeWindow(start=date(2010, 1, 1), end=date(2016, 12, 31))
    )
    valid: TimeWindow = Field(
        default_factory=lambda: TimeWindow(start=date(2017, 1, 1), end=date(2019, 12, 31))
    )
    test: TimeWindow = Field(
        default_factory=lambda: TimeWindow(start=date(2021, 1, 1), end=date(2024, 12, 31))
    )

    purge_days: int = Field(
        default=730,
        ge=0,
        description="Samples in a training block whose label window overlaps a later "
        "block are dropped. Must be at least the longest label horizon in use; the "
        "loader raises it if not.",
    )
    embargo_days: int = Field(
        default=30,
        ge=0,
        description="Extra isolation between blocks, in TRADING days, applied on top of "
        "purging to weaken serial correlation across the boundary.",
    )
    test_block_years: int = Field(
        default=3,
        ge=1,
        description="Length of the validation and test blocks in walk-forward. Must "
        "exceed purge_days + embargo as a fraction of a year, or purging inverts the "
        "validation block and every fold is discarded.",
    )
    step_years: int = Field(default=1, ge=1)
    min_effective_window_days: int = Field(
        default=365,
        ge=0,
        description="Minimum length of any effective block in the static split. Purging "
        "shortens blocks without shortening their nominal windows, so a config can "
        "look reasonable and leave a few months of validation data; this turns that "
        "into a load-time error instead of a thin, unreported evaluation. Set to 0 to "
        "disable. Not applied to walk-forward folds, whose blocks are short by "
        "construction and which skip themselves with a warning.",
    )
    per_label: bool = Field(
        default=False,
        description="Run one split per label with purge_days set to that label's own "
        "horizon. Recovers a lot of data for tail_risk, at the cost of comparing "
        "labels over different periods.",
    )

    @property
    def embargo_calendar_days(self) -> int:
        """Embargo converted from trading days to calendar days."""
        return round(self.embargo_days * _TRADING_TO_CALENDAR)

    @property
    def margin_days(self) -> int:
        """Total isolation required between blocks: purge plus embargo."""
        return self.purge_days + self.embargo_calendar_days

    @model_validator(mode="after")
    def _check_ordering(self) -> SplitConfig:
        if not (self.train.end < self.valid.start <= self.valid.end < self.test.start):
            raise ValueError(
                "split windows must be strictly ordered and non-overlapping: "
                f"train {self.train.render()}, valid {self.valid.render()}, test {self.test.render()}"
            )
        return self


class StructuredConfig(_StrictModel):
    """Track A: the gradient-boosted model over structured signals."""

    kind: Literal["hist_gbdt", "logistic"] = "hist_gbdt"
    calibration: Literal["isotonic", "platt", "none"] = Field(
        default="isotonic",
        description="Fitted on the validation fold only. Isotonic needs more data than "
        "Platt to be stable; with a few dozen positives prefer platt.",
    )
    feature_groups: list[FeatureGroupName] = Field(
        default_factory=lambda: list(_DEFAULT_FEATURE_GROUPS)
    )
    params: dict[str, Any] = Field(default_factory=dict)
    use_sample_weight: bool = True
    permutation_importance_repeats: int = Field(default=5, ge=0)
    permutation_importance_sample: int = Field(default=2000, ge=100)


class TextBaselineConfig(_StrictModel):
    """A deliberately cheap CPU text model.

    This exists so the text track always has a baseline that runs anywhere. The
    QLoRA adapter has to beat it to justify its cost, and the ablation table is
    not allowed to compare the LLM only against the structured track.
    """

    kind: Literal["tfidf_logreg"] = "tfidf_logreg"
    max_features: int = Field(default=50_000, ge=100)
    ngram_min: int = Field(default=1, ge=1, le=3)
    ngram_max: int = Field(default=2, ge=1, le=4)
    min_df: int = Field(default=2, ge=1)
    C: float = Field(default=1.0, gt=0.0)
    calibration: Literal["isotonic", "platt", "none"] = "platt"
    max_chars: int = Field(default=40_000, ge=100)

    @model_validator(mode="after")
    def _check_ngram(self) -> TextBaselineConfig:
        if self.ngram_max < self.ngram_min:
            raise ValueError("ngram_max must be >= ngram_min")
        return self


class LoraConfig(_StrictModel):
    """Track B: QLoRA fine-tuning of the base language model.

    Defaults follow `docs/04-training.md`, table 2.2. They are sized for a single
    32 GB card (RTX 5090) at 4096 tokens with gradient checkpointing on.
    """

    base_model: str = "Qwen/Qwen3-14B"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"

    lora_r: int = Field(default=32, ge=1)
    lora_alpha: int = Field(default=64, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    bias: Literal["none", "all", "lora_only"] = "none"

    max_seq_length: int = Field(default=4096, ge=256)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = Field(default=0.03, ge=0.0, lt=1.0)
    num_train_epochs: int = Field(default=3, ge=1)
    per_device_train_batch_size: int = Field(default=1, ge=1)
    per_device_eval_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=16, ge=1)
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    bf16: bool = True
    fp16: bool = False
    attn_implementation: Literal["sdpa", "eager", "flash_attention_2"] = Field(
        default="sdpa",
        description="flash_attention_2 requires self-compilation on Windows and is "
        "deliberately not the default. See docs/adr/0002-training-stack-windows.md.",
    )
    packing: bool = Field(
        default=False,
        description="Packing concatenates samples across prompt boundaries. Each sample "
        "here carries its own label and horizon, so packing would corrupt supervision.",
    )
    dataloader_num_workers: int = Field(
        default=0,
        ge=0,
        description="0 on Windows: worker processes require spawn-safe pickling and are "
        "the usual cause of DataLoader failures there.",
    )
    eval_strategy: Literal["no", "steps", "epoch"] = "epoch"
    save_total_limit: int = Field(default=3, ge=1)
    logging_steps: int = Field(default=10, ge=1)
    report_to: str = "none"
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    packing_max_chars: int | None = None
    device_map: Literal["auto", "single", "cpu"] = "single"
    output_dir: Path = Path("artifacts/lora")

    @model_validator(mode="after")
    def _check_precision(self) -> LoraConfig:
        if self.bf16 and self.fp16:
            raise ValueError("bf16 and fp16 are mutually exclusive")
        if self.per_device_train_batch_size > 1 and self.gradient_checkpointing and not self.bf16:
            logger.warning(
                "fp16 without bf16 on a long-context run is prone to loss scaling "
                "overflow; prefer bf16 on Blackwell"
            )
        return self


class FusionConfig(_StrictModel):
    """Track C: combining the two score sources.

    The stacker is fitted on the validation fold only. Fitting it on test — or on
    training predictions, which are optimistically biased — is the most common way
    a fusion layer produces a number that cannot be reproduced out of sample.
    """

    kind: Literal["logistic_stack", "rank_average"] = "logistic_stack"
    weights: tuple[float, float] = Field(
        default=(0.5, 0.5),
        description="(structured, text) weights, used only by rank_average.",
    )
    use_rank_inputs: bool = Field(
        default=True,
        description="Feed cross-sectional ranks as well as raw scores to the stacker, so "
        "it degrades gracefully when one track has a different scale.",
    )
    seed: int = DEFAULT_SEED

    @model_validator(mode="after")
    def _check_weights(self) -> FusionConfig:
        if self.kind == "rank_average":
            if len(self.weights) != 2 or any(w < 0 for w in self.weights):
                raise ValueError("weights must be two non-negative numbers")
            if sum(self.weights) <= 0:
                raise ValueError("weights must not sum to zero")
        return self


class ThresholdConfig(_StrictModel):
    """Acceptance gates. These are targets to report against, not tuned parameters."""

    auc: float = Field(default=0.75, ge=0.5, le=1.0)
    ks: float = Field(default=0.30, ge=0.0, le=1.0)
    ece: float = Field(default=0.05, ge=0.0, le=1.0)
    ic: float = Field(default=0.05, ge=0.0, le=1.0)
    icir: float = Field(default=0.5, ge=0.0)
    psi: float = Field(default=0.25, ge=0.0)
    csi: float = Field(default=0.25, ge=0.0)
    min_pr_auc_lift: float = Field(
        default=1.0,
        ge=0.0,
        description="PR-AUC divided by the positive rate. 1.0 means 'no better than "
        "random'; the fused model must exceed the structured-only value of this ratio.",
    )


class RollingConfig(_StrictModel):
    """Walk-forward evaluation settings."""

    enabled: bool = True
    window_years: int = Field(default=2, ge=1)
    min_positives_per_fold: int = Field(default=5, ge=1)
    bootstrap_samples: int = Field(default=500, ge=0)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)


class StressPeriod(_StrictModel):
    """A named historical regime used for stress testing."""

    name: str
    start: date
    end: date
    note: str = ""


class ReportConfig(_StrictModel):
    """Report generation."""

    title: str = "Shingan risk model report"
    include_figures: bool = True
    max_feature_rows: int = Field(default=25, ge=1)
    top_evidence_examples: int = Field(default=5, ge=0)


class EvalConfig(_StrictModel):
    """Evaluation settings."""

    n_bins: int = Field(default=10, ge=2, le=100)
    n_quantiles: int = Field(default=5, ge=2, le=20)
    beta: float = Field(default=2.0, gt=0.0)
    top_fraction: float = Field(default=0.05, gt=0.0, lt=1.0)
    min_test_rows: int = Field(default=50, ge=1)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    rolling: RollingConfig = Field(default_factory=RollingConfig)
    stress_periods: list[StressPeriod] = Field(
        default_factory=lambda: [
            StressPeriod(
                name="global_financial_crisis",
                start=date(2008, 9, 1),
                end=date(2009, 6, 30),
                note="Outside the default POC window; skipped with an explicit note when unused.",
            ),
            StressPeriod(name="covid_crash", start=date(2020, 2, 19), end=date(2020, 4, 30)),
            StressPeriod(name="rate_shock", start=date(2022, 1, 1), end=date(2022, 12, 31)),
        ]
    )
    report: ReportConfig = Field(default_factory=ReportConfig)


class ProjectConfig(_StrictModel):
    """The complete, validated configuration for one run."""

    project: ProjectMeta = Field(default_factory=ProjectMeta)
    data: DataConfig = Field(default_factory=DataConfig)
    labels: LabelConfig = Field(default_factory=LabelConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    structured: StructuredConfig = Field(default_factory=StructuredConfig)
    text_baseline: TextBaselineConfig = Field(default_factory=TextBaselineConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    @model_validator(mode="after")
    def _check_purge_covers_horizons(self) -> ProjectConfig:
        """Never let the purge margin be smaller than the longest label horizon.

        A too-small margin does not raise anywhere downstream; it just makes the
        test metrics slightly too good. Raising it silently is the lesser evil, but
        it is logged, because it changes the effective size of every training set.
        """
        required = self.labels.max_calendar_horizon_days()
        if self.split.purge_days < required:
            logger.warning(
                "split.purge_days=%d is smaller than the longest label horizon (%d days); "
                "raising it to %d to avoid leaking labels across the split boundary",
                self.split.purge_days,
                required,
                required,
            )
            self.split.purge_days = required
        return self

    def project_paths(self) -> ProjectPaths:
        """Resolved directory layout for this configuration."""
        return ProjectPaths.from_root(self.project.root)

    def run_metadata(self) -> dict[str, Any]:
        """A flat, JSON-serialisable description of the run.

        Written next to every report so a number can be traced back to the exact
        configuration that produced it.
        """
        return {
            "project": self.project.name,
            "seed": self.project.seed,
            "data_version": self.project.data_version,
            "sources": list(self.data.sources),
            "offline": self.data.offline,
            "universe_size": len(self.data.universe),
            "data_window": [self.data.start.isoformat(), self.data.end.isoformat()],
            "targets": list(self.labels.targets),
            "label_horizons": {
                label: self.labels.horizon_days(label) for label in self.labels.targets
            },
            "split": {
                "train": self.split.train.render(),
                "valid": self.split.valid.render(),
                "test": self.split.test.render(),
                "purge_days": self.split.purge_days,
                "embargo_days": self.split.embargo_days,
                "margin_days": self.split.margin_days,
                "per_label": self.split.per_label,
            },
            "structured_kind": self.structured.kind,
            "calibration": self.structured.calibration,
            "lora": {
                "base_model": self.lora.base_model,
                "lora_r": self.lora.lora_r,
                "lora_alpha": self.lora.lora_alpha,
                "max_seq_length": self.lora.max_seq_length,
                "load_in_4bit": self.lora.load_in_4bit,
            },
            "fusion_kind": self.fusion.kind,
        }


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``.

    Lists are replaced, not concatenated: a config that sets
    ``data.universe: [JPM]`` means exactly one company, not one company plus the
    ten inherited defaults. The inputs are not mutated.

    Args:
        base: Lower-priority mapping.
        override: Higher-priority mapping.

    Returns:
        A new merged mapping.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _coerce_env_value(raw: str) -> Any:
    """Best-effort scalar coercion for environment overrides."""
    lowered = raw.strip().lower()
    if lowered in {"true", "yes", "1", "on"}:
        return True
    if lowered in {"false", "no", "0", "off"}:
        return False
    if lowered in {"none", "null", ""}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


#: Environment variable -> dotted config path. Kept as an explicit allow-list so
#: that an unrelated variable can never silently change a run.
_ENV_OVERRIDES: dict[str, str] = {
    "SHINGAN_ROOT": "project.root",
    "SHINGAN_SEED": "project.seed",
    "SHINGAN_DATA_VERSION": "project.data_version",
    "SHINGAN_SEC_USER_AGENT": "data.sec_user_agent",
    "SHINGAN_OFFLINE": "data.offline",
    "SHINGAN_CACHE_DIR": "data.cache_dir",
    "SHINGAN_LORA_BASE_MODEL": "lora.base_model",
    "SHINGAN_LORA_OUTPUT_DIR": "lora.output_dir",
}


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Apply :data:`_ENV_OVERRIDES` to a raw configuration mapping."""
    result = copy.deepcopy(raw)
    for env_name, dotted in _ENV_OVERRIDES.items():
        if env_name not in os.environ:
            continue
        value = _coerce_env_value(os.environ[env_name])
        cursor = result
        parts = dotted.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
        logger.debug("config override from %s: %s = %r", env_name, dotted, value)
    return result


def load_raw_config(path: Path | str) -> dict[str, Any]:
    """Read a YAML config file into a plain mapping.

    Args:
        path: File to read.

    Returns:
        The parsed mapping, or an empty mapping for an empty file.

    Raises:
        FileNotFoundError: If the file does not exist. Raised explicitly rather
            than letting PyYAML return None, because a missing overlay silently
            doing nothing is a nasty failure mode.
        ValueError: If the file does not contain a mapping at the top level.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(
            f"top level of {config_path} must be a mapping, got {type(loaded).__name__}"
        )
    return loaded


def load_config(
    base: Path | str | None = None,
    overlays: list[Path | str] | None = None,
    *,
    root: Path | str | None = None,
) -> ProjectConfig:
    """Load, merge, override and validate a configuration.

    Args:
        base: Base YAML file. Defaults to ``configs/default.yaml`` under the
            resolved project root.
        overlays: Zero or more YAML files deep-merged onto the base, in order.
        root: Explicit project root, bypassing auto-detection.

    Returns:
        A validated :class:`ProjectConfig`.

    Raises:
        FileNotFoundError: If the base config cannot be found.
        pydantic.ValidationError: If the merged result violates any constraint,
            including an unknown key.
    """
    paths = ProjectPaths.from_root(root)
    base_path = Path(base) if base is not None else paths.configs / "default.yaml"
    raw = load_raw_config(base_path)
    for overlay in overlays or []:
        overlay_path = Path(overlay)
        raw = deep_merge(raw, load_raw_config(overlay_path))
        logger.debug("merged config overlay %s", overlay_path)
    raw = _apply_env_overrides(raw)
    config = ProjectConfig.model_validate(raw)
    logger.debug("configuration validated from %s", base_path)
    return config


def default_config() -> ProjectConfig:
    """A fully defaulted configuration, with no file I/O.

    Used by tests and by the synthetic generator, which must not depend on the
    presence or contents of ``configs/``.
    """
    return ProjectConfig()


__all__ = [
    "DEFAULT_CONFIG_FILES",
    "DataConfig",
    "DataSourceName",
    "EvalConfig",
    "FusionConfig",
    "LabelConfig",
    "LoraConfig",
    "ProjectConfig",
    "ProjectMeta",
    "RiskLabelName",
    "RollingConfig",
    "SplitConfig",
    "StressPeriod",
    "StructuredConfig",
    "SyntheticConfig",
    "TextBaselineConfig",
    "ThresholdConfig",
    "TimeWindow",
    "deep_merge",
    "default_config",
    "load_config",
    "load_raw_config",
]
