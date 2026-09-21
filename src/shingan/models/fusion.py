"""Track C: combining the two score sources into one calibrated risk score.

The design rule from docs/01-architecture.md, stated once so every method below can
be read against it: **the stacker is fitted on the validation fold only.** Two
tempting alternatives are both wrong.

* Fitting on the *test* fold makes the reported fusion a fitted parameter of the
  block it is measured on. The number is then reproducible only on that block.
* Fitting on *training* scores is subtler and worse: those scores come from a model
  that was fitted on the very rows it is scoring, so they are optimistically
  biased, and a stacker fitted on biased inputs learns a weight for the structured
  track it would not deserve out of sample. The ablation table exists to measure
  exactly this, so it must not be built on it.

Two kinds are supported, from :class:`~shingan.config.FusionConfig`:

``logistic_stack``
    A one-layer logistic regression over the two scores (and, when
    ``use_rank_inputs`` is set, their cross-sectional ranks as well). Fitting is
    trivial and the weights are readable.

``rank_average``
    A fixed weighted average of the two **cross-sectional ranks**. No fitting is
    needed, so it cannot overfit the validation fold — useful precisely when the
    validation fold holds too few positives to fit anything.

Degenerate folds are survivable. A validation fold with one positive, or none,
cannot fit a logistic stacker; that is the normal state of a ``tail_risk`` fold at a
two-percent base rate, not an error. Rather than raise — the pipeline calls
:meth:`RiskFusion.fit` without a guard, because a fusion layer is never the reason a
label should be dropped — the class falls back to the rank average and records why
in ``diagnostics``, which the report prints.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shingan.config import FusionConfig
from shingan.eval.metrics import average_precision, roc_auc
from shingan.logging_utils import get_logger
from shingan.models.persistence import load_model_payload, write_model

logger = get_logger(__name__)

#: Column names the stacker's design matrix is keyed by. The pipeline translates its
#: own path names (``structured`` / ``text_baseline``) into these at one boundary —
#: see :func:`shingan.pipeline._ablation_table`.
STRUCTURED_SCORE = "score_structured"
TEXT_SCORE = "score_text"

#: Design-matrix columns holding the cross-sectional percentile ranks, added when
#: ``fusion.use_rank_inputs`` is set.
RANK_STRUCTURED = "rank_structured"
RANK_TEXT = "rank_text"

#: Name of the artifact written by :meth:`RiskFusion.save`.
MODEL_FILENAME = "fusion.joblib"

#: Path names used by the ablation table. Deliberately the fusion module's own
#: vocabulary rather than the pipeline's: this table is about *inputs*, not about the
#: pipeline's three paths.
PATH_STRUCTURED = "structured"
PATH_TEXT = "text"
PATH_FUSED = "fused"
ABLATION_PATHS: tuple[str, ...] = (PATH_STRUCTURED, PATH_TEXT, PATH_FUSED)

#: Probabilities are clipped away from the closed interval so that a downstream log
#: transform cannot take a log of exactly zero.
_PROBABILITY_EPSILON = 1e-6

#: Value a missing or unrankable score takes in the design matrix. One half is the
#: neutral element for a rank in [0, 1] and the value the pipeline's own docstring
#: documents, so a row that could not be ranked contributes nothing rather than
#: contributing a zero that reads as "the lowest risk in the cross-section".
_NEUTRAL_INPUT = 0.5

#: Fewest positives a logistic stacker can be fitted on and still be a fit rather than a
#: memorisation. A logistic regression needs two classes to run at all, so one positive
#: is technically accepted — and that is the problem: with a single positive the
#: optimiser drives its coefficient to whatever separates that one row, and the weights
#: reported afterwards describe the row, not the signal. The floor matches
#: ``eval.rolling.min_positives_per_fold``, the codebase's own convention for "too thin
#: to say anything", rather than being a new number invented here.
#:
#: Below it the fit still happens — silently substituting a different model would be a
#: hidden decision, which is the failure mode this module's docstring is about — but the
#: degeneracy is recorded in :attr:`RiskFusion.diagnostics` and travels into the report.
MIN_POSITIVES_FOR_LOGISTIC_STACK = 5

#: Anything of the shape ``metric(y_true, y_score) -> mapping``. The declared return
#: is a Mapping so that one call can report several metrics — the ablation table
#: needs AUC and PR-AUC side by side, because a path that wins on one and loses on the
#: other is a real and reportable outcome.
MetricFn = Callable[[Any, Any], Any]


# --------------------------------------------------------------------------- #
# Cross-sectional ranking
# --------------------------------------------------------------------------- #


def cross_sectional_rank(
    values: Sequence[float] | pd.Series | np.ndarray,
    dates: Sequence[Any] | pd.Series | np.ndarray | None = None,
) -> np.ndarray:
    """Percentile rank of ``values`` inside its own date, in ``[0, 1]``.

    Ranking is what makes the two tracks comparable. The structured track emits a
    calibrated probability and the text baseline emits one too, but their scales drift
    between runs and their cross-sectional dispersion differs — a stacker fed raw
    scores has to learn a per-run scale correction it cannot generalise. Ranks are
    scale-free and therefore stable.

    Args:
        values: Scores to rank.
        dates: Date per row. When given, ranking is within each date, so the rank
            answers "how risky is this company relative to its peers today". When
            None, the whole vector is ranked as one block — the correct fallback when
            the caller has a single cross-section.

    Returns:
        Percentile ranks, NaN where the input is NaN or its date is NaT. A date that
        holds a single scored row yields 1.0 for that row (the last percentile), which
        is a genuine limitation of a one-name cross-section rather than a bug: the
        report's backtest section refuses to run below
        :data:`~shingan.pipeline.MIN_NAMES_PER_DATE` names for the same reason.
    """
    series = pd.Series(np.asarray(values, dtype=float))
    if dates is None:
        return series.rank(pct=True, na_option="keep").to_numpy(dtype=float)

    stamps = pd.to_datetime(pd.Series(np.asarray(dates)), errors="coerce")
    ranks = series.groupby(stamps.to_numpy()).rank(pct=True, na_option="keep")
    return ranks.to_numpy(dtype=float)


# --------------------------------------------------------------------------- #
# The fusion layer
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class RiskFusion:
    """Combines the structured and text scores into one risk score.

    Args:
        config: The ``fusion`` block of the project configuration.
    """

    config: FusionConfig
    _model: Any = field(default=None, init=False, repr=False)
    _feature_names: list[str] = field(default_factory=list, init=False)
    _kind: str = field(default="unfitted", init=False)
    _base_rate: float = field(default=0.0, init=False)
    _fit_split: str = field(default="", init=False)
    diagnostics: dict[str, Any] = field(default_factory=dict, init=False)

    # -- fitting ----------------------------------------------------------

    def fit(
        self,
        structured: Sequence[float] | pd.Series | np.ndarray,
        text: Sequence[float] | pd.Series | np.ndarray,
        y: Sequence[float] | pd.Series | np.ndarray,
        *,
        dates: Sequence[Any] | pd.Series | np.ndarray | None = None,
        fit_split: str = "valid",
    ) -> RiskFusion:
        """Fit the stacker on one fold's scores.

        Args:
            structured: Structured-track scores for the fitting fold.
            text: Text-track scores for the same rows, same order.
            y: Binary labels for the same rows, same order.
            dates: Date per row, for the cross-sectional ranks. Strongly recommended:
                without it the ranks are computed over the fold as a single block,
                which mixes 2017 names against 2019 names.
            fit_split: Name of the fold being fitted on, recorded so the report can
                state where the stacker's weights came from. ``docs/05-evaluation.md``
                requires the fusion's fitting fold to be named, because a stacker
                silently fitted on the wrong block is the failure this parameter makes
                visible.

        Returns:
            ``self``, fitted.
        """
        self._fit_split = str(fit_split)
        y_array = np.asarray(y, dtype=float).ravel()
        self._base_rate = float(np.nanmean(y_array)) if y_array.size else 0.0

        matrix = self._design_matrix(structured, text, dates=dates)
        self._feature_names = [str(column) for column in matrix.columns]
        self.diagnostics = {
            "kind": str(self.config.kind),
            "fit_split": self._fit_split,
            "n_fit_rows": int(len(matrix)),
            "positives_fit": int(np.nansum(y_array >= 0.5)),
            "use_rank_inputs": bool(self.config.use_rank_inputs),
            "inputs": list(self._feature_names),
        }
        if self.config.kind == "rank_average":
            self.diagnostics["note"] = (
                "rank_average needs no fitting; the weights below are the configured "
                "constants, applied to the cross-sectional ranks"
            )
            self.diagnostics["weights"] = self._normalised_weights()
            self._model = None
            self._kind = "rank_average"
            return self

        n_finite = int(np.isfinite(y_array).sum())
        positives = int(np.nansum(y_array >= 0.5))
        n_classes = int(np.unique(y_array[np.isfinite(y_array)] >= 0.5).size)
        if n_finite == 0 or n_classes < 2:
            # A logistic fit needs two classes. One class is the normal state of a
            # low-base-rate fold, not an error, and the rank average is at least a
            # defensible ranking, so it is what ships.
            self.diagnostics["note"] = (
                f"the fitting fold holds {n_finite} labelled row(s) and {positives} "
                "positive(s), i.e. a single class, so a logistic stacker cannot be "
                "fitted; the fusion falls back to the rank average rather than raising, "
                "because a fusion layer is not a reason to drop a label"
            )
            self._model = None
            self._kind = "rank_average_fallback"
            return self

        from sklearn.linear_model import LogisticRegression

        self._model = LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver="lbfgs",
            random_state=int(self.config.seed),
        )
        self._model.fit(matrix.to_numpy(dtype=float), (y_array >= 0.5).astype(int))
        self._kind = "logistic_stack"
        coefficients = self._model.coef_.ravel()
        self.diagnostics["coefficients"] = {
            name: float(value) for name, value in zip(self._feature_names, coefficients, strict=False)
        }
        self.diagnostics["intercept"] = float(self._model.intercept_.ravel()[0])
        if positives < MIN_POSITIVES_FOR_LOGISTIC_STACK:
            self.diagnostics["warning"] = (
                f"the stacker was fitted on {positives} positive(s), below the "
                f"{MIN_POSITIVES_FOR_LOGISTIC_STACK} that "
                "`eval.rolling.min_positives_per_fold` treats as the floor for a usable "
                "fold. The fit is reported as it stands rather than replaced, but the "
                "coefficients below describe those rows rather than the signal, so the "
                "fused number should be read as 'a stacker was fitted', not as evidence "
                "that stacking works."
            )
            logger.warning(
                "fusion stacker fitted on %d positive(s) in the %s fold; coefficients are "
                "not interpretable at this count",
                positives,
                self._fit_split,
            )
        return self

    def _normalised_weights(self) -> tuple[float, float]:
        """The configured ``(structured, text)`` weights, scaled to sum to one."""
        first, second = (float(self.config.weights[0]), float(self.config.weights[1]))
        total = first + second
        if total <= 0:  # pragma: no cover - FusionConfig rejects this
            return 0.5, 0.5
        return first / total, second / total

    def _design_matrix(
        self,
        structured: Sequence[float] | pd.Series | np.ndarray,
        text: Sequence[float] | pd.Series | np.ndarray,
        *,
        dates: Sequence[Any] | pd.Series | np.ndarray | None = None,
    ) -> pd.DataFrame:
        """The stacker's inputs: the two scores, plus their ranks when configured.

        Missing values are imputed to :data:`_NEUTRAL_INPUT` rather than dropped. A
        dropped row silently changes the evaluation population, and in the ablation
        that would let the fused path be measured on a different set of rows than the
        two paths it is being compared against — which is the one thing the ablation's
        common mask exists to prevent.
        """
        raw_structured = _as_float_array(structured)
        raw_text = _as_float_array(text)
        if raw_structured.size != raw_text.size:
            raise ValueError(
                "the structured and text score arrays must be the same length, got "
                f"{raw_structured.size} and {raw_text.size}"
            )
        columns: dict[str, np.ndarray] = {
            STRUCTURED_SCORE: raw_structured,
            TEXT_SCORE: raw_text,
        }
        if self.config.use_rank_inputs or self.config.kind == "rank_average":
            columns[RANK_STRUCTURED] = cross_sectional_rank(raw_structured, dates)
            columns[RANK_TEXT] = cross_sectional_rank(raw_text, dates)
        matrix = pd.DataFrame(columns)
        return matrix.where(np.isfinite(matrix), _NEUTRAL_INPUT)

    # -- prediction -------------------------------------------------------

    def predict_proba(
        self,
        structured: Sequence[float] | pd.Series | np.ndarray,
        text: Sequence[float] | pd.Series | np.ndarray,
        *,
        dates: Sequence[Any] | pd.Series | np.ndarray | None = None,
    ) -> np.ndarray:
        """Fused risk score for each row.

        Args:
            structured: Structured-track scores to fuse.
            text: Text-track scores for the same rows.
            dates: Date per row, for the cross-sectional ranks. Must be passed for the
                ranks to match those used at fit time; the pipeline passes the panel's
                ``as_of`` column for exactly this reason.

        Returns:
            Scores in ``[0, 1]``, clipped away from both endpoints.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._kind == "unfitted":
            raise RuntimeError("RiskFusion.predict_proba called before fit()")
        design = self._design_matrix(structured, text, dates=dates)

        if self._model is not None:
            matrix = design.reindex(columns=self._feature_names).to_numpy(dtype=float)
            scores = self._model.predict_proba(matrix)[:, 1]
        else:
            # rank_average, or the fallback taken when the fitting fold could not
            # support a logistic fit. Both are a weighted average of the ranks in the
            # design matrix already built above, which is why they share this branch.
            first, second = self._normalised_weights()
            scores = (
                first * design[RANK_STRUCTURED].to_numpy(dtype=float)
                + second * design[RANK_TEXT].to_numpy(dtype=float)
            )
        return np.clip(np.asarray(scores, dtype=float), _PROBABILITY_EPSILON, 1 - _PROBABILITY_EPSILON)

    # -- persistence ------------------------------------------------------

    def save(self, directory: Path | str) -> Path:
        """Persist the fitted stacker and its diagnostics.

        Returns the directory, so ``RiskFusion.load(stacker.save(d))`` round-trips. The
        file name comes from :data:`MODEL_FILENAME` and never from the caller: ``cli train``
        used to write this class with a bare ``joblib.dump`` under a caller-chosen name,
        which produced an artifact this loader then could not find.
        """
        destination = write_model(
            directory,
            MODEL_FILENAME,
            {
                "config": self.config.model_dump(),
                "model": self._model,
                "feature_names": self._feature_names,
                "kind": self._kind,
                "base_rate": self._base_rate,
                "fit_split": self._fit_split,
                "diagnostics": self.diagnostics,
            },
        )
        logger.info("wrote fusion model to %s", destination / MODEL_FILENAME)
        return destination

    @classmethod
    def load(cls, directory: Path | str) -> RiskFusion:
        """Load a fusion written by :meth:`save`, from the directory ``save`` returned."""
        payload = load_model_payload(
            directory,
            MODEL_FILENAME,
            owner=cls.__name__,
            required=("config", "model", "feature_names", "kind", "fit_split"),
        )
        fusion = cls(FusionConfig(**payload["config"]))
        fusion._model = payload["model"]
        fusion._feature_names = list(payload["feature_names"])
        fusion._kind = str(payload["kind"])
        fusion._base_rate = float(payload["base_rate"])
        fusion._fit_split = str(payload["fit_split"])
        fusion.diagnostics = dict(payload["diagnostics"])
        return fusion


# --------------------------------------------------------------------------- #
# Ablation
# --------------------------------------------------------------------------- #


def _default_metric(y_true: Any, y_score: Any) -> dict[str, float]:
    """AUC and PR-AUC, the two the ablation needs together.

    One metric is not enough here. A fused path that raises AUC while lowering PR-AUC
    is a real and common outcome at a low base rate — the fused score reorders the
    middle of the distribution, where there are almost no positives — and reporting
    only one of them would present that as a clean win or a clean loss.
    """
    return {"auc": roc_auc(y_true, y_score), "pr_auc": average_precision(y_true, y_score)}


def fusion_ablation(
    panel: pd.DataFrame,
    scores: pd.DataFrame,
    y: Sequence[float] | pd.Series | np.ndarray,
    *,
    dates: Sequence[Any] | pd.Series | np.ndarray | None = None,
    fit_mask: Sequence[bool] | pd.Series | None = None,
    apply_mask: Sequence[bool] | pd.Series | None = None,
    config: FusionConfig | None = None,
    metric: MetricFn | None = None,
) -> pd.DataFrame:
    """Structured-only / text-only / fused, over one common fit and apply mask.

    The masks are what make this a comparison. Scored on its own subset, each path
    would be evaluated on a different population, and the difference between the rows
    would be reported as a difference between the models.

    Args:
        panel: The assembled panel. Used for its index, which the masks and the score
            frame are aligned against.
        scores: Frame holding :data:`STRUCTURED_SCORE` and :data:`TEXT_SCORE`, indexed
            like ``panel``. Built by :func:`shingan.pipeline._ablation_table`.
        y: Labels, aligned to ``panel``.
        dates: Date per row, for the cross-sectional ranks. Defaults to the panel's
            ``as_of`` column when present.
        fit_mask: Rows the stacker may be fitted on. Defaults to every row that has
            both scores, which is the honest default for a caller that has only one
            block; the pipeline passes its validation block explicitly.
        apply_mask: Rows to score and measure. Defaults to the fit mask.
        config: Fusion configuration. Defaults to a fresh
            :class:`~shingan.config.FusionConfig`, so the default row describes the
            same stacker as the headline report.
        metric: A ``metric(y_true, y_score)`` callable returning a mapping of metric
            name to value. Defaults to :func:`_default_metric` (AUC and PR-AUC).

    Returns:
        One row per (path, metric), with the population size and base rate carried
        along so a reader can see what each number rests on. Empty when the apply mask
        selects no rows.

    Raises:
        KeyError: If ``scores`` is missing one of the two score columns. Raised with
            the expected names rather than letting a rename upstream surface as a
            ``None`` score and a silently flat ablation.
    """
    missing = [name for name in (STRUCTURED_SCORE, TEXT_SCORE) if name not in scores.columns]
    if missing:
        raise KeyError(
            f"scores frame is missing {missing}; expected the columns "
            f"{STRUCTURED_SCORE!r} / {TEXT_SCORE!r} (the pipeline renames its "
            f"'structured' / 'text_baseline' paths at this boundary)"
        )

    aligned = scores.reindex(panel.index) if not scores.index.equals(panel.index) else scores
    y_series = _aligned_series(y, panel.index, name="y")
    date_series = _aligned_dates(dates, panel, panel.index)

    fit_selection = _aligned_mask(fit_mask, panel.index)
    if fit_selection is None:
        fit_selection = aligned[[STRUCTURED_SCORE, TEXT_SCORE]].notna().all(axis=1)
    apply_selection = _aligned_mask(apply_mask, panel.index)
    if apply_selection is None:
        apply_selection = fit_selection

    if not bool(apply_selection.any()):
        return pd.DataFrame()

    metric_fn = metric or _default_metric
    metric_name = str(getattr(metric_fn, "__name__", "metric"))

    # The stacker is refitted here rather than reused from the caller. It has to be:
    # the caller fitted its stacker on the scores it holds, and this function's fit
    # mask may differ. Reusing it would silently measure a stacker fitted on other
    # rows — the exact confusion the ablation table is supposed to resolve.
    fused_scores = np.full(len(panel), np.nan, dtype=float)
    fusion = RiskFusion(config or FusionConfig())
    if bool(fit_selection.any()) and bool(y_series.loc[fit_selection].notna().any()):
        fusion.fit(
            aligned.loc[fit_selection, STRUCTURED_SCORE],
            aligned.loc[fit_selection, TEXT_SCORE],
            y_series.loc[fit_selection].to_numpy(),
            dates=date_series.loc[fit_selection] if date_series is not None else None,
            fit_split="fit_mask",
        )
        fused_scores = fusion.predict_proba(
            aligned[STRUCTURED_SCORE],
            aligned[TEXT_SCORE],
            dates=date_series,
        )
    else:
        logger.warning(
            "the ablation's fit mask is empty or holds no labelled row; the fused row "
            "is reported as not applicable rather than fitted on nothing"
        )

    path_values: dict[str, pd.Series] = {
        PATH_STRUCTURED: aligned[STRUCTURED_SCORE],
        PATH_TEXT: aligned[TEXT_SCORE],
        PATH_FUSED: pd.Series(fused_scores, index=panel.index),
    }

    y_apply = y_series.loc[apply_selection].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for path in ABLATION_PATHS:
        values = path_values[path].loc[apply_selection].to_numpy(dtype=float)
        if not np.isfinite(values).any():
            rows.append(
                {
                    "path": path,
                    "metric": metric_name,
                    "value": float("nan"),
                    "n_rows": 0,
                    "n_positives": 0,
                    "base_rate": float("nan"),
                    "note": "no finite score in the apply mask",
                }
            )
            continue
        computed = metric_fn(y_apply, values)
        items = (
            computed.items()
            if isinstance(computed, Mapping)
            else [(metric_name, float(computed))]
        )
        finite = np.isfinite(y_apply) & np.isfinite(values)
        for name, value in items:
            rows.append(
                {
                    "path": path,
                    "metric": str(name),
                    "value": float(value),
                    "n_rows": int(finite.sum()),
                    "n_positives": int((y_apply[finite] >= 0.5).sum()),
                    "base_rate": float((y_apply[finite] >= 0.5).mean()) if finite.any() else float("nan"),
                    "note": "",
                }
            )
    table = pd.DataFrame(rows)
    table.attrs["fusion_diagnostics"] = dict(fusion.diagnostics)
    table.attrs["fit_split"] = fusion._fit_split
    return table


# --------------------------------------------------------------------------- #
# Small alignment helpers
# --------------------------------------------------------------------------- #


def _as_float_array(values: Sequence[float] | pd.Series | np.ndarray) -> np.ndarray:
    """Coerce a score column to a 1-D float array."""
    if isinstance(values, pd.Series):
        return values.to_numpy(dtype=float)
    array = np.asarray(values, dtype=float)
    return array.ravel()


def _aligned_series(
    values: Sequence[float] | pd.Series | np.ndarray, index: pd.Index, *, name: str
) -> pd.Series:
    """Put ``values`` on ``index``, trusting an existing Series' own index."""
    if isinstance(values, pd.Series):
        return values.reindex(index) if not values.index.equals(index) else values.astype(float)
    array = np.asarray(values, dtype=float).ravel()
    if array.size != len(index):
        raise ValueError(f"{name} has {array.size} entries but the panel has {len(index)} rows")
    return pd.Series(array, index=index, name=name)


def _aligned_dates(
    dates: Sequence[Any] | pd.Series | np.ndarray | None,
    panel: pd.DataFrame,
    index: pd.Index,
) -> pd.Series | None:
    """Resolve the date column, falling back to the panel's own ``as_of``."""
    if dates is None:
        if "as_of" in panel.columns:
            return pd.to_datetime(panel["as_of"], errors="coerce")
        return None
    if isinstance(dates, pd.Series):
        resolved = dates.reindex(index) if not dates.index.equals(index) else dates
    else:
        resolved = pd.Series(np.asarray(dates).ravel(), index=index)
    return pd.to_datetime(resolved, errors="coerce")


def _aligned_mask(
    mask: Sequence[bool] | pd.Series | None, index: pd.Index
) -> pd.Series | None:
    """Coerce a boolean mask to a Series on ``index``."""
    if mask is None:
        return None
    if isinstance(mask, pd.Series):
        resolved = mask.reindex(index) if not mask.index.equals(index) else mask
        return resolved.astype(bool).fillna(False)
    array = np.asarray(mask).ravel()
    if array.size != len(index):
        raise ValueError(f"mask has {array.size} entries but the panel has {len(index)} rows")
    return pd.Series(array.astype(bool), index=index)


__all__ = [
    "ABLATION_PATHS",
    "MIN_POSITIVES_FOR_LOGISTIC_STACK",
    "MODEL_FILENAME",
    "PATH_FUSED",
    "PATH_STRUCTURED",
    "PATH_TEXT",
    "RANK_STRUCTURED",
    "RANK_TEXT",
    "STRUCTURED_SCORE",
    "TEXT_SCORE",
    "MetricFn",
    "RiskFusion",
    "cross_sectional_rank",
    "fusion_ablation",
]
