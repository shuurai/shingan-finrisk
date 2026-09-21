"""Track A: the calibrated gradient-boosted model over structured signals.

The design contract from docs/01-architecture.md and docs/04-training.md, in one
sentence: **fit on the training block, calibrate on the validation block, never on
train.** Train scores are optimistically biased — a model that fits 250 rows also
sorts them better than it sorts new ones — so a calibrator fitted there reports
over-confident probabilities, and every ECE/Brier number downstream inherits the
error. The validation block is the only block this class ever calibrates on.

Three properties the rest of the pipeline relies on:

* ``fit`` accepts NaN features. The panel is built from real filings, so a bank with
  no ``InventoryNet`` concept has a genuinely missing column, not a zero. Both
  estimators handle missingness (HistGradientBoostingClassifier natively, the
  logistic path through an explicit imputer) rather than dropping rows, because
  dropping a row silently removes a company from the evaluation.
* ``calibration_report`` always says which calibrator ran and why. "Isotonic was
  downgraded to Platt because the calibration fold held 3 positives" is a fact a
  reader needs; an unexplained ECE is not.
* ``predict_proba`` returns probabilities that are finite and in [0, 1] even when the
  calibrator saw a degenerate fold, so a single odd label cannot produce NaN scores
  that propagate into the report as "undefined" everywhere.
* **Zero-observation columns are dropped at fit time and named in the report.** A
  feature the data pipeline never managed to produce — ``vix_level`` and
  ``credit_spread_chg_20d`` need external series this POC does not fetch, and
  ``beta_252d`` needs an index benchmark — arrives as an all-NaN column. sklearn's
  histogram binner does not merely ignore such a column, it raises
  ``ValueError: window shape cannot be larger than input array shape`` from
  ``sliding_window_view`` on an empty array, which reads as a library bug rather than
  as "this feature has no data". Dropping them is correct; dropping them *silently*
  would hide the fact that six documented features are not being supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shingan.config import StructuredConfig
from shingan.logging_utils import get_logger
from shingan.models.persistence import load_model_payload, write_model

logger = get_logger(__name__)

#: Below this many positives in the calibration fold, isotonic regression is
#: over-parameterised (it fits a step function with as many steps as positives) and
#: Platt scaling is the honest choice. See docs/04-training.md section 3.
MIN_POSITIVES_FOR_ISOTONIC = 10

#: Probabilities are clipped away from the closed interval so that a log-loss or a
#: logit transform downstream cannot take a log of exactly zero.
_PROBABILITY_EPSILON = 1e-6

#: Name of the artifact written by :meth:`StructuredRiskModel.save`.
MODEL_FILENAME = "model.joblib"


@dataclass(slots=True)
class StructuredRiskModel:
    """Gradient-boosted risk model with an explicit calibration step.

    Args:
        config: The ``structured`` block of the project configuration.
    """

    config: StructuredConfig
    _estimator: Any = field(default=None, init=False, repr=False)
    _calibrator: Any = field(default=None, init=False, repr=False)
    _calibrator_kind: str = field(default="none", init=False)
    _feature_names: list[str] = field(default_factory=list, init=False)
    _base_rate: float = field(default=0.0, init=False)
    calibration_report: dict[str, Any] = field(default_factory=dict, init=False)
    #: Feature columns dropped because the training block held no observation of them.
    #: Named, not counted: "6 features were dropped" is not actionable, "vix_level was
    #: dropped" points at the missing data source.
    dropped_features: list[str] = field(default_factory=list, init=False)

    # -- fitting ----------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_valid: pd.DataFrame,
        y_valid: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> StructuredRiskModel:
        """Fit on train, then calibrate on validation.

        Args:
            X_train: Feature matrix, training block.
            y_train: Binary labels, training block.
            X_valid: Feature matrix, validation block. Used for calibration only —
                never for fitting the estimator.
            y_valid: Binary labels, validation block.
            sample_weight: Optional per-row weights for the training fit.

        Returns:
            ``self``, fitted.

        Raises:
            ValueError: If the training block holds fewer than two positives, which is
                the F8 condition from docs/05-evaluation.md section 10: the caller is
                supposed to skip the label rather than fit a model that cannot rank.
        """
        self._feature_names = [str(column) for column in X_train.columns]
        y_train = np.asarray(y_train, dtype=int).ravel()
        y_valid = np.asarray(y_valid, dtype=int).ravel()
        if int(y_train.sum()) < 2:
            raise ValueError(
                f"the training block holds {int(y_train.sum())} positive(s); a model "
                "cannot be fitted. This is the F8 condition — skip the label or widen "
                "the universe (docs/05-evaluation.md section 10)."
            )
        self._base_rate = float(y_train.mean())

        train_matrix = self._prepare(X_train)
        train_matrix = self._drop_unobserved(train_matrix)

        self._estimator = self._make_estimator()
        if sample_weight is not None and self.config.use_sample_weight:
            self._estimator.fit(train_matrix, y_train, sample_weight=np.asarray(sample_weight))
        else:
            self._estimator.fit(train_matrix, y_train)

        valid_matrix = self._prepare(X_valid, fitting=False)
        raw_valid = self._raw_scores(valid_matrix)
        self._calibrator, self._calibrator_kind, report = self._fit_calibrator(raw_valid, y_valid)
        report["dropped_features"] = list(self.dropped_features)
        if self.dropped_features:
            report["dropped_features_note"] = (
                f"{len(self.dropped_features)} feature(s) were dropped before fitting "
                "because the training block held no finite observation of them. This is "
                "a fact about the data sources, not about the model: these features are "
                "configured but nothing supplies them. Dropping is what keeps the "
                "histogram binner from raising on an empty column."
            )
        self.calibration_report = report
        return self

    def _drop_unobserved(self, matrix: np.ndarray) -> np.ndarray:
        """Remove columns the training block never observed, recording their names.

        The recorder is inside this method rather than at the call site so that the
        kept column order and :attr:`_feature_names` cannot drift apart: every caller
        needs the pruned matrix and the pruned names together, and returning one
        without the other is the way that pairing goes wrong.

        Raises:
            ValueError: If *every* column is unobserved. That is not a modelling
                problem, it is a pipeline that produced an empty feature matrix, and
                the histograms of a zero-column design are not the right place to say
                so.
        """
        observed = np.isfinite(matrix).any(axis=0)
        if observed.all():
            self.dropped_features = []
            return matrix
        dropped = [
            name for name, keep in zip(self._feature_names, observed, strict=True) if not keep
        ]
        if not observed.any():
            raise ValueError(
                f"none of the {len(self._feature_names)} feature columns holds a single "
                "finite value in the training block; the panel has no usable features. "
                "Check structured.feature_groups and that the builder produced those "
                "columns."
            )
        self.dropped_features = dropped
        self._feature_names = [
            name for name, keep in zip(self._feature_names, observed, strict=True) if keep
        ]
        logger.warning(
            "dropped %d feature(s) with no observation in the training block: %s",
            len(dropped),
            ", ".join(dropped),
        )
        return matrix[:, observed]

    def _make_estimator(self) -> Any:
        """Instantiate the configured estimator.

        ``hist_gbdt`` is the default for the reason docs/01-architecture.md gives:
        financial ratios and price features are tabular, of different units, with
        missing values, and a tree ensemble is stable, calibratable and attributable on
        exactly that shape of data.
        """
        params: dict[str, Any] = dict(self.config.params or {})
        if self.config.kind == "hist_gbdt":
            from sklearn.ensemble import HistGradientBoostingClassifier

            defaults: dict[str, Any] = {
                "max_iter": 300,
                "learning_rate": 0.06,
                "max_leaf_nodes": 15,
                # Small leaves: with a few hundred rows and tens of positives, a deep
                # tree fits the fold rather than the signal.
                "min_samples_leaf": 20,
                "l2_regularization": 1.0,
                # Deterministic and reproducible from the commit; the docs' single-seed
                # promise would otherwise not hold for this estimator.
                "random_state": int(params.pop("random_state", 0)),
                "early_stopping": False,
            }
            defaults.update(params)
            return HistGradientBoostingClassifier(**defaults)
        if self.config.kind == "logistic":
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            defaults = {
                "C": 1.0,
                "max_iter": 2000,
                "class_weight": "balanced",
                "random_state": int(params.pop("random_state", 0)),
            }
            defaults.update(params)
            return Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("model", LogisticRegression(**defaults)),
                ]
            )
        raise ValueError(
            f"unknown structured.kind {self.config.kind!r}; expected 'hist_gbdt' or 'logistic'"
        )

    def _prepare(self, X: pd.DataFrame, *, fitting: bool = True) -> np.ndarray:
        """Coerce a feature frame to a numeric matrix in the fitted column order.

        Columns are reindexed rather than positional-matched: a feature that exists at
        fit time and not at predict time is a missing column (NaN), not a shift.
        """
        if not isinstance(X, pd.DataFrame):
            matrix = np.asarray(X, dtype=float)
            if matrix.ndim == 1:
                matrix = matrix.reshape(-1, 1)
            return matrix
        if fitting:
            frame = X
        else:
            frame = X.reindex(columns=self._feature_names)
        return frame.to_numpy(dtype=float, na_value=np.nan)

    def _raw_scores(self, matrix: np.ndarray) -> np.ndarray:
        """Uncalibrated probabilities from the estimator."""
        if hasattr(self._estimator, "predict_proba"):
            scores = self._estimator.predict_proba(matrix)
            if scores.ndim == 2 and scores.shape[1] > 1:
                return scores[:, 1]
            return scores.ravel()
        decision = self._estimator.decision_function(matrix)
        return 1.0 / (1.0 + np.exp(-np.asarray(decision, dtype=float)))

    # -- calibration ------------------------------------------------------

    def _fit_calibrator(
        self, raw_valid: np.ndarray, y_valid: np.ndarray
    ) -> tuple[Any, str, dict[str, Any]]:
        """Fit the configured calibrator on the validation fold.

        Returns:
            ``(calibrator, kind, report)``. The kind is what the report prints, so a
            downgrade is visible rather than silent.
        """
        positives = int(np.asarray(y_valid, dtype=int).sum())
        # Key names are the ones `shingan.pipeline.assemble_report` reads
        # (`n_calibration_rows` / `n_calibration_positives`). They are not cosmetic:
        # report assembly validates that the calibration metadata is complete and
        # refuses to write a report without it, so a differently-named count silently
        # blocks the whole run at the last step rather than degrading one field.
        report: dict[str, Any] = {
            "configured": str(self.config.calibration),
            "n_calibration_rows": int(len(y_valid)),
            "n_calibration_positives": positives,
        }
        if self.config.calibration == "none":
            report["calibrator"] = "none"
            report["note"] = "structured.calibration is 'none'; scores are raw model outputs"
            return None, "none", report

        if positives == 0 or positives == len(y_valid):
            report["calibrator"] = "prior"
            report["note"] = (
                f"the calibration fold is single-class ({positives} positives of "
                f"{len(y_valid)}); no calibrator can be fitted, so scores are the "
                "training base rate"
            )
            return None, "prior", report

        kind = str(self.config.calibration)
        if kind == "isotonic" and positives < MIN_POSITIVES_FOR_ISOTONIC:
            report["note"] = (
                f"isotonic downgraded to platt: the calibration fold holds {positives} "
                f"positives, below the {MIN_POSITIVES_FOR_ISOTONIC} an isotonic fit needs"
            )
            kind = "platt"

        if kind == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            calibrator.fit(raw_valid, y_valid)
            report["calibrator"] = "isotonic"
            return calibrator, "isotonic", report

        # Platt scaling: a one-variable logistic regression on the logit of the score.
        from sklearn.linear_model import LogisticRegression

        logits = _logit(raw_valid).reshape(-1, 1)
        if len(np.unique(y_valid)) < 2:  # pragma: no cover - guarded above
            report["calibrator"] = "prior"
            return None, "prior", report
        calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        calibrator.fit(logits, y_valid)
        report["calibrator"] = "platt"
        return calibrator, "platt", report

    # -- prediction -------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated probabilities for the positive class.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._estimator is None:
            raise RuntimeError("StructuredRiskModel.predict_proba called before fit()")
        raw = self._raw_scores(self._prepare(X, fitting=False))
        return self._apply_calibrator(raw)

    def _apply_calibrator(self, raw: np.ndarray) -> np.ndarray:
        """Map raw scores through the fitted calibrator, robustly."""
        raw = np.asarray(raw, dtype=float)
        if self._calibrator is None:
            if self._calibrator_kind == "prior":
                scores = np.full_like(raw, self._base_rate, dtype=float)
            else:
                scores = raw
        elif self._calibrator_kind == "isotonic":
            scores = np.asarray(self._calibrator.predict(raw), dtype=float)
        else:
            scores = self._calibrator.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]
        return np.clip(scores, _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON)

    # -- persistence ------------------------------------------------------

    def save(self, directory: Path | str) -> Path:
        """Persist the fitted estimator, calibrator and metadata.

        The saved object is the one that was evaluated, never a refit: docs/05-evaluation.md
        requires that a shipped model and the numbers quoted for it cannot drift apart.

        Returns the directory, so ``StructuredRiskModel.load(model.save(d))`` round-trips.
        """
        import joblib

        destination = write_model(
            directory,
            MODEL_FILENAME,
            {
                "config": self.config.model_dump(),
                "estimator": self._estimator,
                "calibrator": self._calibrator,
                "calibrator_kind": self._calibrator_kind,
                "feature_names": self._feature_names,
                "base_rate": self._base_rate,
                "calibration_report": self.calibration_report,
                "dropped_features": list(self.dropped_features),
            },
        )
        logger.info("wrote structured model to %s", destination / MODEL_FILENAME)
        return destination

    @classmethod
    def load(cls, directory: Path | str) -> StructuredRiskModel:
        """Load a model written by :meth:`save`, from the directory ``save`` returned."""
        payload = load_model_payload(
            directory,
            MODEL_FILENAME,
            owner=cls.__name__,
            required=("config", "estimator", "calibrator", "calibrator_kind", "feature_names"),
        )
        model = cls(StructuredConfig(**payload["config"]))
        model._estimator = payload["estimator"]
        model._calibrator = payload["calibrator"]
        model._calibrator_kind = payload["calibrator_kind"]
        model._feature_names = list(payload["feature_names"])
        model._base_rate = float(payload["base_rate"])
        model.calibration_report = dict(payload["calibration_report"])
        model.dropped_features = list(payload.get("dropped_features", []))
        return model


def _logit(probabilities: np.ndarray) -> np.ndarray:
    """Log-odds with the probabilities clipped away from 0 and 1."""
    clipped = np.clip(np.asarray(probabilities, dtype=float), _PROBABILITY_EPSILON, 1 - _PROBABILITY_EPSILON)
    return np.log(clipped / (1.0 - clipped))


__all__ = ["MIN_POSITIVES_FOR_ISOTONIC", "MODEL_FILENAME", "StructuredRiskModel"]
