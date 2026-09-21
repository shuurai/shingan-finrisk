"""Track B0: a TF-IDF bag-of-words baseline on the same text the LoRA sees.

Its only job is to answer the question the fine-tuned LLM has to answer before it is
worth its cost: *does a 14B parameter instruction-tuned model beat a linear model over
term counts on this input?* (docs/04-training.md section 2). A LoRA that loses to
TF-IDF is not a result about LLMs; it is a result about this dataset and this prompt.

Two behaviours worth stating because they are deliberate:

**Same input, same truncation.** The text handed in is the rendered user prompt, and it
is cut at ``max_chars`` — the same budget the LoRA's prompt obeys. Comparing models fed
different inputs would confound the model with the input.

**Empty corpora are survivable.** A structured-only run has no text at all (the SEC
document fetch may be blocked, or the news snapshot may be absent). TF-IDF raises
``ValueError: empty vocabulary`` there, which would take the whole evaluation down over
a path that simply has nothing to say. Instead the model records that it had no
vocabulary and returns the training base rate for every row, which the comparison table
then shows as a flat, uninformative text path — the honest picture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shingan.config import TextBaselineConfig
from shingan.logging_utils import get_logger
from shingan.models.persistence import load_model_payload, write_model

logger = get_logger(__name__)

_PROBABILITY_EPSILON = 1e-6

MODEL_FILENAME = "text_baseline.joblib"


@dataclass(slots=True)
class TextBaselineModel:
    """TF-IDF + logistic regression, calibrated the same way as the structured track.

    Args:
        config: The ``text_baseline`` block of the project configuration.
    """

    config: TextBaselineConfig
    _vectorizer: Any = field(default=None, init=False, repr=False)
    _classifier: Any = field(default=None, init=False, repr=False)
    _calibrator: Any = field(default=None, init=False, repr=False)
    _base_rate: float = field(default=0.0, init=False)
    calibration_report: dict[str, Any] = field(default_factory=dict, init=False)
    warnings: list[str] = field(default_factory=list, init=False)

    # -- fitting ----------------------------------------------------------

    def fit(
        self,
        text_train: pd.Series,
        y_train: np.ndarray,
        text_valid: pd.Series,
        y_valid: np.ndarray,
    ) -> TextBaselineModel:
        """Fit the vectoriser and classifier on train, then calibrate on validation."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        train_documents = _as_documents(text_train, self.config.max_chars)
        valid_documents = _as_documents(text_valid, self.config.max_chars)
        y_train = np.asarray(y_train, dtype=int).ravel()
        y_valid = np.asarray(y_valid, dtype=int).ravel()
        self._base_rate = float(y_train.mean()) if len(y_train) else 0.0

        if not any(document.strip() for document in train_documents):
            self.warnings.append(
                "no text was available in the training fold, so the text path has no "
                "vocabulary. This is expected in a structured-only run (no filing text "
                "fetched) and is reported rather than treated as a model result."
            )
            self.calibration_report = {
                "calibrator": "none",
                "reason": "empty training corpus",
                "n_documents": 0,
                "n_calibration_rows": int(len(y_valid)),
                "n_calibration_positives": int(y_valid.sum()),
            }
            return self

        self._vectorizer = TfidfVectorizer(
            max_features=int(self.config.max_features),
            ngram_range=(int(self.config.ngram_min), int(self.config.ngram_max)),
            min_df=int(self.config.min_df),
            sublinear_tf=True,
        )
        try:
            matrix_train = self._vectorizer.fit_transform(train_documents)
            matrix_valid = self._vectorizer.transform(valid_documents)
        except ValueError as exc:  # empty vocabulary, or a min_df that filtered it all
            self._vectorizer = None
            self.warnings.append(f"the vectoriser could not be fitted ({exc}); text path is flat")
            self.calibration_report = {
                "calibrator": "none",
                "reason": f"vectoriser failed: {exc}",
                "n_calibration_rows": int(len(y_valid)),
                "n_calibration_positives": int(y_valid.sum()),
            }
            return self

        if len(np.unique(y_train)) < 2:
            self.warnings.append(
                f"the training fold holds {int(y_train.sum())} positive(s); the text path "
                "is fitted as a constant"
            )
            self._classifier = None
            return self

        self._classifier = LogisticRegression(
            C=float(self.config.C), max_iter=2000, class_weight="balanced", solver="liblinear"
        )
        self._classifier.fit(matrix_train, y_train)

        raw_valid = self._classifier.predict_proba(matrix_valid)[:, 1]
        self._calibrator, kind, report = self._fit_calibrator(raw_valid, y_valid)
        report["vocabulary_size"] = int(len(self._vectorizer.vocabulary_))
        report["n_documents"] = int(len(train_documents))
        self.calibration_report = report
        if self.config.calibration == "none":
            self.calibration_report.setdefault("calibrator", "none")
        return self

    def _fit_calibrator(
        self, raw_valid: np.ndarray, y_valid: np.ndarray
    ) -> tuple[Any, str, dict[str, Any]]:
        """Platt scaling on the validation fold, or nothing when that is impossible."""
        report: dict[str, Any] = {
            "configured": str(self.config.calibration),
            "n_calibration_rows": int(len(y_valid)),
            "n_calibration_positives": int(np.asarray(y_valid).sum()),
        }
        if self.config.calibration == "none":
            report["calibrator"] = "none"
            return None, "none", report
        if len(np.unique(y_valid)) < 2:
            report["calibrator"] = "prior"
            report["note"] = (
                "the calibration fold is single-class, so no calibrator could be fitted; "
                "the text path scores the training base rate"
            )
            return None, "prior", report

        from sklearn.linear_model import LogisticRegression

        logits = _logit(raw_valid).reshape(-1, 1)
        calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        calibrator.fit(logits, y_valid)
        report["calibrator"] = "platt"
        return calibrator, "platt", report

    # -- prediction -------------------------------------------------------

    def predict_proba(self, texts: pd.Series) -> np.ndarray:
        """Calibrated probability of the positive class for each document."""
        documents = _as_documents(texts, self.config.max_chars)
        if self._vectorizer is None or self._classifier is None:
            # No vocabulary, or a single-class fit: the base rate is the only defensible
            # score, and returning it keeps the comparison table complete.
            base = self._base_rate if self._base_rate > 0 else 0.5
            return np.full(len(documents), np.clip(base, _PROBABILITY_EPSILON, 1 - _PROBABILITY_EPSILON))
        matrix = self._vectorizer.transform(documents)
        raw = self._classifier.predict_proba(matrix)[:, 1]
        if self._calibrator is None:
            if self.calibration_report.get("calibrator") == "prior":
                scores = np.full_like(raw, self._base_rate)
            else:
                scores = raw
        else:
            scores = self._calibrator.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]
        return np.clip(np.asarray(scores, dtype=float), _PROBABILITY_EPSILON, 1 - _PROBABILITY_EPSILON)

    # -- persistence ------------------------------------------------------

    def save(self, directory: Path | str) -> Path:
        """Persist the fitted vectoriser, classifier and calibrator.

        Returns the directory, matching :meth:`load`. The class used to take a file path
        and return it, and had no loader at all, so the Track B0 artifact could be written
        but never read back — the TF-IDF baseline was the one track whose numbers could not
        be reproduced from disk.
        """
        return write_model(
            directory,
            MODEL_FILENAME,
            {
                "config": self.config.model_dump(),
                "vectorizer": self._vectorizer,
                "classifier": self._classifier,
                "calibrator": self._calibrator,
                "base_rate": self._base_rate,
                "calibration_report": self.calibration_report,
            },
        )

    @classmethod
    def load(cls, directory: Path | str) -> TextBaselineModel:
        """Load a baseline written by :meth:`save`, from the directory ``save`` returned."""
        payload = load_model_payload(
            directory,
            MODEL_FILENAME,
            owner=cls.__name__,
            required=("config", "vectorizer", "classifier", "calibrator", "base_rate"),
        )
        model = cls(TextBaselineConfig(**payload["config"]))
        model._vectorizer = payload["vectorizer"]
        model._classifier = payload["classifier"]
        model._calibrator = payload["calibrator"]
        model._base_rate = float(payload["base_rate"])
        model.calibration_report = dict(payload["calibration_report"])
        return model


def _as_documents(texts: pd.Series | list[str], max_chars: int) -> list[str]:
    """Normalise a text column to a list of truncated strings.

    ``max_chars`` is the prompt budget from the configuration, so the baseline sees the
    same input as the LoRA rather than the whole document.
    """
    if isinstance(texts, pd.Series):
        values = texts.fillna("").astype(str).tolist()
    else:
        values = ["" if item is None else str(item) for item in texts]
    limit = int(max_chars)
    return [value[:limit] if limit > 0 else value for value in values]


def _logit(probabilities: np.ndarray) -> np.ndarray:
    """Log-odds with the probabilities clipped away from 0 and 1."""
    clipped = np.clip(np.asarray(probabilities, dtype=float), _PROBABILITY_EPSILON, 1 - _PROBABILITY_EPSILON)
    return np.log(clipped / (1.0 - clipped))


__all__ = ["MODEL_FILENAME", "TextBaselineModel"]
