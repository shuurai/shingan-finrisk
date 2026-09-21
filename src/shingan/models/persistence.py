"""One persistence contract, shared by all three model layers.

Each model in this package is written into a directory of its own and addressed by a
module-level ``MODEL_FILENAME`` constant. The contract is narrow on purpose, so that it
can be the same for every track::

    save(directory) -> the directory      # the directory, not the file
    load(directory) -> the model

Two failures this module exists to prevent. Both were live in this repository, both were
invisible to a test suite that only ever fitted and scored models in memory, and both
surfaced only when someone tried to reload an artifact:

1. **A writer that names the file itself.** ``cli train`` handed a bare object to
   ``joblib.dump`` under ``f"{path}.joblib"`` instead of calling the class's own ``save``.
   The artifact on disk was ``fused.joblib`` while :class:`~shingan.models.fusion.RiskFusion`
   loads ``fusion.joblib`` — so the whole fused track was unreachable from disk, with no
   error at fit time. The filename must come from the class constant, never from a caller
   that guesses.
2. **``save`` returning the file while ``load`` takes the directory.** The natural round
   trip ``load(save(d))`` then fails on a path like ``.../model.joblib/model.joblib``,
   which reads as a corrupt artifact rather than as a disagreement about the signature.

The readers therefore raise with the directory contents in the message. A bare
``FileNotFoundError`` naming a path the caller never chose is exactly the kind of error
that costs an afternoon.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["load_model_payload", "read_model_path", "write_model"]


def write_model(directory: Path | str, filename: str, payload: Any) -> Path:
    """Write ``payload`` to ``directory/filename``, returning the directory.

    Returning the directory rather than the file is what makes ``load(save(d))`` work,
    and it leaves room for the model card and metrics that a model directory is expected
    to hold next to the weights.
    """
    import joblib

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, destination / filename)
    return destination


def read_model_path(directory: Path | str, filename: str, *, owner: str) -> Path:
    """Return the model file inside ``directory``, or raise naming what was found.

    Args:
        directory: The model directory recorded next to the artifact. Also tolerates the
            file itself, which is rejected with the correct call rather than followed by
            a second ``FileNotFoundError`` downstream.
        filename: The ``MODEL_FILENAME`` constant of the loading class.
        owner: The class name, used as the subject of the error message.

    Raises:
        ValueError: If handed a file where a directory is required.
        FileNotFoundError: If the directory, or the file inside it, is absent. The
            directory case lists the ``.joblib`` files actually present, so a renamed or
            renamed-by-an-older-writer artifact is diagnosable without a shell.
    """
    target = Path(directory)
    if target.is_file():
        raise ValueError(
            f"{owner}.load() takes the model directory, but was given the file {target}. "
            f"Call {owner}.load({target.parent}) instead — the directory form is what "
            f"{owner}.save() returns."
        )
    if not target.exists():
        raise FileNotFoundError(
            f"{owner}.load() found no model directory at {target}. Fit and save a model "
            f"first; {owner}.save(directory) creates the directory and returns it."
        )
    path = target / filename
    if not path.exists():
        present = sorted(item.name for item in target.glob("*.joblib"))
        raise FileNotFoundError(
            f"{owner}.load() expected {filename} inside {target}, but that directory "
            f"holds: {', '.join(present) if present else 'no .joblib files'}. An artifact "
            f"written by an older writer may carry a different name; re-run the training "
            f"command to rewrite it under the current name."
        )
    return path


def load_model_payload(
    directory: Path | str, filename: str, *, owner: str, required: Sequence[str]
) -> dict[str, Any]:
    """Read a model's metadata dict, rejecting an artifact written by a different scheme.

    The readers in this package all unpack a dict of named fields. An artifact written by
    a writer that pickled the model *object* instead — which is what ``cli train`` used to
    do for the text and fused tracks — then fails as ``'TextBaselineModel' object is not
    subscriptable``, naming neither the file nor the cause. Checking the shape here turns
    that into a re-run instruction.

    Args:
        directory: Model directory, or the model file.
        filename: The ``MODEL_FILENAME`` constant of the loading class.
        owner: The class name, used as the subject of the message.
        required: Keys the payload must hold.

    Raises:
        ValueError: If the payload is not a dict, or is missing a required key.
    """
    import joblib

    path = read_model_path(directory, filename, owner=owner)
    payload = joblib.load(path)
    if not isinstance(payload, dict):
        raise ValueError(
            f"{owner}.load() expected a metadata dict in {path}, but found a "
            f"{type(payload).__name__}. An artifact pickled as a bare object was written by "
            f"an older version of the training command; re-run it to rewrite {filename} in "
            f"the current format."
        )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(
            f"{owner}.load() found {path} but it lacks {missing}; it holds "
            f"{sorted(payload)}. The artifact predates a change to the saved fields — "
            f"re-run the training command to rewrite it."
        )
    return payload
