"""Filesystem layout.

Every module resolves paths through here instead of hard-coding strings, so the
project can be relocated (or a run redirected into a scratch directory) by
changing exactly one value.

All paths are :class:`pathlib.Path` objects. Nothing joins path strings by hand
and nothing assumes a POSIX separator, because the reference development machine
is Windows 11.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Files whose presence marks the repository root. ``.git`` is included so the
#: root is still found in a source checkout where ``pyproject.toml`` was removed,
#: and ``pyproject.toml`` is included so the root is still found inside a wheel
#: build or an sdist unpacked without VCS metadata.
ROOT_MARKERS: tuple[str, ...] = ("pyproject.toml", ".git")

#: How many parent directories to walk before giving up. Guards against an
#: infinite loop if a marker exists at the filesystem root.
_MAX_WALK_UP = 40


def find_project_root(start: Path | None = None) -> Path:
    """Return the repository root that contains ``start``.

    Walks upwards from ``start`` (default: this file's directory) until a
    directory containing one of :data:`ROOT_MARKERS` is found.

    Args:
        start: Directory to begin the search from. Defaults to the directory
            containing this module.

    Returns:
        The resolved repository root.

    Raises:
        FileNotFoundError: If no marker is found within :data:`_MAX_WALK_UP`
            levels. This normally means the package is installed somewhere
            without a project checkout, in which case the caller should pass an
            explicit root instead.
    """
    current = (start or Path(__file__).resolve().parent).resolve()
    for _ in range(_MAX_WALK_UP):
        if any((current / marker).exists() for marker in ROOT_MARKERS):
            return current
        if current.parent == current:
            break
        current = current.parent
    raise FileNotFoundError(
        f"could not locate the project root above {start or Path(__file__)}; "
        f"expected a directory containing one of {ROOT_MARKERS}. "
        "Pass an explicit root instead."
    )


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Resolved directories used by the pipeline.

    Attributes are grouped by lifecycle: ``raw`` is write-once source data,
    ``interim`` and ``processed`` are regenerable, and ``artifacts`` holds model
    outputs and reports. Only ``raw`` is treated as irreplaceable.
    """

    root: Path

    # Source data, in increasing order of processing.
    data: Path
    raw: Path
    interim: Path
    processed: Path
    external: Path

    # Configuration and documentation, resolved for convenience in tooling.
    configs: Path
    docs: Path

    # Outputs.
    artifacts: Path
    models: Path
    reports: Path
    figures: Path
    cache: Path

    @classmethod
    def from_root(cls, root: Path | str | None = None) -> ProjectPaths:
        """Build the layout for a given root, defaulting to the detected one."""
        resolved = Path(root).resolve() if root is not None else find_project_root()
        data = resolved / "data"
        artifacts = resolved / "artifacts"
        return cls(
            root=resolved,
            data=data,
            raw=data / "raw",
            interim=data / "interim",
            processed=data / "processed",
            external=data / "external",
            configs=resolved / "configs",
            docs=resolved / "docs",
            artifacts=artifacts,
            models=artifacts / "models",
            reports=artifacts / "reports",
            figures=artifacts / "figures",
            cache=resolved / ".cache",
        )

    def ensure(self) -> ProjectPaths:
        """Create every directory in the layout, including nested parents.

        Safe to call repeatedly. ``data/`` and ``artifacts/`` are both gitignored,
        so on a fresh clone none of these exist yet.

        Returns:
            ``self``, so the call can be chained: ``paths = ProjectPaths.from_root().ensure()``.
        """
        for directory in (
            self.data,
            self.raw,
            self.interim,
            self.processed,
            self.external,
            self.artifacts,
            self.models,
            self.reports,
            self.figures,
            self.cache,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def dataset(self, name: str = "panel") -> Path:
        """Path of a processed dataset file, e.g. ``data/processed/panel.parquet``."""
        return self.processed / f"{name}.parquet"

    def run_dir(self, name: str) -> Path:
        """Path of a run output directory under ``artifacts/``."""
        return self.artifacts / name


def default_paths(root: Path | str | None = None) -> ProjectPaths:
    """Convenience wrapper around :meth:`ProjectPaths.from_root`."""
    return ProjectPaths.from_root(root)
