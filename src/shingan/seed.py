"""Determinism helpers.

A risk model whose results move between runs cannot be reviewed, so the project
takes the awkward position that *every* stochastic path is seeded, including the
ones that "obviously" do not matter. Resampling, synthetic generation, model
initialisation, data ordering and train/validation splits all take an explicit
seed.

Two rules this module exists to enforce:

1. Use :func:`rng`, which returns a :class:`numpy.random.Generator`. The legacy
   global ``numpy.random.seed`` API is not used anywhere in this package.
2. Do not rely on Python's hash randomisation. String hashing is salted per
   process, so any ordering derived from ``set`` iteration is a latent source of
   irreproducibility; :func:`set_global_seed` records the limitation rather than
   pretending to fix it.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

from shingan.logging_utils import get_logger

logger = get_logger(__name__)

#: Default seed. 13 is arbitrary; the point is that it is a constant rather than
#: "whatever the clock said", and that every config file uses the same value so
#: two runs of two different entry points remain comparable.
DEFAULT_SEED = 13


def rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Return a fresh numpy generator.

    Args:
        seed: Seed for the generator.

    Returns:
        A new :class:`numpy.random.Generator` backed by PCG64. Prefer a locally
        created generator over a module-level one: a shared generator makes
        results depend on call order, which is exactly the kind of hidden
        coupling that breaks reproducibility when code is refactored.
    """
    return np.random.default_rng(seed)


def set_global_seed(seed: int = DEFAULT_SEED, *, seed_torch: bool = True) -> None:
    """Seed the global random state of the libraries that have one.

    Call this once at the start of any entry point that trains a model. It seeds
    Python's :mod:`random`, numpy's legacy global state, and — when available —
    torch (CPU and all CUDA devices, plus ``torch.use_deterministic_algorithms``).

    Torch is imported lazily and only if it is already in ``sys.modules`` or can
    be imported; the CPU-only install does not have it and must not fail here.

    Note:
        This deliberately does **not** attempt to set ``PYTHONHASHSEED``. That
        variable is read by the interpreter at startup, so setting it from
        inside a running process has no effect on that process. Code in this
        package therefore avoids depending on hash order — see
        :func:`stable_order`.

    Args:
        seed: Seed applied to every library.
        seed_torch: Set to False to skip torch entirely, which is useful when you
            want to time a run without the (small) overhead of the CUDA sync.
    """
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError(f"seed must be in [0, 2**32 - 1], got {seed}")

    random.seed(seed)
    np.random.seed(seed)

    if not seed_torch:
        return

    torch: Any | None = None
    try:  # pragma: no cover - exercised only where torch is installed
        import torch as _torch

        torch = _torch
    except ImportError:
        torch = None

    if torch is None:
        return  # pragma: no cover - the CPU-only install takes this branch

    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - requires a CUDA device
        torch.cuda.manual_seed_all(seed)
        # Deterministic kernels are slower and some are unavailable on some
        # architectures; `warn_only=True` keeps a Blackwell-only kernel from
        # aborting a training run over a determinism guarantee we cannot keep.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.debug("torch RNG seeded with %d (cuda available=%s)", seed, torch.cuda.is_available())


def stable_order(values: Any) -> list[Any]:
    """Return the elements of ``values`` in a deterministic order.

    ``set`` iteration order depends on string hash randomisation, which is salted
    per process, so any code path that iterates a set produces different (though
    equally valid) output on each run. Sorting first fixes a canonical order.

    Args:
        values: Any iterable of mutually comparable items, typically strings.

    Returns:
        A sorted list. Mixed-type iterables raise ``TypeError`` from ``sorted``,
        which is preferable to silently falling back to insertion order.
    """
    return sorted(values)


def describe_environment() -> dict[str, Any]:
    """Collect a small, safe snapshot of the runtime for reproducibility records.

    Returned values are embedded in run metadata so a report can be tied back to
    the environment that produced it. Nothing here is secret and nothing requires
    a GPU: the torch entries are populated only when torch is importable, since
    the point of the record is to be writable on any machine.

    Returns:
        Mapping with the Python and numpy versions, the platform string, and —
        when torch is present — its version, the CUDA version it was built
        against, and the detected device name and compute capability.
    """
    import platform
    import sys

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "executable": sys.executable,
    }

    try:  # pragma: no cover - depends on the local environment
        import pandas as pd

        info["pandas"] = pd.__version__
    except ImportError:  # pragma: no cover
        pass

    try:  # pragma: no cover - depends on the local environment
        import sklearn

        info["scikit_learn"] = sklearn.__version__
    except ImportError:  # pragma: no cover
        pass

    torch: Any | None
    try:  # pragma: no cover - depends on the local environment
        import torch as _torch

        torch = _torch
    except ImportError:
        torch = None

    if torch is not None:  # pragma: no cover - requires torch
        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
            info["compute_capability"] = ".".join(
                str(part) for part in torch.cuda.get_device_capability(0)
            )
            info["device_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
            )

    return info


def seed_worker(worker_id: int) -> None:  # pragma: no cover - requires a DataLoader
    """Seed a torch ``DataLoader`` worker process deterministically.

    Pass as ``DataLoader(worker_init_fn=seed_worker, generator=g)``. Without this,
    each worker inherits an unpredictable numpy state and augmentation becomes
    irreproducible. Note that on Windows the default ``num_workers=0`` means this
    is never called, which is one more reason the default is 0.
    """
    del worker_id  # the value is unused; numpy derives a distinct state per worker
    worker_seed = np.random.randint(0, 2**32 - 1)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    os.environ.setdefault("SHINGAN_WORKER_SEED", str(worker_seed))
