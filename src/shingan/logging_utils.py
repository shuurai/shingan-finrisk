"""Logging configuration.

The pipeline is run both interactively (where colour and progress matter) and
from CI (where a stable, greppable, timestamped stream matters). One setup
function covers both: a Rich console handler for humans plus an optional plain
file handler for the record.

Importing this module never configures anything; :func:`setup_logging` is called
explicitly by the CLI entry points and by the test fixtures.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from rich.logging import RichHandler

CONSOLE_FORMAT = "%(message)s"
FILE_FORMAT = "%(asctime)s %(levelname)-8s %(name)s [%(filename)s:%(lineno)d] %(message)s"

#: Root logger name for the whole package. Everything under ``shingan.*``
#: inherits the handlers installed here.
ROOT_LOGGER_NAME = "shingan"

_configured = False


def setup_logging(
    level: int | str = logging.INFO,
    log_file: Path | str | None = None,
    *,
    force: bool = False,
) -> logging.Logger:
    """Configure the package root logger.

    Args:
        level: Threshold for the console handler. Accepts a level name such as
            ``"DEBUG"`` or a :mod:`logging` constant.
        log_file: Optional path for a plain-text log. Parent directories are
            created. Messages at ``level`` and above are written; file output is
            never colourised so it can be diffed.
        force: Reconfigure even if logging was already set up. Tests use this to
            avoid relying on import order.

    Returns:
        The configured logger, so callers can immediately emit a message.
    """
    global _configured

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    if _configured and not force:
        return logger

    # Remove handlers installed by a previous call so repeated setup in a single
    # process (pytest, or a notebook re-running a cell) does not duplicate output.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.handlers.clear()

    resolved_level = logging.getLevelName(level) if isinstance(level, str) else level
    if not isinstance(resolved_level, int):
        raise ValueError(f"unknown log level: {level!r}")

    console = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        markup=False,
        log_time_format="%H:%M:%S",
    )
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT))
    logger.addHandler(console)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
        logger.addHandler(file_handler)

    logger.setLevel(resolved_level)
    # The logger writes its own records and is the single entry point for the
    # package; propagating would double-log through the stdlib root logger.
    logger.propagate = False
    _configured = True

    # A stray handler on the stdlib root logger would duplicate Rich output.
    if not logging.getLogger().handlers:
        logging.getLogger().addHandler(logging.NullHandler())

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the package root.

    Args:
        name: Usually ``__name__``. When None, the package root logger is
            returned. Names that already sit under :data:`ROOT_LOGGER_NAME` are
            used verbatim; anything else is nested beneath it.

    Returns:
        A :class:`logging.Logger` that inherits the configured handlers.
    """
    if name is None:
        return logging.getLogger(ROOT_LOGGER_NAME)
    if name == ROOT_LOGGER_NAME or name.startswith(f"{ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def is_interactive() -> bool:
    """Whether stdout looks like an interactive terminal.

    Used to decide between a live progress bar and plain periodic logging.
    """
    return sys.stdout.isatty() and os.environ.get("TERM") != "dumb"
