"""Entry point for ``python -m shingan``.

The console script declared in ``pyproject.toml`` and this module both call
:func:`shingan.cli.main`, so the two invocation styles cannot drift apart. The
``__main__`` guard is required rather than stylistic: on Windows the ``spawn`` start
method re-imports this module in every child process, and without the guard each child
would re-enter the CLI and fork again.
"""

from __future__ import annotations

from shingan.cli import main

if __name__ == "__main__":
    main()
