# Contributing to Shingan

Thanks for considering a contribution. This project sits in a domain where it is
unusually easy to fool yourself, so the contribution rules below are stricter
than in a typical ML repo. The one-sentence version: **never let a number enter
the project that a reviewer cannot reproduce.**

## What is most useful

| Kind of contribution | Notes |
| --- | --- |
| Reproducing the offline demo on your own machine | Most valuable early signal. Report the exact command and the full output, including your OS, Python version and GPU. |
| Real-data adapters that have actually been run | If you make `data/edgar.py`, `data/news.py` or `data/prices.py` work against live endpoints, that is a headline contribution. Say exactly what you fetched and when. |
| New label event sources | Public sources of record for rating actions, enforcement actions, restatements or audit opinions. Vendor obligations are not acceptable for the POC. |
| Evaluation critique | If a metric or split design in `docs/05-evaluation.md` is wrong, say so with the counter-example. |
| Bug reports with a reproducer | See the issue template; the `shingan doctor` output is required. |

## What will be declined

- Results computed on synthetic data presented as evidence about real markets.
- Any change that introduces look-ahead: future prices, post-`as_of` news,
  restatements observed before they were filed, or any column derived from
  `event_date`. See `docs/02-data.md` section 4.1.
- Binary "future up/down" as a risk label. That is alpha with terrible
  signal-to-noise, not risk. See `docs/03-labeling.md`.
- Committed data payloads, price series, vendor data, secrets or checkpoints.
- Claims that the model can be used for trading, credit decisions or audit
  conclusions. This is research and education software.

## Development setup

### Windows 11 (the reference machine)

```powershell
git clone https://github.com/shuurai/shingan.git
cd shingan
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest -q
```

The core install is CPU-only on purpose. If you need the GPU training extra, read
`docs/06-windows-setup.md` first — on an RTX 5090 you must install torch from the
cu128 index before `pip install -e ".[train]"`, or pip will quietly install the
CPU wheel.

### macOS / Linux

```bash
git clone https://github.com/shuurai/shingan.git
cd shingan
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest -q
```

With `uv`: `uv sync --extra dev && uv run pytest -q`. `uv` also understands the
pinned cu128 torch index declared in `pyproject.toml`, so `uv sync --extra train`
does the right thing on Blackwell hardware.

## The loop before you open a pull request

```bash
pre-commit install          # once
ruff check . && ruff format --check .
mypy src
pytest -q
shingan demo --out artifacts/demo    # the end-to-end smoke test
```

All four must be clean. CI runs the same commands on Linux and Windows against
Python 3.11 and 3.12, and it does not install the `train` extra (no GPU on
runners), so keep anything that imports torch behind a lazy import and let the
GPU tests skip instead of fail.

## Code standards

- Python 3.11+ syntax. Full type annotations on every function and method
  (`mypy` is configured with `disallow_untyped_defs`). Tests are exempt.
- Line length 100, double quotes, `ruff format` decides layout. Do not fight it.
- Pure functions for anything numeric, so it can be unit-tested without fixtures.
- Determinism is mandatory: every stochastic path takes a `seed` and every
  generator comes from `numpy.random.default_rng(seed)` — never the legacy global
  `numpy.random` API.
- Platform portability is mandatory: `pathlib` instead of string paths, always
  pass `encoding="utf-8"` to `open()`, never use `fcntl` or `os.fork`, and put a
  `if __name__ == "__main__":` guard in any module that constructs a
  `DataLoader` so Windows' `spawn` start method can re-import it safely.
- New data columns must be added to the data dictionary in `docs/02-data.md` in
  the same pull request. The dictionary is the contract between modules.

## Tests

- Every new metric needs a test with a **hand-computed expected value**, not a
  round-trip through the implementation.
- Every change to splitting or labelling needs a leakage test: assert that the
  new code cannot see the future, ideally by recomputing an output on truncated
  history and requiring bitwise-identical results.
- Mark anything needing network with `@pytest.mark.network` and anything needing
  a GPU with `@pytest.mark.gpu`. Neither may run in CI; use
  `pytest.skip(..., allow_module_level=True)` when the dependency is absent.

## Documentation

- User-facing docs go under `docs/` and must be linked from `docs/index.md`.
- Architectural decisions get a short ADR in `docs/adr/` using the existing
  Context / Decision / Consequences shape. Superseding a decision means adding a
  new ADR, not editing the old one.
- If your change makes a doc wrong, fixing the doc is part of the change, not a
  follow-up.

## Commits and pull requests

Conventional Commits prefixes are encouraged (`feat:`, `fix:`, `docs:`,
`test:`, `refactor:`, `chore:`). The PR template asks for the exact verification
commands you ran and their output; paste real output rather than describing it.

`main` is protected. Please open a pull request rather than pushing to it, and
rebase rather than merging `main` back into your branch.

## Licensing

By contributing you agree that your contribution is licensed under the
Apache License 2.0, the same terms as this repository. There is no separate
contributor licence agreement. Do not contribute code you do not have the right
to license, and do not contribute data whose terms forbid redistribution.
