# scripts/

Wrapper scripts for the two things you cannot do with a one-line `pip` command: set up an
environment on Windows, and run the POC without remembering the argument order. Plus one
script that is not a wrapper: [`smoke_train.py`](#smoke_trainpy-does-the-training-path-still-run-on-this-machine),
which runs the whole training path on a tiny model so a broken one is found in a minute
rather than after a model download.

They are thin. Every one of them ends up calling `python -m shingan`, and the command it
ran is printed before it runs, so the script is never the only record of what happened.

## The wrapper scripts

| Script | What it does |
| --- | --- |
| `bootstrap.ps1` / `bootstrap.sh` | Creates the virtual environment and installs dependencies, including the CUDA PyTorch wheel. |
| `run_poc.ps1` / `run_poc.sh` | Runs `data build` → `train structured` → `eval run` → `eval report`. |

Both come in a PowerShell and a POSIX flavour. The PowerShell pair is the supported path on
the Windows 11 + RTX 5090 training box; `docs/06-windows-setup.md` is the long-form version
of what `bootstrap.ps1` automates.

```powershell
# Windows, first time
powershell -ExecutionPolicy ByPass -File scripts\bootstrap.ps1
powershell -ExecutionPolicy ByPass -File scripts\bootstrap.ps1 -Train   # before training the LoRA
powershell -ExecutionPolicy ByPass -File scripts\run_poc.ps1
```

```bash
# macOS / Linux
bash scripts/bootstrap.sh
bash scripts/run_poc.sh
```

## `smoke_train.py`: does the training path still run on this machine?

Not a wrapper — a check. It is also the one script here with no PowerShell/POSIX pair,
because it has no platform-specific work to do.

```powershell
.venv\Scripts\python.exe scripts\smoke_train.py --keep
```

The trainer's arguments belong to `transformers`/`trl`; this project's YAML belongs to us.
When the two disagree, the error arrives at `SFTConfig(...)` — which in a real run happens
*after* the base model has been downloaded and quantised. That cost one run: `transformers`
v5 removed `warmup_ratio`, and the symptom was a `TypeError` 2h05m in.

The script slices a few rows from the real SFT files, writes an overlay pointing at
`trl-internal-testing/tiny-Qwen3ForCausalLM` (same model family as the base model, a few MB),
and runs `python -m shingan train lora` on it: configuration merge → argument mapping →
`SFTTrainer` → one optimiser step → adapter → `run.json`. About a minute, no GPU.

It then checks the artifacts rather than the exit code — `run.json` must report
`warmup_steps` and no `warmup_ratio`, the adapter weights must exist, and the training-stack
versions must be recorded — because a run that trained nothing must not report success.
`--keep` leaves the temporary directory in place for inspection.

Reasoning and the class of bug this closes: section 2.6 of
[`docs/04-training.md`](../docs/04-training.md).

## Make equivalence

The `Makefile` targets do the same work, but GNU Make is not installed on Windows by
default. Use whichever you have.

| Make target | Script equivalent |
| --- | --- |
| `make bootstrap` | `bootstrap.ps1` / `bootstrap.sh` |
| `make build train-structured eval report` | `run_poc.ps1` / `run_poc.sh` |
| `make doctor` | `python -m shingan doctor` |
| `make lint typecheck test` | no wrapper; see the `Makefile` |

## What `bootstrap` does that `pip install` does not

**It installs PyTorch with `--index-url`, not `--extra-index-url`.** The extra-index form
appends a candidate source instead of replacing it, so pip still sees PyPI's higher-numbered
CPU-only wheel and prefers it. The install then succeeds, `import torch` works, and the
failure appears much later as `CUDA error: no kernel image is available` — or, more
confusingly, `torch.cuda.is_available() == False` on a machine with a perfectly good GPU.

**It refuses a `+cpu` build.** After installing, it checks `torch.version.cuda` and exits
non-zero with the fix if PyTorch is a CPU-only build. This is worth the extra step because
the symptom is so far from the cause. `-Cpu` (`--cpu`) skips both the CUDA install and the
check, for machines with no NVIDIA GPU.

**It uses `.venv\Scripts\python.exe` rather than activating the environment.** Activation
is subject to the PowerShell execution policy, which is a per-machine setting this script
cannot assume; calling the interpreter by path sidesteps it entirely.

## Selecting a data configuration

`run_poc` takes `--data-config` (`-DataConfig`). The two that ship are not equivalent, and
the difference matters for how you read the report:

| Overlay | Universe | `default_risk` | `fraud_risk` | `tail_risk` |
| --- | --- | --- | --- | --- |
| `configs/data/poc_largecap10.yaml` (default) | 10 large caps, full span | not evaluated (valid fold has 0 positives) | not evaluated (no positives anywhere) | evaluated |
| `configs/data/demo.yaml` | 10 names, raised event rates | evaluated | evaluated | evaluated |

So **the default configuration produces a report in which only one of the three labels is
evaluated.** That is not a failure: a 10-name universe with a 3-year validation block and a
1-year horizon does not contain enough downgrades to calibrate on, and the report says so in
its Caveats section, quoting the positive counts. The demo overlay exists to exercise the
whole pipeline end to end, and it pays for that by using event rates that are higher than
the label documentation specifies — which is why it is a separate file rather than the
default.

```bash
# the demonstration stack, every label fits
bash scripts/run_poc.sh --data-config configs/data/demo.yaml --out artifacts/demo

# one label only, when the others cannot fit
bash scripts/run_poc.sh --labels tail_risk
```

## Reading the output

`run_poc` ends with `eval report`, which re-runs the pipeline from the configuration
snapshot stored in the run's JSON and compares the headline AUCs against the stored ones. It
prints `re-rendered metrics match the stored JSON` when they agree. The JSON is the record;
the Markdown is a rendering of it.

Anything the report could not evaluate reads as `not evaluated` or `undefined` with a
reason, never as a blank cell. A blank cell reads as a pass, so the report never leaves one.
