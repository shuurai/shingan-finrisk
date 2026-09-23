"""One-minute end-to-end check that `shingan train lora` still runs on this machine.

**Why this exists.** The trainer's argument surface belongs to ``transformers``/``trl``,
and this project's YAML belongs to us. When the two disagree the failure arrives at
``SFTConfig(...)`` — which, in the real run, is *after* the base model has been
downloaded and quantised. A run was lost that way: `transformers` v5 removed
``warmup_ratio`` and the symptom was a ``TypeError`` 2h05m into a 25 GB download.

`tests/test_lora_arguments.py` checks the mapping against the *signature*. This script
checks something the signature cannot: that the whole path — config merge, dataset
preparation, chat-template application, trainer construction, one optimiser step,
adapter save, ``run.json`` — actually completes. It does that on a tiny model of the
same family as the real one, so it costs a few MB and about a minute instead of a GPU
day.

It is deliberately **not** a pytest test: it needs network and it is a smoke check to run
by hand before paying for a real run, not a gate on every commit.

    python scripts/smoke_train.py
    python scripts/smoke_train.py --keep          # leave the artifacts to inspect
    python scripts/smoke_train.py --examples 12   # more rows, still one epoch
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

#: A tiny model from the same family as the real base model (Qwen3). Same tokeniser
#: behaviour, same chat template, same module names for the LoRA targets — which matters,
#: because a LoRA adapter aimed at `q_proj` on a model that has no `q_proj` would fail
#: for a reason that has nothing to do with the code being checked.
DEFAULT_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"

#: The overlay applied on top of `configs/default.yaml`. Everything here is chosen to make
#: the run cheap, not to make it meaningful: one epoch, two examples per optimiser step,
#: 4-bit quantisation off (it needs a CUDA context and is not what is being checked).
OVERLAY = """\
# Written by scripts/smoke_train.py. Not a training configuration — a wiring check.
lora:
  base_model: {model}
  load_in_4bit: false
  bnb_4bit_compute_dtype: float32
  max_seq_length: 512
  num_train_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 2
  gradient_checkpointing: false
  optim: adamw_torch
  bf16: false
  fp16: false
  report_to: none
  logging_steps: 1
  save_total_limit: 1
  eval_strategy: epoch
"""


def _slice(source: Path, destination: Path, count: int) -> int:
    """Write the first ``count`` rows of ``source`` to ``destination``.

    Copying rows rather than sampling them keeps the label mix, the tickers and the
    split values intact — a smoke test that quietly invented its own rows would not be
    exercising `assert_no_test_rows`.
    """
    rows = source.read_text(encoding="utf-8").splitlines()[:count]
    # Bytes, not text: a Windows text-mode write would turn every LF into CRLF, and the
    # project normalises its data files to LF (see docs/04-training.md section 2.5).
    destination.write_bytes(("\n".join(rows) + "\n").encode("utf-8"))
    return len(rows)


def _versions(environment: dict[str, object]) -> str:
    """The training-stack versions from a `run.json` environment block, as one line."""
    return ", ".join(
        f"{name} {environment[name]}"
        for name in ("transformers", "trl", "peft", "datasets", "accelerate", "bitsandbytes")
        if name in environment
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model to smoke-test with.")
    parser.add_argument(
        "--examples", type=int, default=6, help="Training rows to slice from the real SFT file."
    )
    parser.add_argument(
        "--eval-examples", type=int, default=3, help="Validation rows (0 disables evaluation)."
    )
    parser.add_argument(
        "--root", default=None, help="Project root. Defaults to the repository this script is in."
    )
    parser.add_argument(
        "--keep", action="store_true", help="Keep the temporary directory and print its path."
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    sft = root / "data" / "processed" / "sft"
    train_file = sft / "train.jsonl"
    if not train_file.is_file():
        print(
            f"no SFT dataset at {train_file}\n"
            "  build it first:  .venv/Scripts/python -m shingan data sft",
            file=sys.stderr,
        )
        return 2

    workspace = Path(tempfile.mkdtemp(prefix="shingan-smoke-"))
    try:
        sliced_train = workspace / "train.jsonl"
        n_train = _slice(train_file, sliced_train, args.examples)

        eval_args: list[str] = []
        if args.eval_examples:
            valid_file = sft / "valid.jsonl"
            if valid_file.is_file():
                sliced_valid = workspace / "valid.jsonl"
                _slice(valid_file, sliced_valid, args.eval_examples)
                eval_args = ["--eval-file", str(sliced_valid)]

        overlay = workspace / "overlay.yaml"
        overlay.write_text(OVERLAY.format(model=args.model), encoding="utf-8")
        output_dir = workspace / "lora"

        command = [
            sys.executable,
            "-m",
            "shingan",
            "train",
            "lora",
            "--root",
            str(root),
            "--train-config",
            str(overlay),
            "--train-file",
            str(sliced_train),
            *eval_args,
            "--output-dir",
            str(output_dir),
        ]
        print("$ " + " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=root)
        if completed.returncode != 0:
            print(f"\ntrain lora exited {completed.returncode}", file=sys.stderr)
            return completed.returncode

        # The run has to have produced the things a real run produces. Without these
        # checks the script would report success for a run that trained nothing.
        run_json = output_dir / "run.json"
        if not run_json.is_file():
            print(f"\nno run.json at {run_json}", file=sys.stderr)
            return 1
        payload = json.loads(run_json.read_text(encoding="utf-8"))

        arguments = payload.get("trainer_arguments") or {}
        problems: list[str] = []
        if "warmup_ratio" in arguments:
            problems.append("trainer_arguments still carries warmup_ratio")
        if "warmup_steps" not in arguments:
            problems.append("trainer_arguments does not report the resolved warmup_steps")
        if payload.get("n_train_examples") != n_train:
            problems.append(
                f"run.json says {payload.get('n_train_examples')} training rows, sliced {n_train}"
            )
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            if not (output_dir / "adapter" / name).is_file():
                problems.append(f"no adapter/{name}")

        # The versions that decide which argument names are legal have to be in the record,
        # or a later failure cannot be explained from the artifact.
        environment = payload.get("environment") or {}
        unrecorded = [name for name in ("transformers", "trl", "peft") if name not in environment]
        if unrecorded:
            problems.append(f"run.json does not record the versions of {', '.join(unrecorded)}")

        if problems:
            print("\nsmoke failed:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1

        print("\nsmoke passed: config -> mapping -> trainer -> step -> adapter -> run.json")
        print(f"  base model        {payload['base_model']}")
        print(f"  train / eval rows {payload['n_train_examples']} / {payload['n_eval_examples']}")
        print(
            f"  trainer arguments {len(arguments)} keys, warmup_steps={arguments['warmup_steps']}"
        )
        print(f"  environment       {_versions(environment)}")
        return 0
    finally:
        if args.keep:
            print(f"kept: {workspace}")
        else:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
