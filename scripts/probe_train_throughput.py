#!/usr/bin/env python
"""Measure what one training step actually costs, before spending days on it.

Why this exists
---------------
The 2026-10-01 wide-pool run was launched on a projection: 23.5 s/step measured once at
a 4,096-token window, scaled by the 1.71x median token ratio of the new corpus, giving
~40 s/step and 5.5-7.4 h for one epoch. The run actually took **2,032-3,240 s/step** --
50 to 80 times the projection -- from its very first step, for 246 steps, across 6.5
days, and then died without writing a loss value or a single checkpoint. A projection
cannot be checked from inside the run it is wrong about.

This script is the check. It slices the real SFT corpus to a couple of optimisation
steps, runs the **production** command (`shingan train lora`) with the same data and
training overlays, and reports the rate the trainer itself measured, plus the VRAM and
host-memory high-water marks that decide whether the card could host the run at all.

It deliberately re-implements nothing: a probe that reimplements the training loop
measures the probe. What it adds is the slice, the sampling, and a verdict.

    python scripts/probe_train_throughput.py \
        --sft-dir data/processed/sft_stage2_wide \
        --train-config configs/train/qlora_qwen3_14b.yaml \
        --rows 32

Exit status is 0 when the measured rate is inside the baseline's tolerance and 1 when it
is not, so a runbook can gate on it. It never writes to the real output directory.
"""

from __future__ import annotations

import argparse
import ctypes
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

# The measured baseline, and the reason the tolerance is a factor rather than a number:
# two runs of the *same* configuration (artifacts/lora and artifacts/lora-contract-v2)
# differ by 13.7% (23.5 vs 26.7 s/step), so single-digit precision here would be a
# fiction. A 2x band still catches a 50x pathology on its first step.
BASELINE_S_PER_STEP = 23.5
BASELINE_SPREAD = 1.137  # the slower of the two identical-configuration runs
BASELINE_WINDOW_TOKENS = 4096
TOLERANCE_FACTOR = 2.0

# `transformers` renders progress as `  4%|##  | 20/457 [12:11:13<288:14:46, 2374.57s/it]`
# on stderr with carriage returns. Both the step counter and the rate the trainer
# measured come out of that line -- the rate is the trainer's own, not our stopwatch.
PROGRESS_RE = re.compile(r"(\d+)/(\d+) \[(\d+:\d+:\d+|\d+:\d+)<(\S+), ([\d.]+)(s/it|it/s)\]")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sft-dir", type=Path, default=Path("data/processed/sft_stage2_wide"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/sft_probe"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/lora-probe"))
    parser.add_argument("--train-config", type=Path, default=Path("configs/train/qlora_qwen3_14b.yaml"))
    parser.add_argument("--rows", type=int, default=32, help="train rows to keep; steps = rows / accumulate")
    parser.add_argument("--log", type=Path, default=Path("lora_probe.log"))
    parser.add_argument("--timeout-seconds", type=int, default=5400, help="abort after this long")
    parser.add_argument("--dry-run", action="store_true", help="slice and print the command, run nothing")
    return parser.parse_args(argv)


def slice_corpus(src: Path, dst: Path, rows: int) -> dict[str, int]:
    """Write the first ``rows`` lines of each split to ``dst``. Idempotent by overwrite."""
    dst.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "valid"):
        source = src / f"{split}.jsonl"
        if not source.is_file():
            raise SystemExit(f"missing {source}")
        kept = []
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                if len(kept) >= rows:
                    break
                if line.strip():
                    kept.append(line if line.endswith("\n") else line + "\n")
        (dst / f"{split}.jsonl").write_bytes("".join(kept).encode("utf-8"))
        counts[split] = len(kept)
    return counts


class _Memory(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def host_ram() -> tuple[float, float]:
    """Total and used host RAM in GiB. Host memory is not incidental to this machine."""
    status = _Memory()
    status.dwLength = ctypes.sizeof(_Memory)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    total = status.ullTotalPhys / 1024**3
    return total, total - status.ullAvailPhys / 1024**3


def gpu_memory() -> tuple[int, int]:
    """Used and total VRAM in MiB, or (-1, -1) when nvidia-smi cannot answer."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        used, total = (int(part) for part in out.splitlines()[0].split(","))
        return used, total
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return -1, -1


class Sampler(threading.Thread):
    """High-water marks while the child runs. A step that never finishes never reports."""

    def __init__(self, interval: float = 5.0) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak_vram = 0
        self.peak_ram = 0.0

    def run(self) -> None:
        while not self.stop_event.is_set():
            used, _ = gpu_memory()
            self.peak_vram = max(self.peak_vram, used)
            _, used_ram = host_ram()
            self.peak_ram = max(self.peak_ram, used_ram)
            self.stop_event.wait(self.interval)

    def stop(self) -> None:
        self.stop_event.set()
        self.join(timeout=self.interval + 5)


def stream_and_parse(process: subprocess.Popen, log_path: Path) -> dict[str, float]:
    """Echo the child verbatim while extracting the last progress line it emitted."""
    last: dict[str, float] = {}
    buffer = b""
    with log_path.open("wb") as log:
        while True:
            chunk = process.stdout.read(4096)  # type: ignore[union-attr]
            if not chunk:
                break
            log.write(chunk)
            log.flush()  # unbuffered so a killed probe still leaves its evidence behind
            buffer += chunk
            *complete, buffer = re.split(rb"[\r\n]", buffer)
            for raw in complete:
                line = raw.decode("utf-8", "replace")
                if line.strip():
                    print(line, flush=True)
                match = PROGRESS_RE.search(line)
                if match:
                    done, total, elapsed, _eta, rate, unit = match.groups()
                    last = {
                        "step": float(done),
                        "total": float(total),
                        "elapsed": float(_clock_seconds(elapsed)),
                        "s_per_step": float(rate) if unit == "s/it" else 1.0 / float(rate),
                    }
    return last


def _clock_seconds(text: str) -> float:
    """``12:11:13`` or ``54:00`` to seconds."""
    parts = [int(p) for p in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def verdict(s_per_step: float, window: int, tolerance: float) -> tuple[bool, str]:
    """Compare against the baseline scaled to this window, or say it cannot be compared."""
    if window <= 0:
        return False, "no window width given, so the baseline cannot be scaled to it"
    expected = BASELINE_S_PER_STEP * BASELINE_SPREAD * (window / BASELINE_WINDOW_TOKENS)
    limit = expected * tolerance
    if s_per_step <= limit:
        return True, f"{s_per_step:.1f} s/step is within {tolerance}x of {expected:.1f} s/step (baseline scaled to {window} tokens)"
    return False, (
        f"{s_per_step:.1f} s/step is {s_per_step / expected:.0f}x the {expected:.1f} s/step baseline "
        f"scaled to {window} tokens -- this is the 2026-10-01 failure mode, do not launch the full run"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    counts = slice_corpus(args.sft_dir, args.out, args.rows)
    accumulate = 16
    steps = -(-counts["train"] // accumulate)
    print(f"probe corpus: {counts['train']} train / {counts['valid']} valid rows -> {steps} step(s)")

    used, total = gpu_memory()
    ram_total, ram_used = host_ram()
    print(f"before: VRAM {used} / {total} MiB free {total - used} | host RAM {ram_used:.1f} / {ram_total:.1f} GiB")

    command = [
        sys.executable,
        "-u",  # a piped stdout is block-buffered, which is how the 2026-10-01 loss lines vanished
        "-m",
        "shingan",
        "train",
        "lora",
        "--train-config",
        str(args.train_config),
        "--train-file",
        str(args.out / "train.jsonl"),
        "--eval-file",
        str(args.out / "valid.jsonl"),
        "--output-dir",
        str(args.output_dir),
    ]
    print("command:", " ".join(command))
    if args.dry_run:
        return 0

    sampler = Sampler()
    sampler.start()
    started = time.perf_counter()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    try:
        last = stream_and_parse(process, args.log)
        process.wait(timeout=args.timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        print(f"\nprobe exceeded --timeout-seconds {args.timeout_seconds} and was killed", flush=True)
        last = {}
    finally:
        sampler.stop()
    wall = time.perf_counter() - started

    print(f"\nelapsed {wall:.1f}s wall clock | exit {process.returncode}")
    print(f"peak VRAM {sampler.peak_vram} MiB | peak host RAM {sampler.peak_ram:.1f} GiB")

    if not last:
        print("VERDICT: no progress line was produced, so no step completed and no rate can be reported")
        return 1

    measured = last["s_per_step"]
    print(f"trainer reported: step {last['step']:.0f}/{last['total']:.0f} at {measured:.1f} s/step")
    ok, message = verdict(measured, int(window_argument(args.train_config)), TOLERANCE_FACTOR)
    print(f"VERDICT: {'pass' if ok else 'FAIL'} -- {message}")
    return 0 if ok else 1


def window_argument(train_config: Path) -> float:
    """The window the probe actually trained at, read from the overlay it was given."""
    for line in train_config.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("max_seq_length:"):
            return float(stripped.split(":", 1)[1].strip())
    raise SystemExit(f"no max_seq_length in {train_config}, so the baseline cannot be scaled")


if __name__ == "__main__":
    raise SystemExit(main())
