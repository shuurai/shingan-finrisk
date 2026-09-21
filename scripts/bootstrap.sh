#!/usr/bin/env bash
#
# Create the project virtual environment and install Shingan's dependencies.
#
# The POSIX counterpart to `make bootstrap`. Same installation, plus the interpreter
# check the Makefile cannot express: the package supports Python 3.11 through 3.13, and
# a 3.14 interpreter fails deep inside a dependency rather than at the start.
#
# Usage:
#   bash scripts/bootstrap.sh                 # core + dev, CPU torch
#   bash scripts/bootstrap.sh --cuda          # torch from PyTorch's cu128 index
#   bash scripts/bootstrap.sh --train         # add the LoRA training stack
#   bash scripts/bootstrap.sh --train --cuda  # the Linux GPU box
#   bash scripts/bootstrap.sh --force         # recreate the environment
#
# Options:
#   --train        install the `train` extra (transformers, peft, trl, bitsandbytes, ...)
#   --cuda         install torch from https://download.pytorch.org/whl/cu128
#   --cpu          force CPU torch (the default on this platform; kept for symmetry)
#   --venv DIR     environment directory, default .venv
#   --python EXE   interpreter to build the environment from, default python3.12
#   --force        delete an existing environment first
#   -h, --help     this message
#
# On macOS, --cuda is meaningless: there is no CUDA backend. On Linux it is what you
# want on the training box, and --index-url is used rather than --extra-index-url for
# the reason in docs/06-windows-setup.md section 3 -- the extra-index form leaves PyPI in
# the running and pip then picks the higher-numbered CPU-only wheel.

set -euo pipefail

VENV=".venv"
PYTHON="python3.12"
TRAIN=0
CUDA=0
FORCE=0

PYTORCH_INDEX="https://download.pytorch.org/whl/cu128"

usage() {
    # Extracted by pattern rather than by line number, so the help cannot drift out of
    # step with the header block above it. The block is the run of `#` lines after the
    # shebang, up to the first blank line.
    sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//' | sed '/./,$!d'
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --train) TRAIN=1 ;;
        --cuda) CUDA=1 ;;
        --cpu) CUDA=0 ;;
        --force) FORCE=1 ;;
        --venv)
            shift
            VENV="${1:?--venv needs a directory}"
            ;;
        --python)
            shift
            PYTHON="${1:?--python needs an interpreter}"
            ;;
        -h | --help) usage 0 ;;
        *)
            echo "unknown option: $1" >&2
            usage 1
            ;;
    esac
    shift
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

step() { printf '\033[36m==> %s\033[0m\n' "$1"; }
die() { printf '\033[31merror: %s\033[0m\n' "$1" >&2; exit 1; }

VENV_PATH="$REPO_ROOT/$VENV"
VENV_PYTHON="$VENV_PATH/bin/python"

# --- 1. virtual environment --------------------------------------------------

if [ "$FORCE" -eq 1 ] && [ -d "$VENV_PATH" ]; then
    step "Removing the existing environment at $VENV"
    rm -rf "$VENV_PATH"
fi

if [ ! -x "$VENV_PYTHON" ]; then
    if ! command -v "$PYTHON" >/dev/null 2>&1; then
        # A bare `python3` is usually present but may be 3.9 or 3.14; try a versioned
        # search before giving up, since neither error is easy to read from pip.
        for candidate in python3.13 python3.11 python3; do
            if command -v "$candidate" >/dev/null 2>&1; then
                echo "note: $PYTHON not found, falling back to $candidate" >&2
                PYTHON="$candidate"
                break
            fi
        done
    fi
    command -v "$PYTHON" >/dev/null 2>&1 ||
        die "no usable interpreter found (looked for $PYTHON, python3.13, python3.11, python3)"

    version="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    case "$version" in
        3.11 | 3.12 | 3.13) ;;
        *) die "$PYTHON is Python $version; the package requires >=3.11,<3.14" ;;
    esac

    step "Creating $VENV with $PYTHON ($version)"
    "$PYTHON" -m venv "$VENV"
else
    step "Reusing the environment at $VENV"
fi

[ -x "$VENV_PYTHON" ] || die "expected an interpreter at $VENV_PYTHON but found none"

# --- 2. packaging tools ------------------------------------------------------

step "Upgrading pip, setuptools and wheel"
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

# --- 3. torch ----------------------------------------------------------------

if [ "$CUDA" -eq 1 ]; then
    step "Installing torch from $PYTORCH_INDEX"
    "$VENV_PYTHON" -m pip install --index-url "$PYTORCH_INDEX" \
        "torch>=2.7.0" torchvision torchaudio
else
    step "Leaving torch to the extras (pass --cuda to install from PyTorch's index)"
fi

# --- 4. the package itself ---------------------------------------------------

if [ "$TRAIN" -eq 1 ]; then
    EXTRAS=".[dev,train,parquet,plots]"
else
    EXTRAS=".[dev]"
fi
step "Installing $EXTRAS"
"$VENV_PYTHON" -m pip install -e "$EXTRAS"

# --- 5. verify ---------------------------------------------------------------

step "Interpreter and CUDA report"
"$VENV_PYTHON" - <<'PY'
import platform
import sys

print("python:", platform.python_version(), "on", platform.system())

try:
    import torch
except ImportError:
    print("torch: not installed (core-only environment)")
    sys.exit(0)

print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    capability = torch.cuda.get_device_capability(0)
    print("capability:", f"sm_{capability[0]}{capability[1]}")
    print("bf16 supported:", torch.cuda.is_bf16_supported())
    print("total vram GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
PY

if [ "$CUDA" -eq 1 ] && ! "$VENV_PYTHON" -c \
    'import sys, torch; sys.exit(0 if torch.version.cuda else 1)'; then
    # A +cpu build imports cleanly and fails only at the first kernel launch, which is a
    # long way from the install that caused it. Refuse it here instead.
    die "torch has no CUDA build. Reinstall with:
  $VENV_PYTHON -m pip install --index-url $PYTORCH_INDEX \"torch>=2.7.0\""
fi

step "Done"
echo ""
echo "  Activate with:  source $VENV/bin/activate"
echo "  Health check:   $VENV/bin/python -m shingan doctor"
echo "  End-to-end POC: bash scripts/run_poc.sh"
echo ""
