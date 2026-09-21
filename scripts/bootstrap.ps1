#Requires -Version 5.1
<#
.SYNOPSIS
    Create the project virtual environment and install Shingan's dependencies.

.DESCRIPTION
    The Windows counterpart to `make bootstrap`. It performs the same installation and
    adds the two steps the Makefile cannot express:

      * PyTorch must be installed from PyTorch's own cu128 index with --index-url, not
        --extra-index-url. The extra-index form appends a candidate source, so pip still
        prefers the higher-numbered CPU-only wheel on PyPI and the install "succeeds"
        with a `+cpu` build. See docs/06-windows-setup.md section 3.
      * A `+cpu` torch build is rejected outright by -Verify below. It imports cleanly,
        reports torch.cuda.is_available() == False or crashes on the first kernel, and
        the failure surfaces far from its cause.

    The environment is used through .venv\Scripts\python.exe rather than by activating
    it, so the script works regardless of the execution policy that governs
    Activate.ps1.

.PARAMETER PythonVersion
    Interpreter version passed to the `py` launcher. Default 3.12, which is what CI and
    docs/06-windows-setup.md target. The package supports 3.11 through 3.13.

.PARAMETER Venv
    Virtual environment directory, relative to the repository root. Default .venv.

.PARAMETER Train
    Also install the `train` extra (torch, transformers, peft, trl, datasets,
    accelerate, bitsandbytes, ...). Needed for `shingan train lora` and nothing else.

.PARAMETER Cpu
    Skip the CUDA wheel installation and let the `train` extra resolve torch from PyPI.
    Use on a machine without an NVIDIA GPU; the LoRA path will not run there.

.PARAMETER Force
    Delete and recreate the environment if it already exists.

.EXAMPLE
    powershell -ExecutionPolicy ByPass -File scripts\bootstrap.ps1

    Core + dev dependencies, CUDA torch. The usual first command on the 5090 box.

.EXAMPLE
    powershell -ExecutionPolicy ByPass -File scripts\bootstrap.ps1 -Train

    Adds the training stack. Required before `shingan train lora`.

.EXAMPLE
    powershell -ExecutionPolicy ByPass -File scripts\bootstrap.ps1 -Cpu -Force

    Rebuild from scratch without CUDA, e.g. on a laptop.
#>
[CmdletBinding()]
param(
    [string] $PythonVersion = "3.12",
    [string] $Venv = ".venv",
    [switch] $Train,
    [switch] $Cpu,
    [switch] $Force
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# The panel and the JSONL splits carry non-ASCII text (company names, filing excerpts).
# Without this, PowerShell 5.1 pipes through the console code page and mangles it.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$PyTorchIndex = "https://download.pytorch.org/whl/cu128"
$VenvPath = Join-Path $RepoRoot $Venv
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

function Write-Step {
    param([string] $Message)
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-Checked {
    param([string] $Label, [scriptblock] $Command)
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

# --- 1. virtual environment --------------------------------------------------

if ($Force -and (Test-Path $VenvPath)) {
    Write-Step "Removing the existing environment at $Venv"
    Remove-Item -Recurse -Force $VenvPath
}

if (-not (Test-Path $VenvPython)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        Write-Step "Creating $Venv with py -$PythonVersion"
        Invoke-Checked "venv creation" { py "-$PythonVersion" -m venv $Venv }
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        Write-Step "Creating $Venv with python (the 'py' launcher was not found)"
        Invoke-Checked "venv creation" { python -m venv $Venv }
    }
    else {
        throw "Neither 'py' nor 'python' is on PATH. Install Python $PythonVersion first."
    }
}
else {
    Write-Step "Reusing the environment at $Venv"
}

if (-not (Test-Path $VenvPython)) {
    throw "Expected an interpreter at $VenvPython but found none."
}

# --- 2. packaging tools ------------------------------------------------------

Write-Step "Upgrading pip, setuptools and wheel"
Invoke-Checked "pip upgrade" {
    & $VenvPython -m pip install --upgrade pip setuptools wheel
}

# --- 3. torch, from the CUDA index ------------------------------------------

if ($Cpu) {
    Write-Step "Skipping the CUDA wheel (-Cpu was given; torch will come from PyPI)"
}
else {
    Write-Step "Installing torch from $PyTorchIndex"
    Write-Host "    --index-url replaces the source entirely. --extra-index-url would" -ForegroundColor DarkGray
    Write-Host "    leave PyPI in the running, and pip would pick its CPU-only wheel." -ForegroundColor DarkGray
    Invoke-Checked "torch install" {
        & $VenvPython -m pip install --index-url $PyTorchIndex "torch>=2.7.0" torchvision torchaudio
    }
}

# --- 4. the package itself ---------------------------------------------------

$Extras = if ($Train) { ".[dev,train,parquet,plots]" } else { ".[dev]" }
Write-Step "Installing $Extras"
Invoke-Checked "package install" {
    & $VenvPython -m pip install -e $Extras
}

# `.[train]` resolves torch from PyPI and can overwrite the cu128 wheel installed above.
# The wheel from PyTorch's index normally has the higher version, so this is rare, but a
# silent downgrade would be discovered much later and somewhere else.
if ($Train -and -not $Cpu) {
    Write-Step "Confirming the training extra did not replace the CUDA wheel"
    Invoke-Checked "torch check" {
        & $VenvPython -c "import sys, torch; sys.exit(0 if torch.version.cuda else 1)"
    }
}

# --- 5. verify ---------------------------------------------------------------

Write-Step "Interpreter and CUDA report"
& $VenvPython -c @'
import platform
import sys

print("python:", platform.python_version(), "on", platform.system())

try:
    import torch
except ImportError:
    # Only reachable with -Cpu, where nothing installed torch. Reported rather than
    # traced: a missing torch is the expected state there, not a failure.
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
'@

# A `+cpu` build passes `import torch` and fails only at the first kernel launch, which
# is a long way from the install that caused it. Refuse it here instead.
if (-not $Cpu) {
    & $VenvPython -c @'
import sys

import torch

if "+cpu" in torch.__version__ or torch.version.cuda is None:
    sys.exit(
        "installed torch is a CPU-only build ("
        + torch.__version__
        + "). Re-run this script without -Cpu, or install manually with:\n"
        + "  .venv\\Scripts\\python.exe -m pip install --index-url "
        + "https://download.pytorch.org/whl/cu128 \"torch>=2.7.0\""
    )
'@
    if ($LASTEXITCODE -ne 0) {
        throw "torch verification failed; see the message above."
    }
}

Write-Step "Done"
Write-Host ""
Write-Host "  Activate with:  $Venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "  Health check:   $Venv\Scripts\python.exe -m shingan doctor" -ForegroundColor Green
Write-Host "  End-to-end POC: powershell -ExecutionPolicy ByPass -File scripts\run_poc.ps1" -ForegroundColor Green
Write-Host ""
