#Requires -Version 5.1
<#
.SYNOPSIS
    Run the Shingan POC end to end on the honest configuration.

.DESCRIPTION
    The Windows counterpart to `make build train-structured eval report`, in one command
    and in the order the pipeline requires:

      1. data build        assemble the panel (as-of joins, features, three label families)
      2. train structured  fit the calibrated GBDT track and persist it
      3. eval run          fit all three tracks, evaluate on the test block, write the report
      4. eval report       re-render the Markdown from the run's JSON and check the two agree

    Step 4 is not decorative. It re-runs the pipeline from the configuration snapshot in
    the JSON and compares the headline AUCs; a mismatch means the report and its own
    payload have diverged, which is reported loudly rather than left to be discovered.

    This runs on CPU. It fits the structured track and the TF-IDF text baseline, not the
    LoRA. `shingan train lora` is the only command that needs the GPU, and `eval run`
    deliberately leaves the fine-tuned path out rather than pretending it ran.

    The data config defaults to the honest POC stack (10 large caps over the full panel
    span), not the demo overlay. The demo raises the synthetic event rates so that every
    label fits at least one positive in each split; its base rates are therefore not the
    ones the label documentation specifies, and a report produced with it is a
    demonstration rather than evidence. Use -DataConfig to select it deliberately.

.PARAMETER Out
    Run directory for the report, panel and JSON payload. Default artifacts/poc.

.PARAMETER DataConfig
    Data overlay merged onto the base configuration. Default configs/data/poc_largecap10.yaml.

.PARAMETER Labels
    Comma-separated label names to evaluate. Defaults to every label in the configuration.

.PARAMETER SkipBuild
    Reuse the existing panel instead of rebuilding it. The panel is rebuilt in memory on
    every run regardless -- this only skips the CSV write.

.PARAMETER SkipTrain
    Do not fit and persist the structured track. The evaluation refits it anyway; persisting
    is what makes the saved artifact and the reported metrics the same model.

.EXAMPLE
    powershell -ExecutionPolicy ByPass -File scripts\run_poc.ps1

.EXAMPLE
    powershell -ExecutionPolicy ByPass -File scripts\run_poc.ps1 -Out artifacts\demo -DataConfig configs\data\demo.yaml

    The demonstration stack: every label fits, at the cost of unrealistically high base rates.

.EXAMPLE
    powershell -ExecutionPolicy ByPass -File scripts\run_poc.ps1 -Labels default_risk,tail_risk

    Two labels only. Useful when fraud_risk cannot fit for lack of positives.
#>
[CmdletBinding()]
param(
    [string] $Out = "artifacts/poc",
    [string] $DataConfig = "configs/data/poc_largecap10.yaml",
    [string] $Labels = "",
    [switch] $SkipBuild,
    [switch] $SkipTrain
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    $VenvPython = "python"
    Write-Host "note: no .venv found, using the interpreter on PATH" -ForegroundColor Yellow
}

function Write-Step {
    param([string] $Message)
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-Steps {
    param([string[]] $Arguments)
    Write-Host "    python -m shingan $($Arguments -join ' ')" -ForegroundColor DarkGray
    & $VenvPython -m shingan @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "shingan $($Arguments -join ' ') failed with exit code $LASTEXITCODE."
    }
}

# The data and eval overlays are explicit rather than left to the CLI defaults, so the
# command printed by this script is the whole command: nothing about the run depends on
# which configuration happens to be the default in this revision.
# `--config`, `--data-config` and friends are options of each *subcommand*, not of the
# top-level entry point, so they follow `data build` / `eval run` rather than preceding
# them. Passed before the subcommand they fail with "No such option: --config".
$Common = @("--config", "configs/default.yaml", "--data-config", $DataConfig)
$LabelArgs = if ($Labels) { @("--labels", $Labels) } else { @() }

Write-Step "Shingan POC against $DataConfig"
Write-Host "    run directory: $Out" -ForegroundColor DarkGray
Write-Host ""

if (-not $SkipBuild) {
    Write-Step "1/4  data build -- assemble the panel"
    Invoke-Steps (@("data", "build") + $Common)
}

if (-not $SkipTrain) {
    Write-Step "2/4  train structured -- fit and persist the calibrated GBDT"
    Invoke-Steps (@("train", "structured") + $Common)
}

Write-Step "3/4  eval run -- fit all tracks, evaluate on the test block, write the report"
Invoke-Steps (@("eval", "run") + $Common +
    @("--eval-config", "configs/eval/default.yaml", "--run-dir", $Out) + $LabelArgs)

Write-Step "4/4  eval report -- re-render and check it against the stored JSON"
Invoke-Steps @("eval", "report", "--run-dir", $Out)

Write-Host ""
Write-Step "Done"
Get-ChildItem -Path $Out -Filter "*.md" -ErrorAction SilentlyContinue |
    ForEach-Object { Write-Host "  report:  $($_.FullName)" -ForegroundColor Green }
Get-ChildItem -Path $Out -Filter "*.json" -ErrorAction SilentlyContinue |
    ForEach-Object { Write-Host "  payload: $($_.FullName)" -ForegroundColor Green }
Write-Host ""
Write-Host "  The JSON is the record. Anything the Markdown claims should be traceable to it." -ForegroundColor DarkGray
Write-Host "  Falsification checks that could not be evaluated read as 'not_evaluated' with a" -ForegroundColor DarkGray
Write-Host "  reason -- that is a stated gap, not a pass." -ForegroundColor DarkGray
Write-Host ""
