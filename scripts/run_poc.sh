#!/usr/bin/env bash
#
# Run the Shingan POC end to end on the honest configuration.
#
# The POSIX counterpart to `make build train-structured eval report`, in one command and
# in the order the pipeline requires:
#
#   1. data build        assemble the panel (as-of joins, features, three label families)
#   2. train structured  fit the calibrated GBDT track and persist it
#   3. eval run          fit all three tracks, evaluate on the test block, write the report
#   4. eval report       re-render the Markdown from the run's JSON and check the two agree
#
# Step 4 is not decorative: it re-runs the pipeline from the configuration snapshot in the
# JSON and compares the headline AUCs, so a divergence between a report and its own payload
# is reported rather than discovered later.
#
# This runs on CPU. Only `shingan train lora` needs the GPU, and `eval run` deliberately
# leaves the fine-tuned text path out instead of pretending it ran.
#
# The data config defaults to the honest POC stack (10 large caps over the full span), not
# the demo overlay. The demo raises synthetic event rates so that every label fits in every
# split; its base rates are not the ones the label documentation specifies, so a report made
# with it is a demonstration rather than evidence.
#
# Usage:
#   bash scripts/run_poc.sh
#   bash scripts/run_poc.sh --out artifacts/demo --data-config configs/data/demo.yaml
#   bash scripts/run_poc.sh --labels default_risk,tail_risk
#   bash scripts/run_poc.sh --skip-build --skip-train
#
# Options:
#   --out DIR            run directory, default artifacts/poc
#   --data-config PATH   data overlay, default configs/data/poc_largecap10.yaml
#   --labels LIST        comma-separated labels; default is every label in the config
#   --skip-build         skip the panel CSV write
#   --skip-train         skip persisting the structured model
#   -h, --help           this message

set -euo pipefail

OUT="artifacts/poc"
DATA_CONFIG="configs/data/poc_largecap10.yaml"
LABELS=""
SKIP_BUILD=0
SKIP_TRAIN=0

usage() {
    sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//' | sed '/./,$!d'
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --out)
            shift
            OUT="${1:?--out needs a directory}"
            ;;
        --data-config)
            shift
            DATA_CONFIG="${1:?--data-config needs a path}"
            ;;
        --labels)
            shift
            LABELS="${1:?--labels needs a value}"
            ;;
        --skip-build) SKIP_BUILD=1 ;;
        --skip-train) SKIP_TRAIN=1 ;;
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

if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
else
    PYTHON="python3"
    echo "note: no .venv found, using $PYTHON" >&2
fi

step() { printf '\033[36m==> %s\033[0m\n' "$1"; }

# Every argument is spelled out rather than left to the CLI defaults, so the command this
# script prints is the whole command: nothing about the run depends on which configuration
# happens to be the default in this revision of the file.
#
# `--config`, `--data-config` and friends are options of each *subcommand*, not of the
# top-level entry point, so they follow `data build` / `eval run` rather than preceding
# them. Passed before the subcommand they fail with "No such option: --config".
COMMON=(--config configs/default.yaml --data-config "$DATA_CONFIG")
LABEL_ARGS=()
if [ -n "$LABELS" ]; then
    LABEL_ARGS=(--labels "$LABELS")
fi

run() {
    printf '\033[90m    %s -m shingan %s\033[0m\n' "$PYTHON" "$*"
    "$PYTHON" -m shingan "$@"
}

step "Shingan POC against $DATA_CONFIG"
printf '\033[90m    run directory: %s\033[0m\n' "$OUT"
echo ""

if [ "$SKIP_BUILD" -eq 0 ]; then
    step "1/4  data build -- assemble the panel"
    run data build "${COMMON[@]}"
fi

if [ "$SKIP_TRAIN" -eq 0 ]; then
    step "2/4  train structured -- fit and persist the calibrated GBDT"
    run train structured "${COMMON[@]}"
fi

step "3/4  eval run -- fit all tracks, evaluate on the test block, write the report"
run eval run "${COMMON[@]}" --eval-config configs/eval/default.yaml \
    --run-dir "$OUT" ${LABEL_ARGS[@]+"${LABEL_ARGS[@]}"}

step "4/4  eval report -- re-render and check it against the stored JSON"
run eval report --run-dir "$OUT"

echo ""
step "Done"
for report in "$OUT"/*.md; do
    [ -e "$report" ] && printf '\033[32m  report:  %s\033[0m\n' "$report"
done
for payload in "$OUT"/*.json; do
    [ -e "$payload" ] && printf '\033[32m  payload: %s\033[0m\n' "$payload"
done
echo ""
printf '\033[90m  The JSON is the record. Anything the Markdown claims should be traceable to it.\033[0m\n'
printf '\033[90m  Falsification checks that could not be evaluated read as "not_evaluated" with a\033[0m\n'
printf '\033[90m  reason -- that is a stated gap, not a pass.\033[0m\n'
echo ""
