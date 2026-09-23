"""Command-line interface.

Every command here is a thin wrapper over :mod:`shingan.pipeline`. The split matters:
the pipeline holds the logic and returns plain data, so the end-to-end flow is testable
without a terminal, and this module is left with argument parsing, presentation and exit
codes. When a command here grows a decision — which rows to use, what a missing metric
means — that decision belongs in the pipeline instead.

Two conventions run through the commands:

* ``--config`` names the base YAML and ``--data-config`` / ``--eval-config`` /
  ``--train-config`` are overlays deep-merged onto it, in that order. Everything is
  validated by the pydantic models in :mod:`shingan.config`, which reject unknown keys,
  so a misspelled option is an error rather than a silent no-op.
* A command that cannot do its job exits non-zero and says why on stderr. It does not
  print a plausible number and exit 0.
"""

from __future__ import annotations

import json
import logging
import math
import platform
import sys
from dataclasses import replace
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from shingan.__about__ import (
    DATA_SCHEMA_VERSION,
    HF_DATASET_ID,
    HF_MODEL_ID,
    LICENSE_ID,
    PROJECT_DISPLAY_NAME,
    PROJECT_TAGLINE,
    __version__,
)
from shingan.config import DEFAULT_CONFIG_FILES, DEMO_CONFIG_FILES, ProjectConfig, load_config
from shingan.logging_utils import get_logger, setup_logging
from shingan.paths import ProjectPaths

logger = get_logger(__name__)

console = Console()
error_console = Console(stderr=True, style="bold red")

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=f"{PROJECT_DISPLAY_NAME} — {PROJECT_TAGLINE}.",
)
data_app = typer.Typer(no_args_is_help=True, help="Build the panel and the SFT dataset.")
train_app = typer.Typer(
    no_args_is_help=True, help="Fit the structured track or fine-tune the text track."
)
eval_app = typer.Typer(no_args_is_help=True, help="Evaluate a run and render the report.")
publish_app = typer.Typer(no_args_is_help=True, help="Publish artifacts to the Hugging Face Hub.")
app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")
app.add_typer(publish_app, name="publish")

# ---------------------------------------------------------------------------
# Shared option types
# ---------------------------------------------------------------------------

ConfigOpt = Annotated[
    Path | None,
    typer.Option("--config", help="Base YAML. Defaults to configs/default.yaml."),
]
DataConfigOpt = Annotated[
    Path | None,
    typer.Option("--data-config", help="Overlay merged onto the base for the data section."),
]
EvalConfigOpt = Annotated[
    Path | None,
    typer.Option("--eval-config", help="Overlay merged onto the base for the eval section."),
]
TrainConfigOpt = Annotated[
    Path | None,
    typer.Option("--train-config", help="Overlay merged onto the base for the lora section."),
]
RootOpt = Annotated[
    Path | None,
    typer.Option(
        "--root", help="Project root. Defaults to the nearest directory containing pyproject.toml."
    ),
]
VerboseOpt = Annotated[
    bool,
    typer.Option("--verbose", "-v", help="Log at debug level, including the per-row decisions."),
]


def _configure_logging(verbose: bool) -> None:
    """Send package logs to a Rich console handler at the requested level.

    ``force`` is set because a single process may serve several commands (in tests, or a
    notebook re-running a cell), and a second call whose level is ignored is a confusing
    way to lose a ``--verbose`` flag.
    """
    setup_logging(level=logging.DEBUG if verbose else logging.INFO, force=True)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"{PROJECT_DISPLAY_NAME} {__version__}  ({LICENSE_ID})")
        raise typer.Exit


@app.callback()
def _root_callback(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Print the version and exit.",
        ),
    ] = False,
) -> None:
    """Shingan command-line entry point."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_stack(
    *,
    config: Path | None,
    data_config: Path | None,
    eval_config: Path | None,
    train_config: Path | None,
    root: Path | None,
) -> tuple[ProjectConfig, ProjectPaths]:
    """Assemble the effective configuration from the base plus overlays.

    The overlays default to the *documented* stack rather than to "nothing", so that
    ``shingan eval run`` with no options means the same thing as the Makefile target.
    Passing ``--data-config`` replaces the default overlay rather than adding to it,
    which is the only behaviour that lets a caller opt out of one.
    """
    paths = ProjectPaths.from_root(root)
    # DEFAULT_CONFIG_FILES entries are repository-relative and already carry the
    # `configs/` prefix, so they join onto the *root*. Joining them onto paths.configs
    # asks for configs/configs/default.yaml, which does not exist.
    base = Path(config) if config is not None else paths.root / DEFAULT_CONFIG_FILES[0]
    defaults = {
        "data": paths.root / DEFAULT_CONFIG_FILES[1],
        "eval": paths.root / DEFAULT_CONFIG_FILES[2],
        "train": None,
    }
    overlays: list[Path | str] = []
    for explicit, key in (
        (data_config, "data"),
        (eval_config, "eval"),
        (train_config, "train"),
    ):
        if explicit is not None:
            overlays.append(Path(explicit))
        elif defaults[key] is not None:
            overlays.append(defaults[key])  # type: ignore[arg-type]
    return load_config(base, overlays, root=root), paths


def _demo_stack(root: Path | None) -> tuple[ProjectConfig, ProjectPaths]:
    """The demo configuration stack, which is deliberately not the default one."""
    paths = ProjectPaths.from_root(root)
    base = paths.root / DEMO_CONFIG_FILES[0]
    overlays: list[Path | str] = [paths.root / name for name in DEMO_CONFIG_FILES[1:]]
    return load_config(base, overlays, root=root), paths


def _redirect(paths: ProjectPaths, out: Path) -> ProjectPaths:
    """A :class:`ProjectPaths` whose outputs land under ``out`` instead of artifacts/.

    Only the output directories move. ``data/`` and the cache stay where they are, so a
    run redirected into a scratch directory still finds its inputs and does not
    re-download anything.
    """
    resolved = out.resolve()
    return replace(
        paths,
        artifacts=resolved,
        models=resolved / "models",
        reports=resolved,
        figures=resolved / "figures",
        processed=resolved,
    )


def _package_versions(names: tuple[str, ...]) -> dict[str, str]:
    """Installed version per distribution, or ``"not installed"``."""
    found: dict[str, str] = {}
    for name in names:
        try:
            found[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            found[name] = "not installed"
    return found


def _label_table(outcome: Any) -> Table:
    """Per-path metrics for one label, as the report presents them."""
    from shingan.pipeline import COMPARISON_ORDER

    table = Table(title=f"{outcome.label}", title_justify="left", show_lines=False)
    table.add_column("path")
    table.add_column("AUC", justify="right")
    table.add_column("KS", justify="right")
    table.add_column("KS dir", justify="right")
    table.add_column("PR-AUC", justify="right")
    table.add_column("base rate", justify="right")
    table.add_column("positives", justify="right")

    if not outcome.fitted:
        table.add_row("[yellow]not evaluated[/yellow]", "", "", "", "", "", "")
        return table
    # `COMPARISON_ORDER`, not `outcome.reports.items()`. Dict order puts the matched
    # control last — below the fused row, three rows away from the paths it is the control
    # for — which is where a reader stops looking for a baseline. This also makes the
    # console order the same as the comparison table in the JSON and Markdown artifacts;
    # two renderings of one run that disagree on row order invite reading them as
    # different tables. Any path outside the tuple still prints, at the end.
    order = [path for path in COMPARISON_ORDER if path in outcome.reports]
    order += [path for path in outcome.reports if path not in COMPARISON_ORDER]
    for path in order:
        report = outcome.reports[path]
        table.add_row(
            path,
            f"{report.auc:.4f}" if report.auc == report.auc else "undefined",
            f"{report.ks:.4f}" if report.ks == report.ks else "undefined",
            # Beside the KS it belongs to, because the whole point of the pair is that a
            # large KS with `negatives_higher` is a broken score, not a strong one.
            report.ks_direction,
            f"{report.pr_auc:.4f}" if report.pr_auc == report.pr_auc else "undefined",
            f"{report.base_rate:.4f}",
            str(report.n_positives),
        )
    return table


def _print_label_summaries(result: Any) -> None:
    """One small table per label, plus the reason any label was skipped."""
    for outcome in result.outcomes.values():
        console.print(_label_table(outcome))
        if not outcome.fitted:
            console.print(f"  [yellow]reason:[/yellow] {outcome.reason}")
        if outcome.ic is not None:
            result_ic = outcome.ic
            # Two t-statistics, and the distinction is the point: `ic.t_stat` treats the
            # per-period ICs as independent, which overlapping forward windows are not.
            # The Newey-West figure is the one to quote, so both are printed.
            raw_t = "n/a" if math.isnan(result_ic.t_stat) else f"{result_ic.t_stat:.2f}"
            nw_t = "n/a" if outcome.ic_tstat is None else f"{outcome.ic_tstat:.2f}"
            console.print(
                f"  IC {result_ic.ic:+.4f} (ICIR {result_ic.icir:+.2f}) over "
                f"{result_ic.periods_used}/{result_ic.n_periods} periods; "
                f"t = {raw_t}, Newey-West t = {nw_t}  <- quote this one"
            )
        interval = outcome.pr_auc_difference
        if interval is not None:
            if interval.n_blocks < 2:
                console.print(
                    f"  PR-AUC difference not estimable: only {interval.n_blocks} time block(s) "
                    "in the test span, so a bootstrap interval would have zero width. "
                    "This is a property of the test window's length, not of the model."
                )
            else:
                verdict = "crosses zero" if interval.crosses_zero else "excludes zero"
                console.print(
                    f"  fused - structured PR-AUC {interval.estimate:+.4f} "
                    f"[{interval.low:.4f}, {interval.high:.4f}] ({verdict})"
                )
        console.print()


def _fail(message: str, *, hint: str | None = None) -> None:
    """Print a failure and exit non-zero."""
    error_console.print(f"error: {message}")
    if hint:
        console.print(Panel(hint, title="how to fix", border_style="yellow"))
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# version / doctor
# ---------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Print version and release identifiers."""
    table = Table(show_header=False, box=None)
    table.add_row("version", __version__)
    table.add_row("license", LICENSE_ID)
    table.add_row("data schema", DATA_SCHEMA_VERSION)
    table.add_row("hf model", HF_MODEL_ID)
    table.add_row("hf dataset", HF_DATASET_ID)
    console.print(table)


@app.command()
def doctor(root: RootOpt = None, verbose: VerboseOpt = False) -> None:
    """Report the environment, the GPU, and the Windows-specific constraints.

    Runs on a machine with no GPU and no training dependencies: every probe is guarded,
    and a missing package is reported as missing rather than raising. The exit code is
    zero even when the training stack is absent, because "the train extra is not
    installed" is a fact about this machine, not a failure of the command.
    """
    _configure_logging(verbose)
    paths = ProjectPaths.from_root(root)

    environment = Table(title="environment", title_justify="left")
    environment.add_column("key")
    environment.add_column("value")
    environment.add_row("python", sys.version.split()[0])
    environment.add_row("executable", sys.executable)
    environment.add_row("platform", platform.platform())
    environment.add_row("machine", platform.machine())
    environment.add_row("root", str(paths.root))
    environment.add_row(
        "configs", "present" if paths.configs.is_dir() else "[yellow]missing[/yellow]"
    )
    console.print(environment)

    core = _package_versions(
        ("numpy", "pandas", "scikit-learn", "scipy", "pydantic", "typer", "pyyaml", "jinja2")
    )
    core_table = Table(title="core dependencies", title_justify="left")
    core_table.add_column("package")
    core_table.add_column("version")
    for name, found in core.items():
        style = "red" if found == "not installed" else "green"
        core_table.add_row(name, f"[{style}]{found}[/{style}]")
    console.print(core_table)

    from shingan.models.lora import (
        INSTALL_HINT,
        check_device_compatibility,
        dependency_report,
        describe_train_environment,
        missing_required_modules,
    )

    report = dependency_report()
    train_table = Table(title="training stack (optional `train` extra)", title_justify="left")
    train_table.add_column("package")
    train_table.add_column("importable")
    for name, available in report.items():
        train_table.add_row(name, "[green]yes[/green]" if available else "[yellow]no[/yellow]")
    console.print(train_table)

    missing = missing_required_modules()
    if missing:
        console.print(
            Panel(
                INSTALL_HINT,
                title=f"train extra incomplete — missing {', '.join(missing)}",
                border_style="yellow",
            )
        )
    else:
        device_table = Table(title="training environment", title_justify="left")
        device_table.add_column("key")
        device_table.add_column("value")
        for key, value in describe_train_environment().items():
            device_table.add_row(key, str(value))
        console.print(device_table)

    # The Windows constraints are configuration, not hardware, so they are checked from
    # the config file rather than from the machine. Checking them here means a mistaken
    # edit to `configs/train/*.yaml` is caught before a multi-hour run rather than after.
    try:
        config, _ = _load_stack(
            config=None, data_config=None, eval_config=None, train_config=None, root=root
        )
    except Exception as exc:  # a config problem must not mask the environment report
        console.print(f"[yellow]configuration could not be loaded: {exc}[/yellow]")
    else:
        invariants = Table(title="windows / blackwell invariants", title_justify="left")
        invariants.add_column("setting")
        invariants.add_column("value")
        invariants.add_column("ok")
        checks = (
            ("lora.dataloader_num_workers", config.lora.dataloader_num_workers, lambda v: v == 0),
            ("lora.packing", config.lora.packing, lambda v: v is False),
            ("lora.attn_implementation", config.lora.attn_implementation, lambda v: v == "sdpa"),
            ("lora.load_in_4bit", config.lora.load_in_4bit, lambda v: v is True),
            ("data.offline", config.data.offline, lambda v: v is True),
        )
        for name, value, ok in checks:
            invariants.add_row(
                name, str(value), "[green]ok[/green]" if ok(value) else "[red]check[/red]"
            )
        console.print(invariants)

    for warning in check_device_compatibility():
        console.print(Panel(warning, border_style="red", title="device warning"))

    if sys.platform == "win32":
        console.print(
            "Windows: keep the checkout path short (MAX_PATH 260), keep text I/O on "
            "encoding='utf-8', and leave dataloader_num_workers at 0. See docs/06-windows-setup.md."
        )


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


@data_app.command("synth")
def data_synth(
    config: ConfigOpt = None,
    data_config: DataConfigOpt = None,
    eval_config: EvalConfigOpt = None,
    root: RootOpt = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="Write the generated tables here as CSV.")
    ] = None,
    verbose: VerboseOpt = False,
) -> None:
    """Generate the deterministic synthetic dataset and report its base rates.

    This is an inspection step, not a pipeline stage: ``data build`` regenerates the same
    tables from the same configuration, so nothing depends on this having been run. It
    exists so the generator's output can be looked at — and its realised base rates
    compared against ``TARGET_BASE_RATES`` — without writing a panel first.
    """
    _configure_logging(verbose)
    project, _paths = _load_stack(
        config=config,
        data_config=data_config,
        eval_config=eval_config,
        train_config=None,
        root=root,
    )

    from shingan.data.synthetic import generate_synthetic_dataset

    synthetic = project.data.synthetic.model_copy(
        update={
            "start": project.data.start,
            "end": project.data.end,
            "n_companies": max(
                2, min(project.data.synthetic.n_companies, len(project.data.universe) or 12)
            ),
        }
    )
    dataset = generate_synthetic_dataset(synthetic)

    summary = Table(title="generated tables", title_justify="left")
    summary.add_column("table")
    summary.add_column("rows", justify="right")
    counts = {
        "prices": dataset.prices,
        "fundamentals": dataset.fundamentals,
        "filings": dataset.filings,
        "news": dataset.news,
        "events": dataset.events,
    }
    for name, frame in counts.items():
        summary.add_row(name, f"{len(frame):,}")
    console.print(summary)
    console.print(
        f"companies {dataset.prices['ticker'].nunique()}  |  "
        f"window {project.data.start} .. {project.data.end}  |  "
        f"base_rate_multiplier {synthetic.base_rate_multiplier}"
    )
    if not dataset.events.empty:
        kinds = Table(title="events by kind", title_justify="left")
        kinds.add_column("kind")
        kinds.add_column("count", justify="right")
        for kind, count in dataset.events["event_kind"].value_counts().items():
            kinds.add_row(str(kind), str(count))
        console.print(kinds)

    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        for name, frame in counts.items():
            destination = out / f"{name}.csv"
            frame.to_csv(destination, index=False, encoding="utf-8")
        console.print(f"wrote {len(counts)} tables under {out}")


@data_app.command("build")
def data_build(
    config: ConfigOpt = None,
    data_config: DataConfigOpt = None,
    eval_config: EvalConfigOpt = None,
    root: RootOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """Assemble the processed panel: as-of joins, features, labels and splits.

    Writes under ``data/processed/``. The leakage scan runs as part of the build and its
    verdict is printed; a build that violates a point-in-time invariant raises rather than
    returning a panel that would silently inflate every downstream metric.
    """
    _configure_logging(verbose)
    project, paths = _load_stack(
        config=config,
        data_config=data_config,
        eval_config=eval_config,
        train_config=None,
        root=root,
    )

    from shingan.data.builder import build_panel
    from shingan.labeling.definitions import build_label_definitions

    build = build_panel(project, write=True)
    panel = build.panel

    overview = Table(title="panel", title_justify="left", show_header=False, box=None)
    overview.add_row("shape", f"{panel.shape[0]:,} rows x {panel.shape[1]} columns")
    overview.add_row("companies", str(panel["ticker"].nunique()))
    overview.add_row("window", f"{panel['as_of'].min()} .. {panel['as_of'].max()}")
    overview.add_row("processed dir", str(paths.processed))
    overview.add_row("artifacts", str(len(build.written)))
    console.print(overview)

    windows = build.split_report.windows
    splits = Table(title="splits (after purging)", title_justify="left")
    splits.add_column("split")
    splits.add_column("nominal")
    splits.add_column("effective")
    splits.add_column("rows", justify="right")
    for name in ("train", "valid", "test"):
        splits.add_row(
            name,
            getattr(windows, f"nominal_{name}").render(),
            getattr(windows, f"effective_{name}").render(),
            str(build.split_report.counts.get(name, 0)),
        )
    console.print(splits)
    console.print(
        f"purge {windows.purge_days}d + embargo {windows.embargo_calendar_days}d "
        f"= margin {windows.margin_days}d"
    )

    definitions = build_label_definitions(project.labels)
    rates = Table(title="label base rates", title_justify="left")
    rates.add_column("label")
    rates.add_column("observable", justify="right")
    rates.add_column("positives", justify="right")
    rates.add_column("rate", justify="right")
    rates.add_column("horizon", justify="right")
    for label in definitions:
        name = str(label)
        mask = panel[f"label_mask_{name}"].astype(bool)
        observable = int(mask.sum())
        positives = int(panel.loc[mask, f"label_{name}"].sum())
        rate = positives / observable if observable else float("nan")
        rates.add_row(
            name,
            str(observable),
            str(positives),
            f"{rate:.4f}" if rate == rate else "undefined",
            str(project.labels.horizon_days(label)),
        )
    console.print(rates)

    scan = build.leakage
    clean = bool(scan.get("feature_matrix_clean"))
    reserved = scan.get("reserved_columns_present") or []
    alarms = scan.get("correlation_alarms") or []
    leakage = Table(title="leakage scan", title_justify="left", show_header=False, box=None)
    leakage.add_row("feature matrix clean", "[green]yes[/green]" if clean else "[red]NO[/red]")
    leakage.add_row("features checked", str(scan.get("n_features", "unknown")))
    leakage.add_row(
        "reserved columns present in the panel",
        f"{len(reserved)} (label_/fwd_/event_ columns — excluded from the feature matrix by design)",
    )
    leakage.add_row("correlation alarms", str(len(alarms)))
    console.print(leakage)
    if alarms:
        console.print(Panel(str(alarms), border_style="yellow", title="correlation alarms"))
    if not clean:
        _fail(
            "the leakage scan found a point-in-time violation",
            hint=(
                "A reserved column reached the feature matrix. Fix the feature selection "
                "before using this panel: every metric computed from it would be inflated."
            ),
        )


@data_app.command("sft")
def data_sft(
    config: ConfigOpt = None,
    data_config: DataConfigOpt = None,
    eval_config: EvalConfigOpt = None,
    root: RootOpt = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="Directory for the JSONL and its manifest.")
    ] = None,
    labels: Annotated[
        str | None,
        typer.Option("--labels", help="Comma-separated label names. Defaults to labels.targets."),
    ] = None,
    no_news: Annotated[
        bool,
        typer.Option("--no-news", help="Leave news out of the prompt (a documented ablation)."),
    ] = False,
    verbose: VerboseOpt = False,
) -> None:
    """Emit the instruction-format JSONL used to train the text track.

    ``train.jsonl`` and ``valid.jsonl`` are written, plus ``manifest.json``. The test
    block is deliberately **not** emitted: it is the evaluation sample, and writing it
    into the training file is the leakage the purge margin exists to prevent. The
    manifest records that decision, along with the target rule, so the choice travels
    with the file.
    """
    _configure_logging(verbose)
    project, paths = _load_stack(
        config=config,
        data_config=data_config,
        eval_config=eval_config,
        train_config=None,
        root=root,
    )

    from shingan.data.builder import build_panel
    from shingan.data.schema import write_jsonl
    from shingan.pipeline import sft_examples

    wanted = [item.strip() for item in labels.split(",") if item.strip()] if labels else None
    build = build_panel(project, write=False)
    try:
        records, manifest = sft_examples(
            build.panel, build, project, labels=wanted, include_news=not no_news
        )
    except ValueError as exc:
        _fail(str(exc))

    destination = Path(out) if out is not None else paths.processed / "sft"
    destination.mkdir(parents=True, exist_ok=True)
    for split, entries in records.items():
        write_jsonl(entries, destination / f"{split}.jsonl")
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    table = Table(title=f"SFT dataset → {destination}", title_justify="left")
    table.add_column("label")
    table.add_column("train", justify="right")
    table.add_column("train +", justify="right")
    table.add_column("valid", justify="right")
    table.add_column("valid +", justify="right")
    for label, counts in manifest["per_label"].items():
        table.add_row(
            label,
            str(counts["train"]),
            str(counts["positives_train"]),
            str(counts["valid"]),
            str(counts["positives_valid"]),
        )
    console.print(table)
    console.print(
        f"target rule: [bold]{manifest['target_rule']}[/bold]\n"
        "A binary target makes `severity` degenerate (low/critical only). Ranking metrics "
        "are unaffected; the text track's probabilities will need a calibrator before they "
        "can be read as probabilities."
    )
    if manifest["prompts_truncated"]:
        console.print(
            f"[yellow]{manifest['prompts_truncated']} prompt(s) exceeded the "
            f"{manifest['character_budget']}-character budget[/yellow] (max_seq_length "
            f"{manifest['max_seq_length']}); recorded per example in meta.truncated."
        )


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


@train_app.command("structured")
def train_structured(
    config: ConfigOpt = None,
    data_config: DataConfigOpt = None,
    eval_config: EvalConfigOpt = None,
    root: RootOpt = None,
    label: Annotated[
        str | None, typer.Option("--label", help="Single label to fit. Defaults to labels.targets.")
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="Where to persist fitted models.")
    ] = None,
    verbose: VerboseOpt = False,
) -> None:
    """Fit the calibrated structured (GBDT) track on CPU and persist it.

    Runs on a laptop: the structured track needs no GPU. The model written to disk is the
    same object the evaluation measured, taken off the outcome rather than re-fitted, so
    the saved artifact and the reported metrics cannot drift apart.
    """
    _configure_logging(verbose)
    project, paths = _load_stack(
        config=config,
        data_config=data_config,
        eval_config=eval_config,
        train_config=None,
        root=root,
    )
    wanted = [label] if label else [str(item) for item in project.labels.targets]

    from shingan.pipeline import PATH_FUSED, PATH_STRUCTURED, PATH_TEXT, run_pipeline

    result = run_pipeline(project, paths, labels=wanted, write=False)
    destination = Path(out) if out is not None else paths.models / "structured"
    destination.mkdir(parents=True, exist_ok=True)

    saved: list[tuple[str, str, Path]] = []
    for name, outcome in result.outcomes.items():
        if not outcome.fitted:
            console.print(f"[yellow]{name}: not fitted — {outcome.reason}[/yellow]")
            continue
        label_dir = destination / name
        label_dir.mkdir(parents=True, exist_ok=True)
        # Each model persists itself, so the file name is the class's own MODEL_FILENAME
        # constant. Writing them here with a raw joblib.dump and a caller-chosen name is
        # how `fused.joblib` came to sit next to a loader that looks for `fusion.joblib`.
        for path in (PATH_STRUCTURED, PATH_TEXT, PATH_FUSED):
            model = outcome.models.get(path)
            if model is None:
                continue
            try:
                saved.append((name, path, Path(model.save(label_dir))))
            except Exception as exc:
                console.print(f"[yellow]{name}: could not persist {path}: {exc}[/yellow]")
        (label_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "label": name,
                    "counts": {
                        "train": outcome.n_train,
                        "valid": outcome.n_valid,
                        "test": outcome.n_test,
                        "positives_train": outcome.positives_train,
                        "positives_valid": outcome.positives_valid,
                        "positives_test": outcome.positives_test,
                    },
                    "calibrator": outcome.calibrator,
                    "metrics": {path: report.as_dict() for path, report in outcome.reports.items()},
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
    console.print(_label_table(result.headline_outcome() or next(iter(result.outcomes.values()))))

    if not saved:
        _fail(
            "no label could be fitted, so nothing was written",
            hint=(
                "Every label in this configuration is too rare for its training fold to hold "
                "two positives. That is the F8 condition, not a bug. Raise "
                "data.synthetic.base_rate_multiplier for a pipeline demo, or widen the "
                "universe for a real sample. See docs/05-evaluation.md section 10."
            ),
        )
    console.print(f"persisted {len(saved)} artifact(s) under {destination}")


@train_app.command("lora")
def train_lora_command(
    config: ConfigOpt = None,
    train_config: TrainConfigOpt = None,
    root: RootOpt = None,
    train_file: Annotated[
        Path | None, typer.Option("--train-file", help="SFT JSONL of train-fold rows.")
    ] = None,
    eval_file: Annotated[
        Path | None, typer.Option("--eval-file", help="SFT JSONL of validation-fold rows.")
    ] = None,
    base: Annotated[str | None, typer.Option("--base", help="Override lora.base_model.")] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Where checkpoints and run.json are written."),
    ] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Override project.seed.")] = None,
    verbose: VerboseOpt = False,
) -> None:
    """QLoRA fine-tune the text track. Needs the `train` extra and a CUDA device.

    On a machine without the extra this exits non-zero with the install instructions
    rather than degrading to a CPU run: a 14B 4-bit fine-tune on CPU is not a slower
    version of this command, it is a different and useless one.
    """
    _configure_logging(verbose)
    project, paths = _load_stack(
        config=config, data_config=None, eval_config=None, train_config=train_config, root=root
    )
    if base is not None:
        project = project.model_copy(
            update={"lora": project.lora.model_copy(update={"base_model": base})}
        )

    from shingan.models.lora import (
        INSTALL_HINT,
        LoraLeakageError,
        MissingTrainDependencies,
        TrainingStackMismatch,
        train_lora,
    )

    resolved_train = Path(train_file) if train_file else paths.processed / "sft" / "train.jsonl"
    resolved_eval = Path(eval_file) if eval_file else paths.processed / "sft" / "valid.jsonl"
    if not resolved_train.is_file():
        _fail(
            f"training file not found: {resolved_train}",
            hint="Build it first:  shingan data sft --out data/processed/sft",
        )
    if not resolved_eval.is_file():
        console.print(
            f"[yellow]no validation file at {resolved_eval}; training without per-epoch "
            "evaluation, so epoch selection is not possible[/yellow]"
        )
        resolved_eval = Path()

    try:
        result = train_lora(
            project.lora,
            resolved_train,
            eval_file=resolved_eval if resolved_eval.is_file() else None,
            output_dir=output_dir,
            paths=paths,
            seed=seed if seed is not None else project.project.seed,
        )
    except MissingTrainDependencies:
        _fail("the training stack is not installed", hint=INSTALL_HINT)
    except LoraLeakageError as exc:
        _fail(str(exc))
    except TrainingStackMismatch as exc:
        # The message already names the offending keys, the installed versions and the
        # reinstall command; a hint on top of it would be the same sentence twice.
        _fail(str(exc))

    table = Table(title="LoRA run", title_justify="left")
    table.add_column("key")
    table.add_column("value")
    table.add_row("output dir", str(result.output_dir))
    table.add_row("base model", result.base_model)
    table.add_row("train examples", str(result.n_train_examples))
    table.add_row("eval examples", str(result.n_eval_examples))
    for key, value in result.eval_metrics.items():
        table.add_row(f"eval {key}", str(value))
    console.print(table)
    for warning in result.warnings:
        console.print(Panel(warning, border_style="yellow", title="warning"))
    console.print(
        f"[yellow]The reported epoch-{project.lora.num_train_epochs} metrics are selection "
        "metrics[/yellow]: that epoch was evaluated on the same fold it was selected on. "
        "Quote the held-out test block from `shingan eval run` instead."
    )


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


def _run_and_write(
    project: ProjectConfig,
    paths: ProjectPaths,
    *,
    labels: list[str] | None,
    out: Path | None,
    verbose: bool,
) -> tuple[Any, dict[str, Path]]:
    """Run the pipeline, write the outputs, and print the per-label summaries."""
    from shingan.pipeline import run_pipeline, write_outputs

    result = run_pipeline(project, paths, labels=labels, write=False)
    target = _redirect(paths, out) if out is not None else paths
    target.ensure()
    written = write_outputs(result, target)
    _print_label_summaries(result)
    return result, written


@eval_app.command("run")
def eval_run(
    config: ConfigOpt = None,
    data_config: DataConfigOpt = None,
    eval_config: EvalConfigOpt = None,
    root: RootOpt = None,
    run_dir: Annotated[
        Path | None, typer.Option("--run-dir", help="Where the report and panel are written.")
    ] = None,
    labels: Annotated[
        str | None, typer.Option("--labels", help="Comma-separated label names.")
    ] = None,
    verbose: VerboseOpt = False,
) -> None:
    """Fit all three tracks, evaluate on the test block, and write the report.

    This is the command that produces the numbers anyone is expected to read. Writes the
    panel, a Markdown report and a JSON payload, and refuses to write at all if the run
    metadata is incomplete — a report missing its calibration fold reads later as though
    it had one.
    """
    _configure_logging(verbose)
    project, paths = _load_stack(
        config=config,
        data_config=data_config,
        eval_config=eval_config,
        train_config=None,
        root=root,
    )
    wanted = [item.strip() for item in labels.split(",") if item.strip()] if labels else None

    from shingan.eval.report import ReportValidationError

    try:
        _result, written = _run_and_write(
            project, paths, labels=wanted, out=run_dir, verbose=verbose
        )
    except ReportValidationError as exc:
        _fail(
            f"the report is incomplete and was not written: {exc}",
            hint="This is a bug in the report assembly, not in the data. Please file it.",
        )

    for name, path in written.items():
        console.print(f"wrote {name}: {path}")


@eval_app.command("lora")
def eval_lora(
    adapter: Annotated[
        Path, typer.Option("--adapter", help="Adapter directory written by `train lora`.")
    ] = Path("artifacts/lora/adapter"),
    config: ConfigOpt = None,
    data_config: DataConfigOpt = None,
    eval_config: EvalConfigOpt = None,
    root: RootOpt = None,
    label: Annotated[str, typer.Option("--label", help="Label to score.")] = "tail_risk",
    split: Annotated[str, typer.Option("--split", help="test, valid or train.")] = "test",
    limit: Annotated[
        int | None, typer.Option("--limit", help="Score only the first N rows (a smoke run).")
    ] = None,
    out: Annotated[Path | None, typer.Option("--out", help="Directory for the artifact.")] = None,
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens")] = 512,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 4,
    temperature: Annotated[
        float, typer.Option("--temperature", help="0 for greedy; anything else samples.")
    ] = 0.0,
    verify_prompts: Annotated[
        bool,
        typer.Option(
            "--verify-prompts/--no-verify-prompts",
            help="Refuse to score unless the rebuilt prompts match the SFT file byte for byte.",
        ),
    ] = True,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Build and check the prompts, loading no model.")
    ] = False,
    verbose: VerboseOpt = False,
) -> None:
    """Score the fine-tuned text track on one label and split, and write the artifact.

    This is the command that answers the project's central question. Until it existed,
    ``eval run`` fitted structured / TF-IDF / fusion and stopped, so no artifact in the
    repository had ever scored the model the project exists to train.

    Three properties are enforced rather than documented:

    * **The prompts are the ones the adapter trained on.** They are rebuilt through the
      same functions the data builder uses, then compared byte for byte against the SFT
      file, which is a check that costs seconds and cannot run after the fact. The
      alternative failure — scoring a model on inputs it never saw — is invisible in
      every metric: it looks like a weak model.
    * **Unparsed generations are disclosed, not filled in.** A row whose output has no
      score is dropped, and the drop count and the number of dropped **positives** are
      written next to the metrics. Filling a zero would assert the model called it
      negative.
    * **The row says what it is.** The prompt includes a twelve-column structured-signal
      block, so ``text_only_lora`` is prompt-conditioned, not text-only, and is compared
      against a matched baseline rather than against the full feature set.
    """
    _configure_logging(verbose)
    project, paths = _load_stack(
        config=config,
        data_config=data_config,
        eval_config=eval_config,
        train_config=None,
        root=root,
    )

    from datetime import UTC, datetime

    from shingan.data.builder import PROMPT_SIGNAL_COLUMNS, build_panel
    from shingan.eval.lora import (
        PATH_LORA,
        PromptIntegrity,
        build_payload,
        compare_prompts,
        paired_differences,
        score_from_attempts,
        write_artifact,
    )
    from shingan.models.lora import iter_jsonl
    from shingan.pipeline import (
        PATH_FUSED,
        PATH_MATCHED,
        PATH_STRUCTURED,
        PATH_TEXT,
        build_prompt_contexts,
        chars_budget_for_seqlength,
        evaluate_label,
        label_split_frames,
        select_feature_columns,
        text_inputs,
    )
    from shingan.prompts import build_chat_messages, build_user_prompt

    if split not in {"train", "valid", "test"}:
        _fail(f"unknown split {split!r}", hint="choose one of: train, valid, test")

    build = build_panel(project, write=False)
    panel = build.panel
    frames = label_split_frames(panel, label)
    target = frames[split]
    if target.empty:
        _fail(
            f"the {split} block holds no observable rows for {label}",
            hint="Check the label and the split definition in configs/default.yaml.",
        )

    budget = chars_budget_for_seqlength(project)
    contexts = build_prompt_contexts(panel, build, project, label, budget)

    # Sample ids are formed the same way the builder forms them, so the rebuilt prompts
    # can be matched against the file training actually read. The key is the (id, label)
    # pair: the SFT file holds one example per row *and* label, and the id alone repeats
    # across them.
    sample_ids = panel["ticker"].astype(str) + "-" + panel["as_of"].dt.strftime("%Y%m%d")
    prompts_by_row = {
        (str(sample_ids.iloc[position]), label): build_user_prompt(context)
        for position, context in contexts.items()
    }

    integrity: PromptIntegrity | None = None
    if verify_prompts:
        records: list[dict[str, Any]] = []
        for name in ("train", "valid"):
            candidate = paths.processed / "sft" / f"{name}.jsonl"
            if candidate.is_file():
                records.extend(iter_jsonl(candidate))
        if not records:
            _fail(
                "there is no SFT file to check the prompts against",
                hint="Run `shingan data sft` first, or pass --no-verify-prompts to score "
                "without the check (the resulting numbers would not be reproducible).",
            )
        integrity = compare_prompts(records, prompts_by_row)
        if not integrity.ok:
            _fail(
                "the rebuilt prompts do not reproduce the SFT file: "
                f"{integrity.n_matching}/{integrity.n_compared} matched, "
                f"{integrity.n_rebuilt_missing} rows had no rebuilt prompt",
                hint=(
                    "Scoring would measure the model on inputs it was not trained on. "
                    "Rebuild the SFT file (`shingan data sft`) with the same --data-config, "
                    f"or inspect the mismatch: {json.dumps(integrity.examples[:2], default=str)}"
                ),
            )

    positions = list(target.index)
    truth = [int(value) for value in target[f"label_{label}"].to_numpy()]
    if limit is not None:
        positions, truth = positions[:limit], truth[:limit]
    conversations = [build_chat_messages(contexts[position]) for position in positions]

    console.print(
        f"{split} block: {len(target)} observable rows for {label}, "
        f"{int(sum(truth))} positive(s) in the scored subset ({len(positions)} rows)"
    )
    if integrity is not None:
        console.print(
            f"prompt check: {integrity.n_matching}/{integrity.n_compared} rows for {label} "
            f"rebuilt byte-identically to the SFT file "
            f"({integrity.n_scanned} records scanned, {integrity.n_other_label} for other labels)"
        )

    if dry_run:
        lengths = sorted(len(prompt) for prompt in prompts_by_row.values())
        console.print(
            "dry run: prompts built and verified, no model loaded. "
            f"length min/median/max = {lengths[0]}/{lengths[len(lengths) // 2]}/{lengths[-1]} chars; "
            f"truncated {sum(int(c.truncated) for c in contexts.values())} of {len(contexts)}"
        )
        return

    from shingan.models.lora_inference import (
        MissingInferenceDependencies,
        generate_texts,
        load_for_inference,
        score_generation,
    )

    try:
        model, tokenizer, facts = load_for_inference(
            adapter,
            device_map="auto" if project.lora.device_map == "auto" else "none",
            attn_implementation=project.lora.attn_implementation,
            load_in_4bit=project.lora.load_in_4bit,
            compute_dtype=project.lora.bnb_4bit_compute_dtype,
        )
    except MissingInferenceDependencies as exc:
        _fail(str(exc))

    outputs = generate_texts(
        model,
        tokenizer,
        conversations,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        temperature=temperature,
        progress=verbose,
    )
    horizon = int(project.labels.horizon_days(label))
    attempts = [
        score_generation(text, expected_label=label, expected_horizon_days=horizon)
        for text in outputs
    ]
    scored = score_from_attempts(
        truth,
        [attempt.score for attempt in attempts],
        [attempt.reason for attempt in attempts],
        label=label,
        path=PATH_LORA,
        split=split,
    )

    # Per-row predictions, so the aggregate metrics in the artifact can be recomputed
    # rather than believed. The raw generation is capped: 492 rows of verbose output
    # would otherwise dominate the file, and the cap is recorded per row.
    raw_cap = 4000
    predictions = [
        {
            "ticker": str(panel["ticker"].iloc[position]),
            "as_of": str(panel["as_of"].iloc[position]),
            "label": label,
            "y_true": int(outcome_value),
            "score": attempt.score,
            "parsed": attempt.parsed,
            "reason": attempt.reason,
            "raw": text[:raw_cap],
            "raw_truncated": len(text) > raw_cap,
        }
        for position, outcome_value, attempt, text in zip(
            positions, truth, attempts, outputs, strict=True
        )
    ]

    rows: list[dict[str, Any]] = []
    differences: list[dict[str, Any]] = []
    caveats: list[str] = []
    if limit is None:
        import pandas as pd

        features = select_feature_columns(panel, project)
        outcome = evaluate_label(
            panel,
            project,
            label,
            features,
            text_inputs(panel, build, project, label),
            n_boot=project.eval.rolling.bootstrap_samples,
        )
        if outcome.fitted:
            # The matched control is read off the outcome rather than fitted again here.
            # `evaluate_label` now produces it, so `eval run` publishes the same row; a
            # second fit would put two values under one name in two artifacts that a
            # reader is expected to compare.
            matched_report = outcome.reports.get(PATH_MATCHED)
            if matched_report is None:
                caveats.append(
                    "the matched-information baseline row is absent from this artifact: "
                    f"{outcome.matched_reason or 'not fitted'}"
                )
            # Reading order, and the matched baseline sits second on purpose: it is the
            # row the LoRA row is subtracted from, so it has to be adjacent to it.
            for report in (
                outcome.reports[PATH_STRUCTURED],
                matched_report,
                outcome.reports[PATH_TEXT],
                outcome.reports[PATH_FUSED],
            ):
                if report is not None:
                    rows.append(report.as_dict())

            # Every arm on the same rows. The LoRA column is NaN wherever a generation
            # failed to parse, and `paired_differences` drops those rows for all arms
            # rather than comparing one arm on a subset of the other's rows.
            arms: dict[str, Any] = {
                "as_of": target["as_of"],
                "y_true": target[f"label_{label}"].astype(int),
                PATH_STRUCTURED: outcome.scores[PATH_STRUCTURED],
                PATH_TEXT: outcome.scores[PATH_TEXT],
                PATH_FUSED: outcome.scores[PATH_FUSED],
                PATH_LORA: pd.Series(
                    [attempt.score for attempt in attempts], index=positions, dtype="float64"
                ),
            }
            # The text and structured baselines are always there. The matched control is
            # listed first when it exists, because that is the subtraction that means
            # "the text contribution" and the reading order should not bury it — and it is
            # omitted rather than imputed when it does not, since a difference against an
            # all-NaN arm would be reported as "not measured" beside two real ones.
            baselines: list[str] = [PATH_TEXT, PATH_STRUCTURED]
            if PATH_MATCHED in outcome.scores:
                arms[PATH_MATCHED] = outcome.scores[PATH_MATCHED]
                baselines.insert(0, PATH_MATCHED)
            differences = paired_differences(
                pd.DataFrame(arms),
                candidate=PATH_LORA,
                baselines=tuple(baselines),
                n_boot=project.eval.rolling.bootstrap_samples,
                block_days=project.labels.calendar_horizon_days(label),
                alpha=1.0 - project.eval.rolling.confidence_level,
                seed=project.project.seed,
            )
        else:
            caveats.append(f"the baseline rows were not fitted: {outcome.reason}")
    else:
        caveats.append(
            "--limit was used, so the baseline rows are omitted: they are measured on the "
            "whole test block and mixing them with a prefix of it would compare different "
            "row sets."
        )
    rows.append(scored.as_dict())

    data_info = {
        "sources": [str(source) for source in project.data.sources],
        "is_synthetic": bool(panel["is_synthetic"].any())
        if "is_synthetic" in panel.columns
        else None,
        "data_config": str(data_config) if data_config is not None else "base default overlay",
        "n_panel_rows": len(panel),
        "n_companies": panel["ticker"].nunique(),
        "split_counts": {name: len(frame) for name, frame in frames.items()},
    }
    payload = build_payload(
        label=label,
        split=split,
        rows=rows,
        scored=scored,
        adapter=facts.as_dict(),
        generation={
            "policy": "greedy" if temperature <= 0 else "sampled",
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
        },
        prompt={
            "integrity": integrity.as_dict() if integrity is not None else {"checked": False},
            "includes_structured_signals": True,
            "structured_signals": list(PROMPT_SIGNAL_COLUMNS),
            "chars_budget": budget,
            "n_truncated": int(sum(int(context.truncated) for context in contexts.values())),
        },
        data=data_info,
        split_definition=build.split_report.to_dict(),
        predictions=predictions,
        differences=differences,
        caveats=caveats,
    )

    destination = (
        out
        if out is not None
        else paths.artifacts / "lora-eval" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    written = write_artifact(payload, destination)

    table = Table(title=f"{label} / {split}", title_justify="left")
    for column in ("path", "AUC", "KS", "PR-AUC", "positives", "n rows"):
        table.add_column(column, justify="right" if column != "path" else "left")
    for row in rows:
        table.add_row(
            str(row.get("path")),
            _metric(row.get("auc")),
            _metric(row.get("ks")),
            _metric(row.get("pr_auc")),
            str(row.get("n_positives", "?")),
            str(row.get("n_rows", "?")),
        )
    console.print(table)
    for item in differences:
        label_pair = f"{item['a']} - {item['b']} ({item['metric']})"
        if item.get("estimate") is None:
            console.print(f"  {label_pair}: not measured — {item.get('note', '')}")
        else:
            console.print(
                f"  {label_pair}: {item['estimate']:+.4f} "
                f"[{item['ci_low']:+.4f}, {item['ci_high']:+.4f}]"
                f", crosses zero: {item['crosses_zero']}"
                + (f"  ({item['note']})" if item.get("note") else "")
            )
    console.print(
        f"  parse: {scored.n_parsed}/{scored.n_attempted} usable "
        f"(failure rate {scored.failure_rate:.4f}), dropped positives {scored.n_dropped_positives}"
    )
    if scored.n_dropped:
        console.print(f"  [yellow]reasons:[/yellow] {scored.failure_reasons}")
    for name, path in written.items():
        console.print(f"wrote {name}: {path}")


def _metric(value: Any) -> str:
    """A metric for a terminal table: four decimals, or an explicit absence."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "not measured"
    return "not measured" if not math.isfinite(number) else f"{number:.4f}"


@eval_app.command("report")
def eval_report(
    run_dir: Annotated[Path, typer.Option("--run-dir", help="A directory written by `eval run`.")],
    root: RootOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """Re-render the Markdown report from a run directory, and check it against the JSON.

    Re-rendering means re-running the pipeline from the configuration snapshot stored in
    the run's JSON, because the report is assembled from live frames rather than from a
    deserialisable object graph. The stored ``run_id`` is reused so the new Markdown
    replaces the old one instead of accumulating.

    That design has a consequence worth stating: if the code has changed since the run,
    the re-rendered numbers may differ from the stored ones. Rather than hide that, the
    headline AUCs are compared and a mismatch is reported loudly — a silent divergence
    between a report and its own JSON would be worse than either.
    """
    _configure_logging(verbose)
    paths = ProjectPaths.from_root(root)
    payloads = sorted(run_dir.glob("*.json"))
    if not payloads:
        _fail(
            f"no JSON payload in {run_dir}",
            hint="Point --run-dir at a directory written by `shingan eval run`.",
        )
    payload = json.loads(payloads[-1].read_text(encoding="utf-8"))
    metadata_payload = payload.get("metadata", {})
    # `RunMetadata.as_dict` names this `config_snapshot`, not `config`.
    raw_config = metadata_payload.get("config_snapshot")
    if not raw_config:
        _fail(
            f"{payloads[-1]} carries no configuration snapshot",
            hint="The run predates config capture, or the JSON is not a Shingan report.",
        )

    from shingan.pipeline import assemble_report, run_pipeline

    project = ProjectConfig.model_validate(raw_config)
    result = run_pipeline(project, paths, write=False)
    report = assemble_report(result, run_id=metadata_payload.get("run_id"))
    report.metadata.validate()

    destination = run_dir / f"{report.metadata.run_id}.md"
    destination.write_text(report.to_markdown(), encoding="utf-8")
    console.print(f"wrote {destination}")

    stored = {
        (row.get("label"), row.get("path")): row.get("auc")
        for row in (payload.get("comparison") or [])
        if isinstance(row, dict)
    }
    fresh = (
        {
            (row["label"], row["path"]): row.get("auc")
            for row in report.comparison.to_dict(orient="records")
        }
        if not report.comparison.empty
        else {}
    )
    diverged = [
        f"{label_name}/{path_name}: stored {stored_value!r} vs re-rendered {fresh[key]!r}"
        for key, stored_value in stored.items()
        for label_name, path_name in (key,)
        if key in fresh and stored_value != fresh[key]
    ]
    if not stored:
        console.print(
            "[yellow]the stored JSON carries no comparison table, so the re-rendered "
            "numbers could not be checked against it[/yellow]"
        )
    elif diverged:
        console.print(
            Panel(
                "\n".join(diverged[:10]),
                title="re-rendered metrics differ from the stored ones",
                border_style="red",
            )
        )
    else:
        console.print("[green]re-rendered metrics match the stored JSON[/green]")


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------


@app.command()
def demo(
    out: Annotated[
        Path, typer.Option("--out", help="Output directory for the report and panel.")
    ] = Path("artifacts/demo"),
    root: RootOpt = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Override project.seed.")] = None,
    labels: Annotated[
        str | None, typer.Option("--labels", help="Comma-separated label names.")
    ] = None,
    verbose: VerboseOpt = False,
) -> None:
    """End-to-end CPU run on synthetic data: the command that proves the wiring works.

    Uses the **demo configuration**, not the default one, and the difference matters.
    The demo raises ``synthetic.base_rate_multiplier`` so that every label has enough
    positives in every fold to be fitted; the default stack does not, and two of the
    three labels come back ``fitted=False`` with an F8 reference. Neither behaviour is a
    bug — but a demo output is a demonstration of the pipeline, not an estimate of
    anything, and every artifact it writes is stamped ``is_synthetic: true``.
    """
    _configure_logging(verbose)
    project, paths = _demo_stack(root)
    if seed is not None:
        project = project.model_copy(
            update={"project": project.project.model_copy(update={"seed": seed})}
        )
    wanted = [item.strip() for item in labels.split(",") if item.strip()] if labels else None

    console.print(
        Panel(
            f"{PROJECT_DISPLAY_NAME} {__version__}\n"
            f"config: {', '.join(DEMO_CONFIG_FILES)}\n"
            f"base_rate_multiplier={project.data.synthetic.base_rate_multiplier}  "
            f"offline={project.data.offline}  seed={project.project.seed}\n\n"
            "This run demonstrates that the pipeline is wired correctly. Its base rates are "
            "not real, so its metrics describe the implementation, not any market.",
            border_style="cyan",
            title="demo",
        )
    )

    from shingan.eval.report import ReportValidationError

    try:
        result, written = _run_and_write(
            project, paths, labels=wanted, out=Path(out), verbose=verbose
        )
    except ReportValidationError as exc:
        _fail(
            f"the report is incomplete and was not written: {exc}",
            hint="This is a bug in the report assembly, not in the data. Please file it.",
        )

    console.print(
        f"drift reports: {len(result.drift)}   stability reports: {len(result.stability)}"
    )
    backtest = result.backtest
    if backtest.get("applicable"):
        console.print("quantile backtest: applicable")
    else:
        console.print(
            f"quantile backtest: [yellow]not applicable[/yellow] — {backtest.get('reason', '')}"
        )
    for note in result.notes:
        console.print(f"[yellow]note:[/yellow] {note}")

    for name, path in written.items():
        console.print(f"wrote {name}: {path}")


# ---------------------------------------------------------------------------
# publish
# ---------------------------------------------------------------------------


@publish_app.command("hf")
def publish_hf(
    run_dir: Annotated[
        Path | None,
        typer.Option("--run-dir", help="A run directory containing a report JSON."),
    ] = None,
    values_file: Annotated[
        Path | None,
        typer.Option(
            "--values-file",
            help=(
                "JSON object with card values (e.g. artifacts/stage2/card_values.json). "
                "Merged after the report-derived values, so the file wins."
            ),
        ),
    ] = None,
    only: Annotated[
        str,
        typer.Option(
            "--only",
            help=(
                "Which card to gate and upload: both, dataset, or model. The dataset card "
                "can be published before the fine-tuned model exists; the model card cannot."
            ),
        ),
    ] = "both",
    repo_model: Annotated[
        str, typer.Option("--repo-model", help="Target model repository id.")
    ] = HF_MODEL_ID,
    repo_dataset: Annotated[
        str, typer.Option("--repo-dataset", help="Target dataset repository id.")
    ] = HF_DATASET_ID,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Render the cards and print the plan; upload nothing.")
    ] = False,
    verbose: VerboseOpt = False,
) -> None:
    """Render the model and dataset cards, and upload them.

    Deliberately conservative: the cards are rendered from the run's own numbers, and any
    placeholder the template still contains is listed. An upload whose card has an
    unfilled ``{{...}}`` slot publishes a claim nobody checked, so ``--dry-run`` is the
    safe default workflow and the real upload refuses to proceed while placeholders remain.

    ``--values-file`` injects human-audited card values (dataset statistics, audit
    results, status declarations); it overrides the report-derived values. A field that
    genuinely cannot be measured must be written as ``not measured`` with the reason —
    never as an estimated number.

    ``--only`` exists because the two cards become complete at different times: the
    labelled panel is real today, while the model card needs a fine-tuned adapter that
    does not exist yet. The refusal check and the upload then apply only to the selection.

    Requires ``huggingface_hub`` and a token in ``HF_TOKEN``. Until the fine-tuned adapter
    exists — which it does not, see docs/04-training.md section 5 — the model-card upload
    path stays untested and this command says so rather than pretending otherwise.
    """
    _configure_logging(verbose)
    paths = ProjectPaths.from_root(None)

    from shingan.eval.report import render_dataset_card, render_model_card

    values: dict[str, Any] = {
        "version": __version__,
        "data_version": DATA_SCHEMA_VERSION,
        "code_version": __version__,
    }
    report_json: Path | None = None
    if run_dir is not None:
        candidates = sorted(Path(run_dir).glob("*.json"))
        if not candidates:
            _fail(f"no report JSON in {run_dir}")
        # Auxiliary JSONs (label review, coverage audit) live next to the run report;
        # the report is the one carrying both a metadata block and a model comparison.
        # Name order alone picked the wrong file in artifacts/stage2 (label_review.json
        # sorts after the timestamp-named report).
        payload: dict[str, Any] | None = None
        for candidate in candidates:
            try:
                parsed = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(parsed, dict) and "metadata" in parsed and "comparison" in parsed:
                report_json, payload = candidate, parsed
        if report_json is None or payload is None:
            report_json = candidates[-1]
            try:
                payload = json.loads(report_json.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                _fail(f"cannot read {report_json}", hint=str(exc))
        console.print(f"[cyan]report: {report_json}[/cyan]")
        values["run_id"] = payload.get("metadata", {}).get("run_id", "unknown")
        comparison = payload.get("comparison") or []
        for row in comparison:
            if isinstance(row, dict) and row.get("path") == "fused":
                values[f"auc_{row.get('label')}"] = row.get("auc")

    if values_file is not None:
        try:
            extra = json.loads(Path(values_file).read_text(encoding="utf-8"))
        except OSError as exc:
            _fail(f"cannot read values file {values_file}", hint=str(exc))
        except json.JSONDecodeError as exc:
            _fail(f"values file {values_file} is not valid JSON", hint=str(exc))
        if not isinstance(extra, dict):
            _fail(f"values file {values_file} must contain a JSON object of card values")
        values.update(extra)
        console.print(f"[cyan]values: {len(extra)} entries from {values_file}[/cyan]")

    if only not in ("both", "dataset", "model"):
        _fail(
            f"unknown --only value {only!r}",
            hint="Use --only both, --only dataset, or --only model.",
        )
    selected = {"both": ("model", "dataset"), "dataset": ("dataset",), "model": ("model",)}[only]

    model_card, model_missing = render_model_card(
        values, template_path=paths.root / "templates" / "model_card.md"
    )
    dataset_card, dataset_missing = render_dataset_card(
        values, template_path=paths.root / "templates" / "dataset_card.md"
    )

    plan = Table(title="publish plan", title_justify="left")
    plan.add_column("artifact")
    plan.add_column("destination")
    plan.add_column("placeholders left")
    plan.add_row("model card", repo_model, str(len(model_missing)))
    plan.add_row("dataset card", repo_dataset, str(len(dataset_missing)))
    console.print(plan)

    for name, missing in (("model card", model_missing), ("dataset card", dataset_missing)):
        if missing:
            console.print(
                f"[yellow]{name}: {len(missing)} unfilled placeholder(s): "
                f"{', '.join(sorted(set(missing))[:8])}[/yellow]"
            )

    if dry_run:
        console.print("[cyan]--dry-run: nothing was uploaded.[/cyan]")
        return

    missing_by_card = {"model": model_missing, "dataset": dataset_missing}
    blocking = [name for name in selected if missing_by_card[name]]
    if blocking:
        _fail(
            "refusing to upload a card with unfilled placeholders: "
            + ", ".join(f"{name} card ({len(missing_by_card[name])})" for name in blocking),
            hint=(
                "Fill the remaining values (see templates/, or pass --values-file) or use "
                "--dry-run to inspect the plan. Publishing a card with an unfilled slot "
                "publishes a claim that was never checked. Use --only dataset or --only "
                "model to gate one card while the other is not ready yet."
            ),
        )

    try:
        from huggingface_hub import HfApi
    except ImportError:
        _fail(
            "huggingface_hub is not installed",
            hint="pip install huggingface_hub, then set HF_TOKEN in the environment.",
        )

    import os

    if not os.environ.get("HF_TOKEN"):
        _fail(
            "HF_TOKEN is not set",
            hint="Export a write token before uploading. Nothing is read from a config file.",
        )

    api = HfApi()
    uploaded: list[str] = []
    if "model" in selected:
        api.upload_file(
            path_or_fileobj=model_card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_model,
            repo_type="model",
            commit_message=f"Shingan {__version__} model card",
        )
        uploaded.append(f"model card to {repo_model}")
    if "dataset" in selected:
        api.upload_file(
            path_or_fileobj=dataset_card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_dataset,
            repo_type="dataset",
            commit_message=f"Shingan {__version__} dataset card",
        )
        uploaded.append(f"dataset card to {repo_dataset}")
    console.print(f"uploaded {', '.join(uploaded)}")


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover - exercised through the console script
    main()
