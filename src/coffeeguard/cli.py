"""``coffeeguard`` command-line interface.

Every pipeline step is one command. Commands are grouped by phase; heavy imports
(torch, timm, ...) happen inside commands so ``coffeeguard --help`` stays fast.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from coffeeguard.config import DataConfig, TrainConfig, load_config
from coffeeguard.utils.log import setup_logging

# Windows pipes/redirects default to the ANSI code page (cp1252), which cannot encode the
# "×", "→" etc. used in help and log text; rendering --help would crash with
# UnicodeEncodeError. Switch to UTF-8 before Typer prints anything.
for _stream in (sys.stdout, sys.stderr):
    if (getattr(_stream, "encoding", "") or "").lower().replace("-", "") != "utf8":
        with contextlib.suppress(AttributeError, ValueError):
            _stream.reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(
    name="coffeeguard",
    help="CoffeeGuard AI — coffee leaf disease pipeline.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
data_app = typer.Typer(help="Data engineering: download, prepare, report.", no_args_is_help=True)
app.add_typer(data_app, name="data")
embed_app = typer.Typer(
    help="Foundation-model embeddings: label audit, probe.", no_args_is_help=True
)
app.add_typer(embed_app, name="embed")

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="YAML config file.")]
SetOpt = Annotated[
    list[str] | None,
    typer.Option("--set", "-s", help="Override a config value: key.sub=value (repeatable)."),
]


@app.callback()
def _main(
    log_level: Annotated[str, typer.Option(help="Logging level.")] = "INFO",
) -> None:
    import os

    from coffeeguard.utils.paths import project_root

    # Keep downloaded pretrained weights inside the project (git-ignored .cache/) unless
    # the user already chose a location; avoids filling the system drive.
    cache = project_root() / ".cache"
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    setup_logging(log_level.upper())


def _data_cfg(config: Path, overrides: list[str] | None) -> DataConfig:
    return load_config(DataConfig, config, overrides)


@app.command()
def info(
    config: ConfigOpt = Path("configs/data.yaml"),
    overrides: SetOpt = None,
) -> None:
    """Show the resolved data config and the runtime environment."""
    from coffeeguard.utils.runs import capture_env

    cfg = _data_cfg(config, overrides)
    typer.echo(json.dumps({"data": cfg.model_dump(mode="json"), "env": capture_env()}, indent=2))


@data_app.command("download")
def data_download(
    config: ConfigOpt = Path("configs/data.yaml"),
    overrides: SetOpt = None,
    force: Annotated[bool, typer.Option(help="Re-download and re-extract.")] = False,
) -> None:
    """Download the Kaggle dataset into data/raw and extract it."""
    from coffeeguard.data.download import download_dataset

    download_dataset(_data_cfg(config, overrides), force=force)


@data_app.command("prepare")
def data_prepare(
    config: ConfigOpt = Path("configs/data.yaml"),
    overrides: SetOpt = None,
    stages: Annotated[
        str, typer.Option(help="Comma-separated stages: scan,embed,group,split.")
    ] = "scan,embed,group,split",
    workers: Annotated[int | None, typer.Option(help="Parallel scan workers.")] = None,
    force: Annotated[bool, typer.Option(help="Rescan even if scan.parquet exists.")] = False,
) -> None:
    """Scan, embed, deduplicate/group and split the dataset (idempotent)."""
    from coffeeguard.data.prepare import prepare

    stage_list = [s.strip() for s in stages.split(",") if s.strip()]
    summary = prepare(_data_cfg(config, overrides), stage_list, workers=workers, force=force)
    typer.echo(
        json.dumps({k: v for k, v in summary.items() if k != "balance"}, indent=2, default=str)
    )


@data_app.command("report")
def data_report(
    config: ConfigOpt = Path("configs/data.yaml"),
    overrides: SetOpt = None,
) -> None:
    """Write EDA / data-quality figures and artifacts/reports/eda_summary.json."""
    from coffeeguard.data.report import make_report

    make_report(_data_cfg(config, overrides))


@data_app.command("aug-preview")
def data_aug_preview(
    config: Annotated[Path, typer.Option("--config", "-c", help="Training recipe YAML.")] = Path(
        "configs/train/effnetv2_b0.yaml"
    ),
    overrides: SetOpt = None,
    out: Annotated[Path, typer.Option(help="Output PNG.")] = Path(
        "artifacts/figures/augmentation_preview.png"
    ),
) -> None:
    """Save a grid of training augmentations (check that lesions survive)."""
    from coffeeguard.data.dataset import read_split
    from coffeeguard.data.report import fig_augmentation_preview
    from coffeeguard.utils.paths import resolve

    tcfg = load_config(TrainConfig, config, overrides)
    dcfg = load_config(DataConfig, tcfg.data_config)
    train_df = read_split(dcfg.splits_dir, "train")
    path = fig_augmentation_preview(train_df, dcfg, tcfg.augment, tcfg.img_size, resolve(out))
    typer.echo(str(path))


@embed_app.command("audit")
def embed_audit(
    config: ConfigOpt = Path("configs/data.yaml"),
    overrides: SetOpt = None,
) -> None:
    """Find likely label errors (cleanlab) and score the DINOv2 linear-probe baseline."""
    from coffeeguard.embeddings.audit import run_audit

    result = run_audit(_data_cfg(config, overrides))
    typer.echo(json.dumps(result, indent=2, default=str))


@app.command()
def train(
    config: Annotated[Path, typer.Option("--config", "-c", help="Training recipe YAML.")],
    overrides: SetOpt = None,
    seed: Annotated[int | None, typer.Option(help="Override the recipe seed.")] = None,
) -> None:
    """Train a model locally (CPU/GPU). Writes runs/<timestamp>-<name>-s<seed>/."""
    from coffeeguard.training.trainer import train as run_train

    extra = list(overrides or [])
    if seed is not None:
        extra.append(f"seed={seed}")
    run_dir = run_train(load_config(TrainConfig, config, extra))
    typer.echo(str(run_dir))


@app.command()
def evaluate(
    bundle: Annotated[
        list[Path], typer.Option("--bundle", "-b", help="Bundle dir(s); repeatable.")
    ],
    config: ConfigOpt = Path("configs/data.yaml"),
    alpha: Annotated[
        float,
        typer.Option(help="Conformal miscoverage; 0.02 = 98% sets."),
    ] = 0.02,
    out_root: Annotated[Path, typer.Option(help="Output folder.")] = Path("artifacts/eval"),
) -> None:
    """Evaluate bundles on val + test: CIs, calibration, conformal sets, error analysis.

    The first bundle is the main model; the others are compared against it (paired bootstrap).
    """
    from coffeeguard.evaluation.evaluate import compare, evaluate_bundle
    from coffeeguard.utils.io import write_json
    from coffeeguard.utils.paths import resolve

    cfg = _data_cfg(config, None)
    out = resolve(out_root)
    results = [evaluate_bundle(resolve(b), cfg, out, alpha) for b in bundle]
    summary = compare(results, out)
    write_json(resolve("artifacts/metrics/test_metrics.json"), summary)
    typer.echo(json.dumps(summary, indent=2))


ood_app = typer.Typer(
    help="Out-of-distribution gate: data, fitting, evaluation.", no_args_is_help=True
)
app.add_typer(ood_app, name="ood")


@ood_app.command("collect")
def ood_collect(
    out: Annotated[Path, typer.Option(help="Output folder (git-ignored).")] = Path("data/ood"),
) -> None:
    """Download a sample of public non-coffee images (+ synthetic frames), split by source."""
    from coffeeguard.ood.collect import collect
    from coffeeguard.utils.paths import resolve

    typer.echo(json.dumps(collect(resolve(out)), indent=2))


@ood_app.command("fit")
def ood_fit(
    bundle: Annotated[Path, typer.Option("--bundle", "-b", help="Bundle dir.")],
    config: ConfigOpt = Path("configs/data.yaml"),
    ood_root: Annotated[Path, typer.Option(help="OOD images.")] = Path("data/ood"),
    out_root: Annotated[Path, typer.Option(help="Report folder.")] = Path("artifacts/ood"),
) -> None:
    """Fit quality gate, OOD scorer and thresholds; report on test/OOD-test; save to bundle."""
    from coffeeguard.ood.fit import fit_gates
    from coffeeguard.utils.paths import resolve

    res = fit_gates(resolve(bundle), _data_cfg(config, None), resolve(ood_root), resolve(out_root))
    typer.echo(
        json.dumps(
            {k: res[k] for k in ("scorer", "tau_ood", "tau_conf", "test", "decisions")}, indent=2
        )
    )


@app.command()
def explain(
    bundle: Annotated[list[Path], typer.Option("--bundle", "-b", help="Bundle dir(s).")],
    config: ConfigOpt = Path("configs/data.yaml"),
    out_root: Annotated[Path, typer.Option(help="Output folder.")] = Path("artifacts/explain"),
) -> None:
    """CAM galleries, leaf-focus score and deletion faithfulness on the test split."""
    from coffeeguard.explainability.analysis import run_explain
    from coffeeguard.utils.paths import resolve

    cfg = _data_cfg(config, None)
    for b in bundle:
        res = run_explain(resolve(b), cfg, resolve(out_root))
        typer.echo(json.dumps({k: v for k, v in res.items() if k != "deletion"}, indent=2))


@app.command()
def robustness(
    bundle: Annotated[list[Path], typer.Option("--bundle", "-b", help="Bundle dir(s).")],
    config: ConfigOpt = Path("configs/data.yaml"),
    out_root: Annotated[Path, typer.Option(help="Output folder.")] = Path("artifacts/robustness"),
) -> None:
    """Corruption sweep (9 x 5 severities) + leaf/background shortcut test on test."""
    from coffeeguard.robustness.sweep import run_robustness, summary_table
    from coffeeguard.utils.paths import resolve

    cfg = _data_cfg(config, None)
    out = resolve(out_root)
    for b in bundle:
        run_robustness(resolve(b), cfg, out)
    # the summary covers every bundle evaluated so far, not only this call's
    from coffeeguard.utils.io import read_json

    table = summary_table([read_json(f) for f in sorted(out.glob("*/robustness.json"))])
    table.to_csv(out / "summary.csv", index=False)
    typer.echo(table.T.to_string())


@app.command()
def curves(
    run: Annotated[Path, typer.Option("--run", "-r", help="Run directory.")],
) -> None:
    """Plot a run's training curves → <run>/figures/training_curves.png."""
    from coffeeguard.training.curves import plot_curves
    from coffeeguard.utils.paths import resolve

    typer.echo(str(plot_curves(resolve(run))))


@app.command()
def export(
    run: Annotated[Path, typer.Option("--run", "-r", help="Run directory containing best.pt.")],
    name: Annotated[str | None, typer.Option(help="Bundle name (default: run name).")] = None,
    out_root: Annotated[Path, typer.Option(help="Where bundles live.")] = Path("artifacts/models"),
) -> None:
    """Export a run's best checkpoint to an ONNX bundle (with torch↔ONNX parity check)."""
    from coffeeguard.export.onnx_export import export_bundle
    from coffeeguard.utils.paths import resolve

    run_dir = resolve(run)
    out = resolve(out_root) / (name or run_dir.name)
    typer.echo(str(export_bundle(run_dir / "best.pt", out)))


remote_app = typer.Typer(help="Train on a Kaggle GPU from here.", no_args_is_help=True)
app.add_typer(remote_app, name="remote")


@remote_app.command("upload-data")
def remote_upload_data(
    config: ConfigOpt = Path("configs/data.yaml"),
    overrides: SetOpt = None,
) -> None:
    """Upload data/processed as the private Kaggle dataset <user>/coffeeguard-processed."""
    from coffeeguard.training.remote import upload_data

    typer.echo(upload_data(_data_cfg(config, overrides)))


@remote_app.command("train")
def remote_train(
    config: Annotated[list[Path], typer.Option("--config", "-c", help="Recipe(s); repeatable.")],
    seeds: Annotated[str, typer.Option(help="Comma-separated seeds, e.g. 0,1,2.")] = "0",
    overrides: SetOpt = None,
    wait: Annotated[bool, typer.Option(help="Wait and download results.")] = True,
    gpu: Annotated[str, typer.Option(help="Kaggle machine shape.")] = "NvidiaTeslaT4",
) -> None:
    """Run one or more recipes × seeds on a Kaggle GPU; results land in runs/."""
    from coffeeguard.training.remote import push_training, wait_and_fetch
    from coffeeguard.utils.paths import portable_path

    jobs = [
        {"config": portable_path(c), "seed": int(s), "overrides": list(overrides or [])}
        for c in config
        for s in seeds.split(",")
    ]
    ref = push_training(jobs, gpu=gpu)
    typer.echo(f"Pushed https://www.kaggle.com/code/{ref}")
    if wait:
        typer.echo(str(wait_and_fetch(ref)))


@remote_app.command("fetch")
def remote_fetch(
    kernel: Annotated[str, typer.Argument(help="Kernel ref, e.g. user/coffeeguard-train.")],
) -> None:
    """Wait for a pushed kernel and download its runs."""
    from coffeeguard.training.remote import wait_and_fetch

    typer.echo(str(wait_and_fetch(kernel)))


if __name__ == "__main__":
    app()
