"""``coffeeguard`` command-line interface.

Every pipeline step is one command. Commands are grouped by phase; heavy imports
(torch, timm, ...) happen inside commands so ``coffeeguard --help`` stays fast.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from coffeeguard.config import DataConfig, TrainConfig, load_config
from coffeeguard.utils.log import setup_logging

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
