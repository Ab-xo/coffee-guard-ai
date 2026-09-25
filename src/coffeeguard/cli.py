"""``coffeeguard`` command-line interface.

Every pipeline step is one command. Commands are grouped by phase; heavy imports
(torch, timm, ...) happen inside commands so ``coffeeguard --help`` stays fast.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from coffeeguard.config import DataConfig, load_config
from coffeeguard.utils.log import setup_logging

app = typer.Typer(
    name="coffeeguard",
    help="CoffeeGuard AI — coffee leaf disease pipeline.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

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


if __name__ == "__main__":
    app()
