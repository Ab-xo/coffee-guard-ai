from __future__ import annotations

import json

from typer.testing import CliRunner

from coffeeguard.cli import app

runner = CliRunner()


def test_help_lists_groups():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "data" in result.output


def test_info_prints_resolved_config():
    result = runner.invoke(app, ["info", "--set", "min_side=42"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["data"]["min_side"] == 42
    assert "python" in payload["env"]
