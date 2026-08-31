from typer.testing import CliRunner

from ners.cli import app


def test_cli_exposes_experiment_workflows_only() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "experiments" in result.stdout
    assert "research" not in result.stdout
    assert "web" in result.stdout
    assert "evaluate" not in result.stdout
    assert "predict" not in result.stdout
    assert "pipeline" not in result.stdout
    assert "annotation" not in result.stdout
