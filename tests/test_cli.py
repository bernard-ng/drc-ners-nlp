from typer.testing import CliRunner

from ners.cli import app


def test_cli_exposes_only_model_workflows() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "train" in result.stdout
    assert "evaluate" in result.stdout
    assert "predict" in result.stdout
    assert "web" in result.stdout
    assert "pipeline" not in result.stdout
    assert "annotation" not in result.stdout
