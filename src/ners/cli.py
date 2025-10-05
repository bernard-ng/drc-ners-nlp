from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from ners.core.config import setup_config, PipelineConfig

app = typer.Typer(help="DRC NERS command-line interface", no_args_is_help=True)


# -------------------------
# Pipeline commands
# -------------------------
pipeline_app = typer.Typer(help="Data processing pipeline")
app.add_typer(pipeline_app, name="pipeline")


@pipeline_app.command("run")
def pipeline_run(
    config: Optional[Path] = typer.Option(None, help="Path to configuration file"),
    env: str = typer.Option("development", help="Environment name"),
) -> None:
    """Run the full processing pipeline."""
    from ners.main import run_pipeline as _run_pipeline

    cfg = setup_config(config_path=config, env=env)
    code = _run_pipeline(cfg)
    raise typer.Exit(code)


# -------------------------
# NER commands
# -------------------------
ner_app = typer.Typer(help="NER dataset and model")
app.add_typer(ner_app, name="ner")


def _load_config(config: Optional[Path], env: str) -> PipelineConfig:
    return setup_config(config_path=config, env=env)


@ner_app.command("feature")
def ner_feature(
    config: Optional[Path] = typer.Option(None, help="Path to configuration file"),
    env: str = typer.Option("development", help="Environment name"),
) -> None:
    from ners.ner import feature as _feature

    cfg = _load_config(config, env)
    _feature(cfg)


@ner_app.command("build")
def ner_build(
    config: Optional[Path] = typer.Option(None, help="Path to configuration file"),
    env: str = typer.Option("development", help="Environment name"),
) -> None:
    from ners.ner import build as _build

    cfg = _load_config(config, env)
    _build(cfg)


@ner_app.command("train")
def ner_train(
    config: Optional[Path] = typer.Option(None, help="Path to configuration file"),
    env: str = typer.Option("development", help="Environment name"),
) -> None:
    from ners.ner import train as _train

    cfg = _load_config(config, env)
    _train(cfg)


@ner_app.command("run")
def ner_run(
    config: Optional[Path] = typer.Option(None, help="Path to configuration file"),
    env: str = typer.Option("development", help="Environment name"),
    reset: bool = typer.Option(
        False, help="Reset intermediate outputs and rerun all steps"
    ),
) -> None:
    from ners.ner import run_pipeline as _ner_pipeline

    cfg = _load_config(config, env)
    code = _ner_pipeline(cfg, reset)
    raise typer.Exit(code)


# -------------------------
# Research commands
# -------------------------
research_app = typer.Typer(help="Research experiments and training")
app.add_typer(research_app, name="research")


@research_app.command("train")
def research_train(
    name: str = typer.Option(..., "--name", help="Model name to train"),
    type: str = typer.Option(..., "--type", help="Experiment type"),
    templates: str = typer.Option(
        "research_templates.yaml", help="Templates file path"
    ),
    config: Optional[Path] = typer.Option(None, help="Path to configuration file"),
    env: str = typer.Option("development", help="Environment name"),
) -> None:
    from ners.research.experiment.experiment_builder import ExperimentBuilder
    from ners.research.model_trainer import ModelTrainer

    cfg = _load_config(config, env)
    exp_builder = ExperimentBuilder(cfg)
    tmpl = exp_builder.load_templates(templates)
    exp_cfg = exp_builder.find_template(tmpl, name, type)

    trainer = ModelTrainer(cfg)
    trainer.train_single_model(
        model_name=exp_cfg.get("name"),
        model_type=exp_cfg.get("model_type"),
        features=exp_cfg.get("features"),
        model_params=exp_cfg.get("model_params", {}),
        tags=exp_cfg.get("tags", []),
    )


# -------------------------
# Monitor commands
# -------------------------
monitor_app = typer.Typer(help="Monitor pipeline checkpoints")
app.add_typer(monitor_app, name="monitor")


@monitor_app.command("status")
def monitor_status(
    config: Optional[Path] = typer.Option(None, help="Path to configuration file"),
    env: str = typer.Option("development", help="Environment name"),
    detailed: bool = typer.Option(
        False, help="Show detailed status (failed batch IDs)"
    ),
) -> None:
    _ = _load_config(config, env)
    from ners.processing.monitoring.pipeline_monitor import PipelineMonitor

    PipelineMonitor().print_status(detailed=detailed)


@monitor_app.command("clean")
def monitor_clean(
    step: Optional[str] = typer.Option(None, help="Step to clean; default all"),
    keep_last: int = typer.Option(1, help="Number of latest checkpoint files to keep"),
    force: bool = typer.Option(False, help="Do not ask for confirmation"),
    config: Optional[Path] = typer.Option(None, help="Path to configuration file"),
    env: str = typer.Option("development", help="Environment name"),
) -> None:
    _ = _load_config(config, env)
    from ners.processing.monitoring.pipeline_monitor import PipelineMonitor

    mon = PipelineMonitor()
    if not force:
        typer.confirm("Clean checkpoints?", abort=True)

    if step:
        mon.clean_step_checkpoints(step, keep_last)
    else:
        for s in mon.steps:
            mon.clean_step_checkpoints(s, keep_last)


@monitor_app.command("reset")
def monitor_reset(
    step: Optional[str] = typer.Option(None, help="Step to reset; default all"),
    force: bool = typer.Option(False, help="Do not ask for confirmation"),
    config: Optional[Path] = typer.Option(None, help="Path to configuration file"),
    env: str = typer.Option("development", help="Environment name"),
) -> None:
    _ = _load_config(config, env)
    from ners.processing.monitoring.pipeline_monitor import PipelineMonitor

    mon = PipelineMonitor()
    if not force:
        msg = f"Reset {step or 'all steps'}? This deletes checkpoints."
        typer.confirm(msg, abort=True)

    if step:
        mon.reset_step(step)
    else:
        for s in mon.steps:
            mon.reset_step(s)


# -------------------------
# Web commands
# -------------------------
web_app = typer.Typer(help="Web UI wrapper")
app.add_typer(web_app, name="web")


@web_app.command("run")
def web_run(
    config: Optional[Path] = typer.Option(None, help="Path to configuration file"),
    env: str = typer.Option("development", help="Environment name"),
) -> None:
    app_path = Path(__file__).parent / "web" / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
    ]
    # Pass configuration via environment variables to avoid argparse in Streamlit
    env_vars = os.environ.copy()
    if config is not None:
        env_vars["NERS_CONFIG"] = str(config)
    env_vars["NERS_ENV"] = env

    raise typer.Exit(subprocess.call(cmd, env=env_vars))


if __name__ == "__main__":  # pragma: no cover
    app()
