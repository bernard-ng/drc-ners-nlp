from __future__ import annotations

import json
import logging
from pathlib import Path
import subprocess
import sys
from typing import Annotated

import typer

from ners.config import (
    DEFAULT_DATASET_PATH,
    ExperimentSettings,
)


app = typer.Typer(
    help="Run controlled CongoNames full-name classification experiments.",
    no_args_is_help=True,
)

experiments_app = typer.Typer(help="Run and compare the study's model architectures.")
app.add_typer(experiments_app, name="experiments")


@app.command("web")
def web_command(
    port: Annotated[int, typer.Option(min=1, max=65_535)] = 8501,
    open_browser: Annotated[
        bool,
        typer.Option(help="Open the local interface in the default browser."),
    ] = True,
) -> None:
    """Launch the local experiment and dataset interface."""

    entrypoint = Path(__file__).with_name("web") / "app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(entrypoint),
        "--server.port",
        str(port),
        "--server.headless",
        str(not open_browser).lower(),
    ]
    result = subprocess.run(command, check=False)
    if result.returncode:
        raise typer.Exit(result.returncode)


@experiments_app.command("list")
def experiments_list() -> None:
    """List each configured model and its local dependency status."""

    from ners.models import MODEL_REGISTRY

    rows = [
        {
            "model": name,
            "family": spec.family,
            "available": spec.available,
            "dependency": spec.dependency,
            "availability_error": spec.availability_error,
        }
        for name, spec in MODEL_REGISTRY.items()
    ]
    typer.echo(json.dumps(rows, indent=2))


@experiments_app.command("train")
def experiments_train(
    name: Annotated[str, typer.Option(help="Experiment name from the template file.")],
    experiment_type: Annotated[
        str,
        typer.Option("--type", help="Template section: baseline, advanced, or tuning."),
    ] = "baseline",
    dataset: Annotated[Path, typer.Option(help="Published names.csv input.")] = (
        DEFAULT_DATASET_PATH
    ),
    templates: Annotated[
        Path,
        typer.Option(help="Experiment definitions."),
    ] = Path("config/experiment_templates.yaml"),
    sample_fraction: Annotated[
        float,
        typer.Option(min=0.0001, max=1.0, help="Shared deterministic study sample."),
    ] = 0.01,
    test_fraction: Annotated[
        float,
        typer.Option(min=0.001, max=0.999, help="Held-out full-name groups."),
    ] = 0.2,
    chunk_size: Annotated[int, typer.Option(min=1)] = 100_000,
) -> None:
    """Train one model from the experiment templates."""

    from ners.experiments import ExperimentBuilder, ExperimentRunner

    _configure_logging()
    try:
        config = ExperimentSettings(
            dataset_path=dataset,
            templates_path=templates,
            sample_fraction=sample_fraction,
            test_fraction=test_fraction,
            chunk_size=chunk_size,
        )
        builder = ExperimentBuilder(config)
        experiment = builder.build(
            name,
            experiment_type=experiment_type,
            sample_fraction=sample_fraction,
            test_fraction=test_fraction,
        )
        runner = ExperimentRunner(config)
        experiment_id = runner.run(experiment)
        result = runner.tracker.get(experiment_id)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        typer.secho(f"Experiment run failed: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from error

    typer.echo(json.dumps(result.to_dict() if result else {}, indent=2))


@experiments_app.command("suite")
def experiments_suite(
    dataset: Annotated[Path, typer.Option(help="Published names.csv input.")] = (
        DEFAULT_DATASET_PATH
    ),
    templates: Annotated[Path, typer.Option()] = Path("config/experiment_templates.yaml"),
    sample_fraction: Annotated[
        float,
        typer.Option(min=0.0001, max=1.0, help="Shared deterministic study sample."),
    ] = 0.01,
    test_fraction: Annotated[
        float,
        typer.Option(min=0.001, max=0.999),
    ] = 0.2,
    chunk_size: Annotated[int, typer.Option(min=1)] = 100_000,
) -> None:
    """Run all locally available baseline architectures on the same split."""

    from ners.experiments import (
        ExperimentBuilder,
        ExperimentRunner,
    )
    from ners.models import MODEL_REGISTRY

    _configure_logging()
    config = ExperimentSettings(
        dataset_path=dataset,
        templates_path=templates,
        sample_fraction=sample_fraction,
        test_fraction=test_fraction,
        chunk_size=chunk_size,
    )
    builder = ExperimentBuilder(config)
    experiments = []
    skipped = []
    for template in builder.templates("baseline"):
        spec = MODEL_REGISTRY[str(template["model_type"])]
        if not spec.available:
            skipped.append(
                {
                    "model": template["model_type"],
                    "dependency": spec.dependency,
                    "reason": spec.availability_error,
                }
            )
            continue
        experiments.append(
            builder.from_template(
                template,
                sample_fraction=sample_fraction,
                test_fraction=test_fraction,
            )
        )

    runner = ExperimentRunner(config)
    experiment_ids = runner.run_batch(experiments)
    comparison = runner.compare(experiment_ids)
    output = {
        "completed": experiment_ids,
        "skipped": skipped,
        "comparison": comparison.to_dicts(),
    }
    typer.echo(json.dumps(output, indent=2, default=str))


@experiments_app.command("compare-views")
def experiments_compare_views(
    names: Annotated[
        list[str] | None,
        typer.Option(
            "--name",
            help="Architecture template to compare; repeat to select several.",
        ),
    ] = None,
    dataset: Annotated[Path, typer.Option(help="Published names.csv input.")] = (
        DEFAULT_DATASET_PATH
    ),
    templates: Annotated[Path, typer.Option()] = Path("config/experiment_templates.yaml"),
    sample_fraction: Annotated[
        float,
        typer.Option(min=0.0001, max=1.0, help="Shared deterministic study sample."),
    ] = 0.01,
    test_fraction: Annotated[
        float,
        typer.Option(min=0.001, max=0.999),
    ] = 0.2,
    chunk_size: Annotated[int, typer.Option(min=1)] = 100_000,
) -> None:
    """Compare surname-included and native-only views on identical held-out groups."""

    from ners.experiments import ExperimentBuilder, ExperimentRunner
    from ners.models import MODEL_REGISTRY

    _configure_logging()
    config = ExperimentSettings(
        dataset_path=dataset,
        templates_path=templates,
        sample_fraction=sample_fraction,
        test_fraction=test_fraction,
        chunk_size=chunk_size,
    )
    builder = ExperimentBuilder(config)
    selected = set(names or MODEL_REGISTRY.names())
    unknown = selected - set(MODEL_REGISTRY)
    if unknown:
        available = ", ".join(MODEL_REGISTRY)
        typer.secho(
            f"Unknown model(s): {', '.join(sorted(unknown))}. Available: {available}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    experiments = []
    skipped = []
    for template in builder.templates("baseline"):
        model_type = str(template["model_type"])
        if model_type not in selected:
            continue
        spec = MODEL_REGISTRY[model_type]
        if not spec.available:
            skipped.append(
                {
                    "model": model_type,
                    "dependency": spec.dependency,
                    "reason": spec.availability_error,
                }
            )
            continue
        experiments.extend(
            builder.name_view_pair(
                template,
                sample_fraction=sample_fraction,
                test_fraction=test_fraction,
            )
        )

    runner = ExperimentRunner(config)
    experiment_ids = runner.run_batch(experiments)
    comparison = runner.compare(experiment_ids)
    deltas = _name_view_deltas(comparison.to_dicts())
    output = {
        "study_contract": {
            "cohort": "exactly_three_tokens",
            "split_group": "first_two_native_tokens",
            "views": ["full", "native_only"],
        },
        "completed": experiment_ids,
        "skipped": skipped,
        "comparison": comparison.to_dicts(),
        "surname_gain": deltas,
    }
    typer.echo(json.dumps(output, indent=2, default=str))


def _name_view_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Compute full-minus-native metric deltas for completed model pairs."""

    by_model: dict[str, dict[str, dict[str, object]]] = {}
    for row in rows:
        model_type = str(row.get("model_type", ""))
        input_view = str(row.get("input_view", ""))
        by_model.setdefault(model_type, {})[input_view] = row

    output: list[dict[str, object]] = []
    for model_type, views in sorted(by_model.items()):
        full = views.get("full")
        native = views.get("native_only")
        if full is None or native is None:
            continue
        values: dict[str, object] = {"model_type": model_type}
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "mcc"):
            column = f"test_{metric}"
            if column in full and column in native:
                full_value = full[column]
                native_value = native[column]
                if isinstance(full_value, (int, float)) and isinstance(
                    native_value, (int, float)
                ):
                    values[f"{metric}_gain"] = float(full_value) - float(native_value)
        output.append(values)
    return output


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


if __name__ == "__main__":  # pragma: no cover
    app()
