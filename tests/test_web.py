from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
from streamlit.testing.v1 import AppTest

from drc_names_classifier.config import ExperimentConfig
from drc_names_classifier.experiments import ExperimentResult, ExperimentStatus
from drc_names_classifier.models import MODEL_REGISTRY
from drc_names_classifier.web import (
    confusion_matrix_frame,
    experiment_results_frame,
    inspect_dataset,
    metric_frame,
    model_availability_frame,
)


def test_dataset_snapshot_is_read_only_and_polars_native(tmp_path: Path) -> None:
    path = tmp_path / "names.csv"
    pl.DataFrame(
        {
            "name": ["synthetic name one", "synthetic name two", ""],
            "sex": ["f", "m", "f"],
            "source": ["a", "b", "c"],
        }
    ).write_csv(path)

    snapshot = inspect_dataset(path, preview_rows=2)

    assert snapshot.row_count == 3
    assert snapshot.empty_name_count == 1
    assert snapshot.invalid_label_count == 0
    assert snapshot.label_distribution.to_dicts() == [
        {"sex": "f", "rows": 2, "share": 2 / 3},
        {"sex": "m", "rows": 1, "share": 1 / 3},
    ]
    assert snapshot.preview.columns == ["name", "sex"]
    assert snapshot.preview.height == 2
    assert list(tmp_path.iterdir()) == [path]


def test_experiment_view_models_have_stable_schemas() -> None:
    start = datetime(2026, 8, 30, 12, 0)
    result = ExperimentResult(
        experiment_id="example",
        config=ExperimentConfig(name="example", model_type="logistic_regression"),
        start_time=start,
        end_time=start + timedelta(seconds=12.5),
        status=ExperimentStatus.COMPLETED,
        train_metrics={"accuracy": 0.9},
        test_metrics={"accuracy": 0.8, "macro_f1": 0.75},
        confusion_matrix=[[8, 2], [3, 7]],
        train_size=80,
        test_size=20,
    )

    experiments = experiment_results_frame([result])

    assert experiments.row(0, named=True)["duration_seconds"] == 12.5
    assert experiments.row(0, named=True)["test_macro_f1"] == 0.75
    assert metric_frame(result).get_column("metric").to_list() == ["accuracy", "macro_f1"]
    assert confusion_matrix_frame(result).get_column("rows").to_list() == [8, 2, 3, 7]
    assert model_availability_frame(MODEL_REGISTRY).height == 12


def test_streamlit_entrypoint_renders_without_exceptions() -> None:
    entrypoint = Path(__file__).parents[1] / "src" / "drc_names_classifier" / "web" / "app.py"

    app = AppTest.from_file(entrypoint, default_timeout=15).run()

    assert not app.exception
    assert app.title[0].value == "DRC Names Classifier"
    assert [tab.label for tab in app.tabs] == [
        "Experiments",
        "Run experiment",
        "Results",
        "Dataset",
    ]
