from __future__ import annotations

from pathlib import Path
from typing import cast
import warnings

import polars as pl
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from drc_names_classifier.config import ExperimentConfig, ExperimentSettings
from drc_names_classifier.experiments import (
    ExperimentBuilder,
    ExperimentRunner,
)
from drc_names_classifier.models import MODEL_REGISTRY


EXPECTED_MODELS = {
    "bigru",
    "cnn",
    "dummy",
    "ensemble",
    "lightgbm",
    "logistic_regression",
    "lstm",
    "naive_bayes",
    "position_logistic_regression",
    "random_forest",
    "transformer",
    "xgboost",
}


def test_all_study_architectures_are_registered_and_templated() -> None:
    config = ExperimentSettings(sample_fraction=0.1)
    templates = ExperimentBuilder(config).load_templates()
    template_models = {template["model_type"] for template in templates["baseline_experiments"]}

    assert set(MODEL_REGISTRY) == EXPECTED_MODELS
    assert template_models == EXPECTED_MODELS
    assert all(
        template["features"] == ["full_name"] for template in templates["baseline_experiments"]
    )


def test_experiment_runner_reuses_split_and_persists_models(tmp_path: Path) -> None:
    dataset_path = tmp_path / "names.csv"
    rows = []
    for index in range(250):
        rows.extend(
            [
                {"name": f"makena family {index} esther", "sex": "f"},
                {"name": f"kabongo family {index} jean", "sex": "m"},
            ]
        )
    pl.DataFrame(rows).write_csv(dataset_path)

    config = ExperimentSettings(
        dataset_path=dataset_path,
        models_dir=tmp_path / "models",
        outputs_dir=tmp_path / "outputs",
        chunk_size=61,
        sample_fraction=1.0,
        test_fraction=0.2,
    )
    runner = ExperimentRunner(config)
    experiments = [
        ExperimentConfig(
            name="test_logistic",
            model_type="logistic_regression",
            model_params={
                "max_features": 1024,
                "max_iter": 500,
                "solver": "saga",
            },
            sample_fraction=1.0,
            test_fraction=0.2,
            cross_validation_folds=0,
        ),
        ExperimentConfig(
            name="test_naive_bayes",
            model_type="naive_bayes",
            model_params={"max_features": 1024},
            sample_fraction=1.0,
            test_fraction=0.2,
            cross_validation_folds=0,
        ),
    ]

    experiment_ids = runner.run_batch(experiments)
    results = [runner.tracker.get(value) for value in experiment_ids]

    assert len(experiment_ids) == 2
    assert all(result is not None for result in results)
    assert len({(result.train_size, result.test_size) for result in results if result}) == 1
    assert all(result.model_path for result in results if result)
    assert all("prediction_examples" not in result.to_dict() for result in results if result)

    loaded = runner.load_model(experiment_ids[0])
    assert loaded is not None
    predictions = loaded.predict(pl.DataFrame({"name": ["example esther", "example jean"]}))
    assert len(predictions) == 2


def test_registry_loads_available_model_without_importing_tensorflow() -> None:
    config = ExperimentConfig(name="test", model_type="logistic_regression")
    model = MODEL_REGISTRY.create(config)

    assert not model.is_fitted


def test_logistic_regression_scales_sparse_counts_and_converges() -> None:
    rows = [
        {
            "name": f"{'makena' if index % 2 else 'kabongo'} family {index}",
            "sex": "f" if index % 2 else "m",
        }
        for index in range(400)
    ]
    frame = pl.DataFrame(rows)
    config = ExperimentConfig(
        name="convergence",
        model_type="logistic_regression",
        model_params={"max_features": 1024},
    )
    model = MODEL_REGISTRY.create(config)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(frame.drop("sex"), frame.get_column("sex"))

    fitted_pipeline = cast(Pipeline, model.model)
    assert isinstance(fitted_pipeline, Pipeline)
    assert isinstance(fitted_pipeline.named_steps["scale"], MaxAbsScaler)
    classifier = fitted_pipeline.named_steps["classifier"]
    assert classifier.n_iter_[0] < classifier.max_iter
    assert not any(issubclass(item.category, ConvergenceWarning) for item in caught)


def test_public_experiment_config_round_trip() -> None:
    config = ExperimentConfig(
        name="public_api",
        model_type="naive_bayes",
        tags=("baseline", "polars"),
        model_params={"alpha": 0.5},
        sample_fraction=0.25,
        test_fraction=0.3,
    )

    assert ExperimentConfig.from_dict(config.to_dict()) == config
    assert config.to_dict()["test_fraction"] == 0.3


def test_builder_creates_controlled_name_view_pair() -> None:
    builder = ExperimentBuilder(ExperimentSettings(sample_fraction=0.1))
    template = next(
        value
        for value in builder.templates("baseline")
        if value["model_type"] == "logistic_regression"
    )

    full, native = builder.name_view_pair(
        template,
        sample_fraction=0.1,
        test_fraction=0.2,
    )

    assert (full.input_view, native.input_view) == ("full", "native_only")
    assert full.split_group_view == native.split_group_view == "native_only"
    assert full.required_token_count == native.required_token_count == 3
    assert full.sample_fraction == native.sample_fraction == 0.1


def test_position_aware_model_leaves_third_channel_missing_for_native_view() -> None:
    config = ExperimentConfig(
        name="position_test",
        model_type="position_logistic_regression",
        model_params={
            "sequence_max_features": 256,
            "token_max_features": 128,
            "min_df": 1,
            "max_iter": 300,
        },
    )
    model = MODEL_REGISTRY.create(config)
    frame = pl.DataFrame(
        {
            "name": [
                "kabongo ilunga",
                "kavira mapendo",
                "mwamba kalala",
                "zawadi tumaini",
            ],
            "sex": ["m", "f", "m", "f"],
        }
    )

    model.fit(frame.drop("sex"), frame.get_column("sex"))
    predictions = model.predict(frame.drop("sex"))

    assert len(predictions) == frame.height
    assert model.get_feature_importance()


def test_saved_input_view_is_enforced_during_prediction() -> None:
    config = ExperimentConfig(
        name="native_artifact",
        model_type="logistic_regression",
        input_view="native_only",
        model_params={"max_features": 128, "min_df": 1, "max_iter": 200},
    )
    model = MODEL_REGISTRY.create(config)
    frame = pl.DataFrame(
        {
            "name": [
                "kabongo ilunga jean",
                "kavira mapendo esther",
                "mwamba kalala paul",
                "zawadi tumaini marie",
            ],
            "sex": ["m", "f", "m", "f"],
        }
    )
    model.fit(frame.drop("sex"), frame.get_column("sex"))

    predictions = model.predict(
        pl.DataFrame(
            {
                "name": [
                    "kabongo ilunga esther",
                    "kabongo ilunga jean",
                ]
            }
        )
    )

    assert predictions[0] == predictions[1]
