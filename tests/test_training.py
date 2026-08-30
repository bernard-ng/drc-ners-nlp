from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from ners.config import TrainingConfig
from ners.model import NameSexClassifier
from ners.training import train_model


def test_streaming_training_persists_reusable_artifacts(tmp_path: Path) -> None:
    dataset_path = tmp_path / "names.csv"
    model_path = tmp_path / "model.joblib"
    rows = []
    for index in range(400):
        rows.extend(
            [
                {"name": f"makena family {index} esther", "sex": "f"},
                {"name": f"kabongo family {index} jean", "sex": "m"},
            ]
        )
    pl.DataFrame(rows).write_csv(dataset_path)

    config = TrainingConfig(
        dataset_path=dataset_path,
        model_path=model_path,
        chunk_size=73,
        n_features=2**12,
        ngram_min=2,
        ngram_max=4,
        test_fraction=0.2,
        random_seed=7,
    )
    result = train_model(config)

    assert result.rows > 0
    assert result.accuracy > 0.9
    assert model_path.is_file()
    assert config.resolved_metrics_path.is_file()

    classifier = NameSexClassifier.load(model_path)
    probabilities = classifier.predict_proba(["example esther", "example jean"])
    assert probabilities.shape == (2, 2)
    assert set(classifier.classes) == {"f", "m"}
    assert classifier.metadata["target"] == "sex"

    metrics = json.loads(config.resolved_metrics_path.read_text(encoding="utf-8"))
    assert metrics["evaluation"]["rows"] == result.rows
    assert metrics["training_config"]["dataset_path"] == str(dataset_path)
