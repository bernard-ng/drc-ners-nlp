"""Sparse full-name feature infrastructure for tree and boosting models."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer

from drc_names_classifier.config import ExperimentConfig
from drc_names_classifier.utils import full_name_series


class SparseTextFeatures:
    """Reusable fitted character n-grams for tree and boosting models."""

    config: ExperimentConfig
    vectorizer: TfidfVectorizer

    def prepare_sparse_text(self, frame: pl.DataFrame) -> Any:
        text = full_name_series(frame).to_numpy()

        if not hasattr(self, "vectorizer"):
            params = self.config.model_params
            ngram_range = params.get("ngram_range", (2, 4))
            if isinstance(ngram_range, list):
                ngram_range = tuple(ngram_range)
            self.vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=ngram_range,
                max_features=int(params.get("max_features", 4096)),
                dtype=np.float32,  # pyright: ignore[reportArgumentType]
                sublinear_tf=True,
            )
            return self.vectorizer.fit_transform(text)
        return self.vectorizer.transform(text)
