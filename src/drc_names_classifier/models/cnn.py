# pyright: reportOptionalMemberAccess=false

from typing import Any

import numpy as np
import polars as pl
from tensorflow.keras.layers import (
    Embedding,
    Conv1D,
    MaxPooling1D,
    GlobalMaxPooling1D,
    Dense,
    Dropout,
    SpatialDropout1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from drc_names_classifier.models.neural import NeuralNetworkModel


class CNNModel(NeuralNetworkModel):
    """One-dimensional convolutional full-name classifier."""

    def build_model(self, vocab_size: int, **kwargs) -> Any:
        """Create the untrained character CNN."""

        params = kwargs
        model = Sequential(
            [
                Embedding(input_dim=vocab_size, output_dim=params.get("embedding_dim", 64)),
                SpatialDropout1D(rate=params.get("embedding_dropout", 0.1)),
                Conv1D(
                    filters=params.get("filters", 64),
                    kernel_size=params.get("kernel_size", 3),
                    activation="relu",
                    padding="same",
                ),
                MaxPooling1D(pool_size=2),
                Conv1D(
                    filters=params.get("filters", 64),
                    kernel_size=params.get("kernel_size", 3),
                    activation="relu",
                    padding="same",
                ),
                GlobalMaxPooling1D(),
                Dense(64, activation="relu"),
                Dropout(params.get("dropout", 0.5)),
                Dense(2, activation="softmax", dtype="float32"),
            ]
        )

        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )
        return model

    def prepare_features(self, X: pl.DataFrame) -> np.ndarray:
        """Tokenize the name column as characters and pad each sequence."""

        text_data = self._collect_text_corpus(X)

        if self.tokenizer is None:
            self.tokenizer = Tokenizer(char_level=True, lower=True, oov_token="<OOV>")
            self.tokenizer.fit_on_texts(text_data)

        sequences = self.tokenizer.texts_to_sequences(text_data)
        max_len = self.config.model_params.get("max_len", 32)

        return pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
