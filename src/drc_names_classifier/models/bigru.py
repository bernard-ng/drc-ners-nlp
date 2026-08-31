# pyright: reportOptionalMemberAccess=false

from typing import Any

import numpy as np
import polars as pl
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from drc_names_classifier.models.neural import NeuralNetworkModel


class BiGRUModel(NeuralNetworkModel):
    """Bidirectional GRU full-name classifier."""

    def build_model(self, vocab_size: int, **kwargs) -> Any:
        params = kwargs
        model = Sequential(
            [
                Embedding(
                    input_dim=vocab_size,
                    output_dim=params.get("embedding_dim", 64),
                    mask_zero=True,
                ),
                Bidirectional(
                    GRU(
                        params.get("gru_units", 32),
                        return_sequences=True,
                        dropout=params.get("dropout", 0.2),
                        recurrent_dropout=params.get("recurrent_dropout", 0.1),
                    )
                ),
                Bidirectional(
                    GRU(
                        params.get("gru_units", 32),
                        dropout=params.get("dropout", 0.2),
                        recurrent_dropout=params.get("recurrent_dropout", 0.1),
                    )
                ),
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
        text_data = self._collect_text_corpus(X)

        if self.tokenizer is None:
            self.tokenizer = Tokenizer(char_level=True, lower=True, oov_token="<OOV>")
            self.tokenizer.fit_on_texts(text_data)

        sequences = self.tokenizer.texts_to_sequences(text_data)
        max_len = self.config.model_params.get("max_len", 32)

        return pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
