from typing import Any, cast

import numpy as np
import pandas as pd
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

from ners.research.neural_network_model import NeuralNetworkModel


class CNNModel(NeuralNetworkModel):
    """1D Convolutional Neural Network for character patterns"""

    def build_model(self, vocab_size: int, **kwargs) -> Any:
        """Build CNN model with known vocabulary size"""
        params = kwargs

        model = Sequential(
            [
                Embedding(
                    input_dim=vocab_size,
                    output_dim=params.get("embedding_dim", 64),
                ),
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

    def prepare_features(self, X: pd.DataFrame) -> np.ndarray:
        """Tokenize text features at character level and return padded sequences"""
        text_data = self._collect_text_corpus(X)

        if self.tokenizer is None:
            self.tokenizer = Tokenizer(
                char_level=True, lower=True, oov_token="<OOV>"
            )

        # Pyright fix: ensure tokenizer is treated as non-Optional
        tokenizer = cast(Tokenizer, self.tokenizer)
        tokenizer.fit_on_texts(text_data)

        sequences = tokenizer.texts_to_sequences(text_data)
        max_len = self.config.model_params.get("max_len", 20)

        return pad_sequences(
            sequences,
            maxlen=max_len,
            padding="post",
            truncating="post",
        )
