from typing import Any

import numpy as np
import pandas as pd
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from research.neural_network_model import NeuralNetworkModel


class BiGRUModel(NeuralNetworkModel):
    """Bidirectional GRU model for name classification"""

    def build_model_with_vocab(self, vocab_size: int, max_len: int = 6, **kwargs) -> Any:
        params = kwargs
        model = Sequential(
            [
                Embedding(input_dim=vocab_size, output_dim=params.get("embedding_dim", 64)),
                Bidirectional(
                    GRU(
                        params.get("gru_units", 32),
                        return_sequences=True,
                        dropout=params.get("dropout", 0.2),
                    )
                ),
                Bidirectional(GRU(params.get("gru_units", 32), dropout=params.get("dropout", 0.2))),
                Dense(64, activation="relu"),
                Dropout(params.get("dropout", 0.5)),
                Dense(2, activation="softmax"),
            ]
        )

        model.compile(
            loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model

    def prepare_features(self, X: pd.DataFrame) -> np.ndarray:
        text_data = []
        for feature_type in self.config.features:
            if feature_type.value in X.columns:
                text_data.extend(X[feature_type.value].astype(str).tolist())

        if not text_data:
            raise ValueError("No text data found in the provided DataFrame.")

        if self.tokenizer is None:
            self.tokenizer = Tokenizer(char_level=False, lower=True, oov_token="<OOV>")
            self.tokenizer.fit_on_texts(text_data)

        sequences = self.tokenizer.texts_to_sequences(text_data[: len(X)])
        max_len = self.config.model_params.get("max_len", 6)

        return pad_sequences(sequences, maxlen=max_len, padding="post")
