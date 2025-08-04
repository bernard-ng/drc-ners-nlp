from typing import Any

import numpy as np
import pandas as pd
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from research.neural_network_model import NeuralNetworkModel


class LSTMModel(NeuralNetworkModel):
    """LSTM model for sequence learning"""

    def build_model_with_vocab(self, vocab_size: int, max_len: int = 6, **kwargs) -> Any:
        params = kwargs
        model = Sequential(
            [
                Embedding(input_dim=vocab_size, output_dim=params.get("embedding_dim", 64)),
                Bidirectional(LSTM(params.get("lstm_units", 32), return_sequences=True)),
                Bidirectional(LSTM(params.get("lstm_units", 32))),
                Dense(64, activation="relu"),
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

        # Initialize tokenizer if needed
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(char_level=False, lower=True, oov_token="<OOV>")
            self.tokenizer.fit_on_texts(text_data)

        # Convert to sequences
        sequences = self.tokenizer.texts_to_sequences(text_data[: len(X)])
        max_len = self.config.model_params.get("max_len", 6)

        return pad_sequences(sequences, maxlen=max_len, padding="post")
