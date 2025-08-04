from typing import Any

import numpy as np
import pandas as pd
from tensorflow.keras.layers import (
    Embedding,
    Conv1D,
    MaxPooling1D,
    GlobalMaxPooling1D,
    Dense,
    Dropout,
)
from tensorflow.keras.models import Sequential

from research.neural_network_model import NeuralNetworkModel


class CNNModel(NeuralNetworkModel):
    """1D Convolutional Neural Network for character patterns"""

    def build_model_with_vocab(self, vocab_size: int, max_len: int = 20, **kwargs) -> Any:
        """Build CNN model with known vocabulary size"""

        params = kwargs
        model = Sequential(
            [
                Embedding(input_dim=vocab_size, output_dim=params.get("embedding_dim", 64)),
                Conv1D(
                    filters=params.get("filters", 64),
                    kernel_size=params.get("kernel_size", 3),
                    activation="relu",
                ),
                MaxPooling1D(pool_size=2),
                Conv1D(
                    filters=params.get("filters", 64),
                    kernel_size=params.get("kernel_size", 3),
                    activation="relu",
                ),
                GlobalMaxPooling1D(),
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
        """Prepare sequences for CNN using extracted features"""
        # X here contains the features already extracted by FeatureExtractor
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        # Get text data from extracted features - use character level for CNN
        text_data = []
        for feature_type in self.config.features:
            if feature_type.value in X.columns:
                text_data.extend(X[feature_type.value].astype(str).tolist())

        if not text_data:
            # Fallback - should not happen if FeatureExtractor is properly configured
            text_data = [""] * len(X)

        # Initialize character-level tokenizer
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(char_level=True, lower=True, oov_token="<OOV>")
            self.tokenizer.fit_on_texts(text_data)

        sequences = self.tokenizer.texts_to_sequences(text_data[: len(X)])
        max_len = self.config.model_params.get("max_len", 20)  # Longer for character level

        return pad_sequences(sequences, maxlen=max_len, padding="post")
