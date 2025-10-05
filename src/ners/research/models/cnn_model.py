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
    SpatialDropout1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from ners.research.neural_network_model import NeuralNetworkModel


class CNNModel(NeuralNetworkModel):
    """1D Convolutional Neural Network for character patterns"""

    def build_model_with_vocab(self, vocab_size: int, **kwargs) -> Any:
        """Build CNN model with known vocabulary size"""

        params = kwargs
        model = Sequential(
            [
                # Learn char/subword embeddings; spatial dropout regularizes across channels
                # to make the model robust to noisy characters and transliteration.
                Embedding(
                    input_dim=vocab_size, output_dim=params.get("embedding_dim", 64)
                ),
                SpatialDropout1D(rate=params.get("embedding_dropout", 0.1)),
                # Small kernels capture short n-gram like patterns; padding='same' keeps
                # sequence length stable for simpler pooling behavior.
                Conv1D(
                    filters=params.get("filters", 64),
                    kernel_size=params.get("kernel_size", 3),
                    activation="relu",
                    padding="same",
                ),
                # Downsample to gain some position invariance and reduce computation.
                MaxPooling1D(pool_size=2),
                # Second conv layer to compose higher-level motifs (e.g., suffix+vowel).
                Conv1D(
                    filters=params.get("filters", 64),
                    kernel_size=params.get("kernel_size", 3),
                    activation="relu",
                    padding="same",
                ),
                # Global max pooling picks strongest motif evidence anywhere in the name.
                GlobalMaxPooling1D(),
                # Compact dense head with dropout to control overfitting.
                Dense(64, activation="relu"),
                Dropout(params.get("dropout", 0.5)),
                # Two-way softmax for binary classification.
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
        """Prepare sequences for CNN using extracted features"""
        # X here contains the features already extracted by FeatureExtractor
        # Get text data from extracted features - use character level for CNN
        text_data = self._collect_text_corpus(X)

        # Initialize character-level tokenizer
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(char_level=True, lower=True, oov_token="<OOV>")
            self.tokenizer.fit_on_texts(text_data)

        sequences = self.tokenizer.texts_to_sequences(text_data)
        max_len = self.config.model_params.get(
            "max_len", 20
        )  # Longer for character level

        # Right-side padding and truncation ensure contiguous non-zero tokens on the left
        return pad_sequences(
            sequences, maxlen=max_len, padding="post", truncating="post"
        )
