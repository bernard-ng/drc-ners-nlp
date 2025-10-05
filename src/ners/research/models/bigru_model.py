from typing import Any

import numpy as np
import pandas as pd
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from ners.research.neural_network_model import NeuralNetworkModel


class BiGRUModel(NeuralNetworkModel):
    """Bidirectional GRU model for name classification"""

    def build_model_with_vocab(self, vocab_size: int, **kwargs) -> Any:
        params = kwargs
        model = Sequential(
            [
                # Mask padding tokens so recurrent layers ignore them; fix input length
                # for better shape inference and to support masking through the stack.
                Embedding(
                    input_dim=vocab_size,
                    output_dim=params.get("embedding_dim", 64),
                    mask_zero=True,
                    input_length=params.get("max_len", 6),
                ),
                # First recurrent block returns full sequences to allow stacking.
                # Moderate dropout + optional recurrent_dropout to reduce overfitting
                # on short names while retaining temporal signal.
                Bidirectional(
                    GRU(
                        params.get("gru_units", 32),
                        return_sequences=True,
                        dropout=params.get("dropout", 0.2),
                        recurrent_dropout=params.get("recurrent_dropout", 0.0),
                    )
                ),
                # Second GRU summarizes to the last hidden state (no return_sequences),
                # capturing bidirectional context efficiently for classification.
                Bidirectional(
                    GRU(
                        params.get("gru_units", 32),
                        dropout=params.get("dropout", 0.2),
                        recurrent_dropout=params.get("recurrent_dropout", 0.0),
                    )
                ),
                # Small dense head; ReLU + dropout for capacity and regularization.
                Dense(64, activation="relu"),
                Dropout(params.get("dropout", 0.5)),
                # Two-way softmax for binary gender classification.
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
        text_data = self._collect_text_corpus(X)

        if self.tokenizer is None:
            self.tokenizer = Tokenizer(char_level=False, lower=True, oov_token="<OOV>")
            self.tokenizer.fit_on_texts(text_data)

        sequences = self.tokenizer.texts_to_sequences(text_data)
        max_len = self.config.model_params.get("max_len", 6)

        # Ensure padding and truncation are applied on the right to keep
        # contiguous non-zero tokens on the left, matching RNN mask expectations.
        return pad_sequences(
            sequences, maxlen=max_len, padding="post", truncating="post"
        )
