from typing import Any

import numpy as np
import pandas as pd
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
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
                # Mask padding tokens; required for LSTM to ignore padded timesteps.
                Embedding(
                    input_dim=vocab_size,
                    output_dim=params.get("embedding_dim", 64),
                    input_length=max_len,
                    mask_zero=True,
                ),
                # Stacked bidirectional LSTMs: first returns sequences to feed the next.
                # Dropout/recurrent_dropout mitigate overfitting on short sequences.
                Bidirectional(
                    LSTM(
                        params.get("lstm_units", 32),
                        return_sequences=True,
                        dropout=params.get("dropout", 0.2),
                        recurrent_dropout=params.get("recurrent_dropout", 0.0),
                    )
                ),
                # Second LSTM condenses sequence to a fixed vector for classification.
                Bidirectional(
                    LSTM(
                        params.get("lstm_units", 32),
                        dropout=params.get("dropout", 0.2),
                        recurrent_dropout=params.get("recurrent_dropout", 0.0),
                    )
                ),
                # Compact dense head with dropout; sufficient capacity for name signals.
                Dense(64, activation="relu"),
                Dropout(params.get("dropout", 0.5)),
                # Two-way softmax for binary classification.
                Dense(2, activation="softmax"),
            ]
        )

        model.compile(
            loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model

    def prepare_features(self, X: pd.DataFrame) -> np.ndarray:
        text_data = self._collect_text_corpus(X)

        # Initialize tokenizer if needed
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(char_level=False, lower=True, oov_token="<OOV>")
            self.tokenizer.fit_on_texts(text_data)

        # Convert to sequences
        sequences = self.tokenizer.texts_to_sequences(text_data)
        max_len = self.config.model_params.get("max_len", 6)

        return pad_sequences(sequences, maxlen=max_len, padding="post")
