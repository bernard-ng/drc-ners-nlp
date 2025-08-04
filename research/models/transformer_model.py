from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Dense,
    GlobalAveragePooling1D,
    MultiHeadAttention,
    Dropout,
    LayerNormalization,
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from research.neural_network_model import NeuralNetworkModel


class TransformerModel(NeuralNetworkModel):
    """Transformer-based model"""

    def build_model_with_vocab(self, vocab_size: int, max_len: int = 6, **kwargs) -> Any:
        params = kwargs

        # Build Transformer model
        inputs = Input(shape=(max_len,))
        x = Embedding(input_dim=vocab_size, output_dim=params.get("embedding_dim", 64))(inputs)

        # Add positional encoding
        positions = tf.range(start=0, limit=max_len, delta=1)
        pos_embedding = Embedding(input_dim=max_len, output_dim=params.get("embedding_dim", 64))(
            positions
        )
        x = x + pos_embedding

        x = self._transformer_encoder(x, params)
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation="relu")(x)
        outputs = Dense(2, activation="softmax")(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )
        return model

    @classmethod
    def _transformer_encoder(cls, x, cfg_params):
        """Transformer encoder block"""

        attn = MultiHeadAttention(
            num_heads=cfg_params.get("transformer_num_heads", 2),
            key_dim=cfg_params.get("transformer_head_size", 64),
        )(x, x)
        x = LayerNormalization(epsilon=1e-6)(x + Dropout(cfg_params.get("dropout", 0.1))(attn))

        ff = Dense(cfg_params.get("transformer_ff_dim", 128), activation="relu")(x)
        ff = Dense(x.shape[-1])(ff)
        return LayerNormalization(epsilon=1e-6)(x + Dropout(cfg_params.get("dropout", 0.1))(ff))

    def prepare_features(self, X: pd.DataFrame) -> np.ndarray:
        text_data = []
        for feature_type in self.config.features:
            if feature_type.value in X.columns:
                text_data.extend(X[feature_type.value].astype(str).tolist())

        if not text_data:
            raise ValueError("No text data found in the provided DataFrame.")

        # Initialize tokenizer if needed
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(oov_token="<OOV>")
            self.tokenizer.fit_on_texts(text_data)

        # Convert to sequences
        sequences = self.tokenizer.texts_to_sequences(text_data[: len(X)])
        max_len = self.config.model_params.get("max_len", 6)

        return pad_sequences(sequences, maxlen=max_len, padding="post")
