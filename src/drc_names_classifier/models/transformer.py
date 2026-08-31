# pyright: reportOptionalMemberAccess=false

from typing import Any

import numpy as np
import polars as pl
import tensorflow as tf
from keras import ops
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

from drc_names_classifier.models.neural import NeuralNetworkModel


class TransformerModel(NeuralNetworkModel):
    """Transformer encoder full-name classifier."""

    def build_model(self, vocab_size: int, **kwargs) -> Any:
        params = kwargs
        max_len = int(params.get("max_len", 32))

        inputs = Input(shape=(max_len,))
        padding_mask = ops.not_equal(inputs, 0)
        attention_mask = ops.expand_dims(padding_mask, axis=1)
        x = Embedding(
            input_dim=vocab_size,
            output_dim=params.get("embedding_dim", 64),
            mask_zero=True,
        )(inputs)

        positions = tf.range(start=0, limit=max_len, delta=1)
        pos_embedding = Embedding(
            input_dim=max_len,
            output_dim=params.get("embedding_dim", 64),
        )(positions)
        x = x + pos_embedding

        x = self._transformer_encoder(x, params, attention_mask)
        x = GlobalAveragePooling1D()(x, mask=padding_mask)
        x = Dense(32, activation="relu")(x)
        x = Dropout(params.get("dropout", 0.1))(x)
        outputs = Dense(2, activation="softmax", dtype="float32")(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    @classmethod
    def _transformer_encoder(cls, x, cfg_params, attention_mask):
        """Apply one attention and feed-forward block."""

        attn = MultiHeadAttention(
            num_heads=cfg_params.get("transformer_num_heads", 2),
            key_dim=cfg_params.get("transformer_head_size", 64),
            dropout=cfg_params.get("attn_dropout", 0.1),
        )(x, x, attention_mask=attention_mask)
        x = LayerNormalization(epsilon=1e-6)(x + Dropout(cfg_params.get("dropout", 0.1))(attn))

        ff = Dense(cfg_params.get("transformer_ff_dim", 128), activation="relu")(x)
        ff = Dense(x.shape[-1])(ff)
        return LayerNormalization(epsilon=1e-6)(x + Dropout(cfg_params.get("dropout", 0.1))(ff))

    def prepare_features(self, X: pl.DataFrame) -> np.ndarray:
        text_data = self._collect_text_corpus(X)

        if self.tokenizer is None:
            self.tokenizer = Tokenizer(char_level=True, lower=True, oov_token="<OOV>")
            self.tokenizer.fit_on_texts(text_data)

        sequences = self.tokenizer.texts_to_sequences(text_data)
        max_len = int(self.config.model_params.get("max_len", 32))

        return pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
