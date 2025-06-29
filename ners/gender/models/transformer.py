import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ProgbarLogger
from tensorflow.keras.layers import (
    Input, Embedding, Dense, GlobalAveragePooling1D,
    MultiHeadAttention, Dropout, LayerNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from misc import GENDER_MODELS_DIR, load_csv_dataset, save_pickle
from ners.gender.models import BaseConfig, load_config, evaluate_proba, logging


@dataclass
class Config(BaseConfig):
    max_len: int = 6
    embedding_dim: int = 64
    transformer_head_size: int = 64
    transformer_num_heads: int = 2
    transformer_ff_dim: int = 128
    dropout: float = 0.1
    batch_size: int = 64


def load_and_prepare(cfg: Config) -> Tuple[np.ndarray, np.ndarray, Tokenizer, LabelEncoder]:
    """
    Load and preprocess the dataset for training a Transformer model.
    This function reads a CSV dataset, tokenizes the names, pads the sequences,
    and encodes the labels. It returns the padded sequences, encoded labels,
    tokenizer, and label encoder.
    """
    logging.info("Loading and preprocessing data")
    df = pd.DataFrame(load_csv_dataset(cfg.dataset_path, cfg.size, cfg.balanced))

    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(df["name"])

    sequences = tokenizer.texts_to_sequences(df["name"])
    padded = pad_sequences(sequences, maxlen=cfg.max_len, padding="post")

    encoder = LabelEncoder()
    labels = encoder.fit_transform(df["sex"])
    return padded, labels, tokenizer, encoder


def transformer_encoder(x, cfg: Config):
    """
    Transformer encoder block that applies multi-head attention and feed-forward
    neural network layers with residual connections and layer normalization.
    """
    attn = MultiHeadAttention(num_heads=cfg.transformer_num_heads, key_dim=cfg.transformer_head_size)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + Dropout(cfg.dropout)(attn))

    ff = Dense(cfg.transformer_ff_dim, activation="relu")(x)
    ff = Dense(x.shape[-1])(ff)
    return LayerNormalization(epsilon=1e-6)(x + Dropout(cfg.dropout)(ff))


def build_model(cfg: Config, vocab_size: int) -> Model:
    """
    Builds a Transformer-based model aimed at sequence processing tasks.
    The model includes an embedding layer integrating positional encodings
    and a Transformer encoder, followed by a global pooling layer,
    a dense hidden layer, and a softmax output layer.
    """
    logging.info("Building Transformer model")
    inputs = Input(shape=(cfg.max_len,))
    x = Embedding(input_dim=vocab_size, output_dim=cfg.embedding_dim)(inputs)

    # Add positional encoding
    positions = tf.range(start=0, limit=cfg.max_len, delta=1)
    pos_embedding = Embedding(input_dim=cfg.max_len, output_dim=cfg.embedding_dim)(positions)
    x = x + pos_embedding

    x = transformer_encoder(x, cfg)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(2, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def cross_validate(cfg: Config, X, y, vocab_size: int):
    """
    Performs cross-validation using the given configuration, dataset, and specified vocabulary size. This function
    splits the dataset into stratified folds, trains a model on each fold, and evaluates its performance on validation
    data. The overall mean and standard deviation of accuracies across all folds are logged.
    """
    logging.info(f"Running {cfg.cv}-fold cross-validation")
    skf = StratifiedKFold(n_splits=cfg.cv, shuffle=True, random_state=cfg.random_state)
    accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logging.info(f"Fold {fold + 1}")
        model = build_model(cfg, vocab_size)
        model.fit(X[train_idx], y[train_idx],
                  epochs=cfg.epochs,
                  batch_size=cfg.batch_size,
                  verbose=0)
        y_pred = model.predict(X[val_idx])
        acc = accuracy_score(y[val_idx], y_pred.argmax(axis=1))
        accuracies.append(acc)
        logging.info(f"Fold {fold + 1} Accuracy: {acc:.4f}")

    logging.info(f"Mean accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")


def save_artifacts(model, tokenizer, encoder):
    """
    Saves the model and associated artifacts to the designated directory. The model
    is serialized and saved in a `.keras` file, while the tokenizer and label
    encoder are serialized into `.pkl` files. If the directory does not exist, it
    is created automatically. This function also logs the completion of the
    operation.
    """
    os.makedirs(GENDER_MODELS_DIR, exist_ok=True)
    model.save(os.path.join(GENDER_MODELS_DIR, "transformer.keras"))

    save_pickle(tokenizer, os.path.join(GENDER_MODELS_DIR, "transformer_tokenizer.pkl"))
    save_pickle(encoder, os.path.join(GENDER_MODELS_DIR, "transformer_label_encoder.pkl"))

    logging.info(f"Model and artifacts saved to {GENDER_MODELS_DIR}")


def main():
    cfg = Config(**vars(load_config("Transformer model")))

    X, y, tokenizer, encoder = load_and_prepare(cfg)
    vocab_size = len(tokenizer.word_index) + 1

    if cfg.cv:
        cross_validate(cfg, X, y, vocab_size)
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    model = build_model(cfg, vocab_size)
    model.summary()

    logging.info("Training Transformer model")
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=[ProgbarLogger()]
    )

    y_proba = model.predict(X_test)
    evaluate_proba(y_test, y_proba, cfg.threshold, class_names=encoder.classes_)

    if cfg.save:
        save_artifacts(model, tokenizer, encoder)


if __name__ == "__main__":
    main()
