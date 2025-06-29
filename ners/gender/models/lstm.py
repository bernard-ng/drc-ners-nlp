import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ProgbarLogger
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from misc import GENDER_MODELS_DIR, load_csv_dataset, save_pickle
from ners.gender.models import load_config, BaseConfig, evaluate_proba, logging


@dataclass
class Config(BaseConfig):
    max_len: int = 6
    embedding_dim: int = 64
    lstm_units: int = 32
    batch_size: int = 64


def load_and_prepare(cfg: Config) -> Tuple[np.ndarray, np.ndarray, Tokenizer, LabelEncoder]:
    """
    Loads and preprocesses data for text classification by tokenizing text data, encoding labels, and padding sequences.
    This function expects a dataset file path, prepares the tokenizer to process text input, and encodes labels for
    model training. The resulting outputs are ready for input into a machine learning pipeline.
    """
    logging.info("Loading and preprocessing data")
    df = pd.DataFrame(load_csv_dataset(cfg.dataset_path, cfg.size, cfg.balanced))

    tokenizer = Tokenizer(char_level=False, lower=True, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["name"])
    sequences = tokenizer.texts_to_sequences(df["name"])
    padded = pad_sequences(sequences, maxlen=cfg.max_len, padding="post")

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df["sex"])

    return padded, labels, tokenizer, label_encoder


def build_model(cfg: Config, vocab_size: int) -> Sequential:
    """
    Builds and compiles a Sequential LSTM-based model. The model consists of an
    embedding layer, two bidirectional LSTM layers, a dense hidden layer with ReLU
    activation, and an output layer with a softmax activation function. The model
    is compiled using sparse categorical crossentropy loss and the Adam optimizer.
    """
    logging.info("Building LSTM model")
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=cfg.embedding_dim),
        Bidirectional(LSTM(cfg.lstm_units, return_sequences=True)),
        Bidirectional(LSTM(cfg.lstm_units)),
        Dense(64, activation="relu"),
        Dense(2, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def cross_validate(cfg: Config, X, y, vocab_size: int):
    """
    Performs cross-validation on the given dataset using the specified model configuration.
    The function uses StratifiedKFold cross-validator to split the dataset into training and
    validation sets for each fold. For each fold, it trains the model, evaluates its accuracy
    on the validation data, and logs the fold-wise and overall results.
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
    Saves the given model, tokenizer, and encoder artifacts to a predefined directory.

    The function ensures that the specified directory for saving artifacts exists,
    then serializes the model, tokenizer, and encoder using appropriate formats. It
    also logs the success of the operation to notify the user of the action taken.
    """
    os.makedirs(GENDER_MODELS_DIR, exist_ok=True)
    model.save(os.path.join(GENDER_MODELS_DIR, "lstm_model.keras"))

    save_pickle(tokenizer, os.path.join(GENDER_MODELS_DIR, "lstm_tokenizer.pkl"))
    save_pickle(encoder, os.path.join(GENDER_MODELS_DIR, "lstm_label_encoder.pkl"))

    logging.info(f"Model and artifacts saved to {GENDER_MODELS_DIR}")


def main():
    cfg = Config(**vars(load_config("Long Short-Term Memory (LSTM) model")))

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

    logging.info("Training model")
    model.fit(X_train, y_train,
              validation_split=0.1,
              epochs=cfg.epochs,
              batch_size=cfg.batch_size,
              callbacks=[ProgbarLogger()])

    y_proba = model.predict(X_test)
    evaluate_proba(y_test, y_proba, cfg.threshold, class_names=encoder.classes_)

    if cfg.save:
        save_artifacts(model, tokenizer, encoder)


if __name__ == "__main__":
    main()
