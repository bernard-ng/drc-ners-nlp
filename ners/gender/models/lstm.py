import argparse
import logging
import os
import pickle
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ProgbarLogger
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from misc import GENDER_MODELS_DIR, load_csv_dataset

logging.basicConfig(level=logging.INFO, format=">> %(message)s")


@dataclass
class Config:
    """
    Configuration for the machine learning model and its training process.

    This class encapsulates the configuration options necessary for initializing,
    training, and evaluating a machine learning model. It allows flexibility
    in specifying dataset details, model parameters, training settings, and
    options for evaluation. Attributes include paths, numerical parameters,
    and flags that guide the model's behavior.

    :ivar dataset_path: Path to the dataset file.
    :type dataset_path: str
    :ivar size: Optional size of the dataset to use. If None, use the full dataset.
    :type size: Optional[int]
    :ivar max_len: Maximum length of sequences used in the model.
    :type max_len: int
    :ivar embedding_dim: Dimensionality of the embedding layer.
    :type embedding_dim: int
    :ivar lstm_units: Number of LSTM units in the model.
    :type lstm_units: int
    :ivar batch_size: Batch size to use during training.
    :type batch_size: int
    :ivar epochs: Number of epochs for model training.
    :type epochs: int
    :ivar test_size: Fraction of data to use for testing.
    :type test_size: float
    :ivar random_state: Seed for random number generation to ensure reproducibility.
    :type random_state: int
    :ivar threshold: Decision threshold for binary classification tasks.
    :type threshold: float
    :ivar cv: Number of cross-validation folds. If None, no cross-validation is used.
    :type cv: Optional[int]
    :ivar save: Flag indicating whether to save the trained model.
    :type save: bool
    """
    dataset_path: str
    size: Optional[int] = None
    max_len: int = 6
    embedding_dim: int = 64
    lstm_units: int = 32
    batch_size: int = 64
    epochs: int = 10
    test_size: float = 0.2
    random_state: int = 42
    threshold: float = 0.5
    cv: Optional[int] = None
    save: bool = False


def load_and_prepare(cfg: Config) -> Tuple[np.ndarray, np.ndarray, Tokenizer, LabelEncoder]:
    """
    Load and preprocess the dataset based on the provided configuration.

    This function performs a series of operations including loading the dataset
    from the specified path, cleaning and preprocessing data (e.g., converting
    to lowercase, stripping whitespace, handling missing values), tokenizing names
    using a tokenizer, and encoding the labels using a label encoder. The final processed
    data and tools (tokenizer and label encoder) are returned for further use.

    :param cfg: Config object containing dataset parameters such as dataset path, size, and
        maximum sequence length.
    :type cfg: Config
    :return: A tuple containing processed padded sequences (numpy ndarray), corresponding
        encoded labels (numpy ndarray), tokenizer object used for preprocessing names,
        and label encoder object used for encoding labels.
    :rtype: Tuple[np.ndarray, np.ndarray, Tokenizer, LabelEncoder]
    """
    logging.info("Loading and preprocessing data")
    df = pd.DataFrame(load_csv_dataset(cfg.dataset_path, cfg.size)).dropna(subset=["name", "sex"])
    df["name"] = df["name"].str.lower().str.strip()
    df["sex"] = df["sex"].str.lower().str.strip()

    tokenizer = Tokenizer(char_level=False, lower=True, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["name"])
    sequences = tokenizer.texts_to_sequences(df["name"])
    padded = pad_sequences(sequences, maxlen=cfg.max_len, padding="post")

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df["sex"])

    return padded, labels, tokenizer, label_encoder


def build_model(cfg: Config, vocab_size: int) -> Sequential:
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


def evaluate_proba(y_true, y_proba, threshold, class_names):
    """
    Evaluate the performance of a binary classification model by calculating key metrics and printing
    a detailed classification report.

    This function thresholds the predicted probabilities to produce binary predictions and calculates
    metrics such as accuracy, precision, recall, and F1 score. It also generates a confusion matrix
    and a classification report for the model's performance. Additionally, metrics are logged and
    informational outputs are printed.

    :param y_true: Ground truth binary labels. Must be a 1-dimensional array or list of integers.
    :param y_proba: Predicted probabilities for each class from the model. It is a 2-dimensional array
        where the second dimension represents class probabilities for each sample.
    :param threshold: Threshold value for converting probabilities into binary predictions. Should be
        a float between 0 and 1.
    :param class_names: List of class names corresponding to the binary labels. Used for labeling the
        classification report.
    :return: None
    """
    y_pred = (y_proba[:, 1] >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    cm = confusion_matrix(y_true, y_pred)

    logging.info(f"Accuracy: {acc:.4f} | Precision: {pr:.4f} | Recall: {rc:.4f} | F1: {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))


def cross_validate(cfg: Config, X, y, vocab_size: int):
    """
    Performs k-fold cross-validation on a dataset using a specified model configuration.

    This function takes a dataset and corresponding labels, splits the dataset into
    k folds (based on the `cv` attribute of the provided configuration object), and
    performs cross-validation using the specified deep learning model. The model is
    built and trained on the training subset for each fold, and the validation subset
    is used to compute accuracy scores. Finally, it logs the individual fold accuracies
    and the overall mean accuracy with its standard deviation.

    :param cfg: Configuration object containing the parameters for cross-validation,
                model training, and other settings. `cv` specifies the number of folds,
                and other attributes such as `epochs`, `batch_size`, and `random_state`
                dictate the training and reproducibility behavior.
    :type cfg: Config
    :param X: Feature data for the dataset. Assumes the input is compatible with the
              model configuration.
    :param y: True labels corresponding to the dataset. The order should correspond
              to the feature set `X`.
    :param vocab_size: Total vocabulary size used for building the model. Determines
                       the structure of the model input.
    :type vocab_size: int
    :return: A list containing the accuracy scores for each fold.
    :rtype: List[float]
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
    Save the model, tokenizer, and label encoder artifacts to predefined file paths
    within the GENDER_MODELS_DIR directory. The function ensures that the model is
    saved in H5 format, while the tokenizer and encoder are serialized using the
    Pickle module. It logs a message indicating the completion of the saving process.

    :param model: The machine learning model object to be saved.
    :type model: Any

    :param tokenizer: The tokenizer object used in preprocessing, to be serialized
        for future use.
    :type tokenizer: Any

    :param encoder: The label encoder object used for encoding labels during
        training, to be serialized for future use.
    :type encoder: Any

    :return: None
    """
    model_path = os.path.join(GENDER_MODELS_DIR, "lstm_model.keras")
    tokenizer_path = os.path.join(GENDER_MODELS_DIR, "lstm_tokenizer.pkl")
    encoder_path = os.path.join(GENDER_MODELS_DIR, "lstm_label_encoder.pkl")

    model.save(model_path)
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    with open(encoder_path, "wb") as f:
        pickle.dump(encoder, f)
    logging.info(f"Model and artifacts saved to {GENDER_MODELS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Train BiLSTM model for name-based gender classification")
    parser.add_argument("--dataset", type=str, default="names.csv")
    parser.add_argument("--size", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cv", type=int)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    cfg = Config(
        dataset_path=args.dataset,
        size=args.size,
        threshold=args.threshold,
        cv=args.cv,
        save=args.save
    )

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
