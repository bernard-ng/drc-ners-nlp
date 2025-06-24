import argparse
import logging
import os
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
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

logging.basicConfig(level=logging.INFO, format=">> %(message)s")


@dataclass
class Config:
    """
    Configuration data class used to store settings and parameters for a machine learning or deep
    learning model.

    This class allows the user to specify various parameters such as dataset path, size of input,
    model architecture details like embedding dimensions, transformer configurations, training settings
    like batch size and epochs, and validation and testing settings. The attributes provide flexibility
    to customize model configuration and training processes.

    :ivar dataset_path: The file path to the dataset.
    :type dataset_path: str
    :ivar size: Optional size parameter, can be used to specify sample size or custom
        configuration based on the user's requirement.
    :type size: Optional[int]
    :ivar max_len: Maximum sequence length for input data, used often in text or sequence
        processing.
    :type max_len: int
    :ivar embedding_dim: The dimensionality of embeddings used in the model.
    :type embedding_dim: int
    :ivar transformer_head_size: The size of each transformer attention head.
    :type transformer_head_size: int
    :ivar transformer_num_heads: The number of attention heads in the transformer model.
    :type transformer_num_heads: int
    :ivar transformer_ff_dim: The dimensionality of the feed-forward network in the transformer.
    :type transformer_ff_dim: int
    :ivar dropout: Dropout rate used for regularization during training.
    :type dropout: float
    :ivar batch_size: Batch size used for training and validation.
    :type batch_size: int
    :ivar epochs: Number of epochs for model training.
    :type epochs: int
    :ivar test_size: Proportion of the dataset to be used for testing.
    :type test_size: float
    :ivar random_state: Random seed value for reproducibility.
    :type random_state: int
    :ivar threshold: Threshold value for model predictions or classification.
    :type threshold: float
    :ivar cv: Cross-validation configuration, if applicable.
    :type cv: Optional[int]
    :ivar save: Boolean flag indicating whether to save the model after training.
    :type save: bool
    """
    dataset_path: str
    size: Optional[int]
    max_len: int = 6
    embedding_dim: int = 64
    transformer_head_size: int = 64
    transformer_num_heads: int = 2
    transformer_ff_dim: int = 128
    dropout: float = 0.1
    batch_size: int = 64
    epochs: int = 10
    test_size: float = 0.2
    random_state: int = 42
    threshold: float = 0.5
    cv: Optional[int] = None
    save: bool = False


def load_and_prepare(cfg: Config) -> Tuple[np.ndarray, np.ndarray, Tokenizer, LabelEncoder]:
    """
    Load and preprocess data for model training or evaluation. This function handles the
    loading of a dataset in CSV format, applies preprocessing to clean and normalize
    the input data, tokenizes text features, and encodes categorical labels.

    The preprocessed data is prepared as padded sequences and encoded labels, which
    can be directly used as inputs for machine learning models. Tokenizer and LabelEncoder
    are returned to ensure consistency between training and inference stages.

    :param cfg: Configuration object containing dataset path, size of the
                dataset to load, and maximum length for padding sequences.
    :type cfg: Config
    :return: A tuple containing padded input sequences for the model, encoded labels,
             the tokenizer used for text sequences, and the encoder used for labels.
    :rtype: Tuple[np.ndarray, np.ndarray, Tokenizer, LabelEncoder]
    """
    logging.info("Loading and preprocessing data")
    df = pd.DataFrame(load_csv_dataset(cfg.dataset_path, cfg.size)).dropna(subset=["name", "sex"])
    df["name"] = df["name"].str.lower().str.strip()
    df["sex"] = df["sex"].str.lower().str.strip()

    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(df["name"])
    sequences = tokenizer.texts_to_sequences(df["name"])
    padded = pad_sequences(sequences, maxlen=cfg.max_len, padding="post")

    encoder = LabelEncoder()
    labels = encoder.fit_transform(df["sex"])
    return padded, labels, tokenizer, encoder


def transformer_encoder(x, cfg: Config):
    """
    Transforms input tensor using a single Transformer encoder block with attention and feedforward
    layers. The encoder applies multi-head attention to the input tensor, adds the output to
    the original tensor for residual connection, and normalizes it. Subsequently, the processed
    tensor passes through a feedforward network with added dropout and normalization.

    :param x: Input tensor to be transformed.
    :type x: TensorFlow tensor
    :param cfg: Configuration object containing Transformer hyperparameters such as the number of
        attention heads, head size, feedforward dimension, and dropout rate.
    :type cfg: Config
    :return: Transformed tensor resulting from applying the Transformer encoder block.
    :rtype: TensorFlow tensor
    """
    attn = MultiHeadAttention(num_heads=cfg.transformer_num_heads, key_dim=cfg.transformer_head_size)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + Dropout(cfg.dropout)(attn))

    ff = Dense(cfg.transformer_ff_dim, activation="relu")(x)
    ff = Dense(x.shape[-1])(ff)
    return LayerNormalization(epsilon=1e-6)(x + Dropout(cfg.dropout)(ff))


def build_model(cfg: Config, vocab_size: int) -> Model:
    """
    Builds a Transformer-based model using Keras/TensorFlow components. The model
    is designed for classification tasks, utilizing embedding layers with positional
    encoding, a Transformer encoder block, and fully connected layers for
    output generation.

    :param cfg: Configuration object containing model-specific hyperparameters
        such as maximum sequence length, embedding dimensions, etc.
    :type cfg: Config
    :param vocab_size: The size of the vocabulary for the embedding layer.
    :type vocab_size: int
    :return: A compiled Keras model, ready for training and evaluation.
    :rtype: Model
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


def evaluate_proba(y_true, y_proba, threshold, class_names):
    """
    Evaluates the performance of a binary classification model by calculating accuracy,
    precision, recall, F1 score, confusion matrix, and generates a classification
    report. This function takes the true labels, predicted probabilities, a decision
    threshold, and class names to assist in the evaluation.

    :param y_true: Ground truth (correct) target values.
    :type y_true: array-like of shape (n_samples,)
    :param y_proba: Predicted probabilities for each class. Expected to be an array
        where the second column corresponds to the probability of the positive class.
    :type y_proba: array-like of shape (n_samples, 2)
    :param threshold: Decision threshold for classifying a sample as positive
        or negative based on predicted probabilities.
    :type threshold: float
    :param class_names: List of class names for labeling the classification report.
    :type class_names: list of str
    :return: None. Outputs performance metrics and confusion matrix to the logging
        system and the console.
    """
    y_pred = (y_proba[:, 1] >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    cm = confusion_matrix(y_true, y_pred)

    logging.info(f"Accuracy: {acc:.4f} | Precision: {pr:.4f} | Recall: {rc:.4f} | F1: {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))


def cross_validate(cfg: Config, X, y, vocab_size: int):
    """
    Evaluate the performance of a model using K-fold cross-validation. This function takes
    configuration settings, input data, target labels, and vocabulary size to perform the
    specified number of cross-validation folds with a stratified approach. For each fold,
    it builds a new model, trains it, predicts the validation set, and calculates accuracy.

    :param cfg: The configuration object containing hyperparameters and settings for
                cross-validation, random state, and training.
    :type cfg: Config
    :param X: The input data samples provided as a dataset.
    :type X: numpy.ndarray
    :param y: The target labels corresponding to the input data samples.
    :type y: numpy.ndarray
    :param vocab_size: The size of the vocabulary, used to configure the language model.
    :type vocab_size: int
    :return: A list containing accuracy scores from each fold in the cross-validation process.
    :rtype: list
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
    Saves the machine learning model and its associated artifacts such as tokenizer and
    label encoder to predefined file paths. This function ensures that the model and
    artifacts can be reloaded later for inference or further use.

    :param model: The machine learning model to be saved.
    :param tokenizer: The tokenizer used for preparing data for the model.
    :param encoder: The label encoder used for encoding target labels.
    :return: None
    """
    os.makedirs(GENDER_MODELS_DIR, exist_ok=True)
    model.save(os.path.join(GENDER_MODELS_DIR, "transformer.keras"))

    save_pickle(tokenizer, os.path.join(GENDER_MODELS_DIR, "transformer_tokenizer.pkl"))
    save_pickle(encoder, os.path.join(GENDER_MODELS_DIR, "transformer_label_encoder.pkl"))

    logging.info(f"Model and artifacts saved to {GENDER_MODELS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Train Transformer model for name-based gender classification")
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
