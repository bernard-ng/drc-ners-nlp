import argparse
import logging
import os
from dataclasses import dataclass
from typing import Tuple, Optional

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import LabelEncoder

from misc import GENDER_MODELS_DIR, load_csv_dataset, save_pickle

logging.basicConfig(level=logging.INFO, format=">> %(message)s")


@dataclass
class Config:
    dataset_path: str
    size: Optional[int]
    test_size: float = 0.2
    ngram_range: Tuple[int, int] = (2, 5)
    max_iter: int = 1000
    random_state: int = 42
    threshold: float = 0.5
    cv: Optional[int] = None
    save: bool = False


def load_and_clean_data(cfg: Config) -> Tuple[pd.Series, pd.Series]:
    """
    Load and clean dataset as specified by the provided configuration. This function reads
    a CSV dataset from the path specified in the configuration, processes it to remove
    missing values from key columns ('name' and 'sex'), and cleans string data in these
    columns by converting them to lowercase and stripping whitespace. The cleaned data
    is then returned as two separate pandas Series objects.

    :param cfg: Configuration object specifying the dataset path and size
    :type cfg: Config
    :return: A tuple containing cleaned `name` and `sex` data as pandas Series objects
    :rtype: Tuple[pd.Series, pd.Series]
    """
    logging.info(f"Loading dataset from {cfg.dataset_path}")
    df = pd.DataFrame(load_csv_dataset(cfg.dataset_path, cfg.size))
    df = df.dropna(subset=["name", "sex"])
    df["name"] = df["name"].str.lower().str.strip()
    df["sex"] = df["sex"].str.lower().str.strip()
    return df["name"], df["sex"]


def encode_labels(y: pd.Series) -> Tuple[pd.Series, LabelEncoder]:
    """
    Encode the labels of a given pandas Series using a LabelEncoder. This process maps categorical
    labels to integers, which is particularly useful for machine learning models that require numerical
    input data.

    :param y: A pandas Series of categorical labels to be encoded.
    :type y: pd.Series
    :return: A tuple containing the encoded labels as a pandas Series and the fitted LabelEncoder object.
    :rtype: Tuple[pd.Series, LabelEncoder]
    """
    logging.info("Encoding labels")
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder


def build_model(cfg: Config) -> Pipeline:
    """
    Builds a machine learning pipeline for text classification.

    This function constructs and returns a scikit-learn pipeline that consists of
    a `CountVectorizer` and a `LogisticRegression` classifier. The vectorizer
    leverages character-level n-grams based on the provided configuration, and the
    logistic regression model is trained with a maximum number of iterations defined
    in the configuration. This pipeline is used for processing text data and training
    classification models.

    :param cfg: Configuration object containing the n-gram range and the maximum
                number of iterations for the logistic regression model.
    :type cfg: Config
    :return: A scikit-learn pipeline with a `CountVectorizer` and `LogisticRegression`
             based on the provided configuration.
    :rtype: Pipeline
    """
    return make_pipeline(
        CountVectorizer(analyzer="char", ngram_range=cfg.ngram_range),
        LogisticRegression(max_iter=cfg.max_iter)
    )


def evaluate_probabilities(y_true, y_proba, threshold: float, class_names):
    """
    Evaluates the performance of a classification model using a specified threshold
    for predicted probabilities. Computes metrics such as accuracy, precision,
    recall, F1-score, and the confusion matrix. Also generates a classification
    report with detailed metrics for each class.

    Logs the evaluation metrics at the specified threshold and prints the confusion
    matrix and classification report.

    :param y_true: Ground truth (correct) labels.
    :type y_true: array-like
    :param y_proba: Predicted probabilities for each class, where each row
        corresponds to an instance and contains probabilities for each target class.
    :type y_proba: numpy.ndarray
    :param threshold: The threshold on predicted probabilities to determine
        class membership for each instance.
    :type threshold: float
    :param class_names: List of class names for the target variable used in the
        classification report.
    :type class_names: list of str
    :return: None
    """
    logging.info(f"Evaluating at threshold = {threshold}")
    y_pred = (y_proba[:, 1] >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    cm = confusion_matrix(y_true, y_pred)

    logging.info(f"Accuracy: {acc:.4f}")
    logging.info(f"Precision: {pr:.4f}, Recall: {rc:.4f}, F1-score: {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))


def cross_validate(cfg: Config, X, y) -> None:
    """
    Performs k-fold cross-validation on the provided dataset using the configuration and
    logs the results including individual fold scores, mean accuracy, and the standard
    deviation of the scores.

    :param cfg: Configuration object containing cross-validation settings such as the
        number of folds to use in the cross-validation (`cv`).
    :type cfg: Config
    :param X: Input feature matrix for the dataset to be used for cross-validation.
    :type X: Any
    :param y: Target labels corresponding to the input feature matrix `X`.
    :type y: Any
    :return: This function does not return any value. Results are logged.
    :rtype: None
    """
    logging.info(f"Running {cfg.cv}-fold cross-validation")
    pipeline = build_model(cfg)
    scores = cross_val_score(pipeline, X, y, cv=StratifiedKFold(n_splits=cfg.cv), scoring="accuracy")
    logging.info(f"Cross-validation scores: {scores}")
    logging.info(f"Mean accuracy: {scores.mean():.4f}, Std: {scores.std():.4f}")


def save_artifacts(model, encoder, cfg: Config):
    """
    Saves machine learning model and label encoder artifacts to specified directories
    within the gender models' directory. This function ensures that the model and encoder
    are serialized and stored as pickle files. It uses the specified configuration settings
    to locate the appropriate directory for storing the files.

    :param model: The machine learning model object to be saved.
    :type model: Any
    :param encoder: The label encoder object used for data preprocessing.
    :type encoder: Any
    :param cfg: Configuration object containing application-specific settings regarding
        paths and directories.
    :type cfg: Config
    :return: None
    """
    save_pickle(model, os.path.join(GENDER_MODELS_DIR, "regression_model.pkl"))
    save_pickle(encoder, os.path.join(GENDER_MODELS_DIR, "regression_label_encoder.pkl"))

    logging.info(f"Model and artifacts saved to {GENDER_MODELS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Train a gender classifier on names")
    parser.add_argument("--dataset", type=str, default="names.csv", help="Path to dataset")
    parser.add_argument("--size", type=int, help="Number of rows to load")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for binary decision")
    parser.add_argument("--cv", type=int, help="Number of folds for cross-validation")
    parser.add_argument("--save", action="store_true", help="Save the model and encoder")
    args = parser.parse_args()

    cfg = Config(
        dataset_path=args.dataset,
        size=args.size,
        threshold=args.threshold,
        cv=args.cv,
        save=args.save
    )

    X_raw, y_raw = load_and_clean_data(cfg)
    y_encoded, encoder = encode_labels(y_raw)

    if cfg.cv:
        cross_validate(cfg, X_raw, y_encoded)
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_encoded, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y_encoded
    )

    model = build_model(cfg)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)
    evaluate_probabilities(y_test, y_proba, cfg.threshold, class_names=encoder.classes_)

    if cfg.save:
        save_artifacts(model, encoder, cfg)


if __name__ == "__main__":
    main()
