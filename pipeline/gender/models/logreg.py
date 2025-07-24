import os
from dataclasses import dataclass
from typing import Tuple

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
from pipeline.gender.models import BaseConfig, load_config, logging


@dataclass
class Config(BaseConfig):
    ngram_range: Tuple[int, int] = (2, 5)
    max_iter: int = 1000


def encode_labels(y: pd.Series) -> Tuple[pd.Series, LabelEncoder]:
    """
    Encode the labels using a LabelEncoder. This function takes a pandas Series of labels,
    fits a LabelEncoder to the labels, and transforms them into a numerical format suitable
    for model training. The transformed labels and the fitted encoder are returned.
    """
    logging.info("Encoding labels")
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder


def build_model(cfg: Config) -> Pipeline:
    """
    Build a logistic regression model pipeline with a character-level CountVectorizer.
    The pipeline consists of a CountVectorizer that transforms the input text into
    character n-grams, followed by a Logistic Regression classifier. The n-gram range
    and maximum iterations for the logistic regression can be configured through the
    provided configuration object.
    """
    return make_pipeline(
        CountVectorizer(analyzer="char", ngram_range=cfg.ngram_range),
        LogisticRegression(max_iter=cfg.max_iter)
    )


def evaluate_proba(y_true, y_proba, threshold: float, class_names):
    """
    Evaluates the performance of a classification model using a specified threshold
    for predicted probabilities. Computes metrics such as accuracy, precision,
    recall, F1-score, and the confusion matrix. Also generates a classification
    report with detailed metrics for each class.

    Logs the evaluation metrics at the specified threshold and prints the confusion
    matrix and classification report.
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
    """
    logging.info(f"Running {cfg.cv}-fold cross-validation")
    pipeline = build_model(cfg)
    scores = cross_val_score(pipeline, X, y, cv=StratifiedKFold(n_splits=cfg.cv), scoring="accuracy")
    logging.info(f"Cross-validation scores: {scores}")
    logging.info(f"Mean accuracy: {scores.mean():.4f}, Std: {scores.std():.4f}")


def save_artifacts(model, encoder):
    """
    Saves the trained model and label encoder artifacts to the specified directory.
    """
    save_pickle(model, os.path.join(GENDER_MODELS_DIR, "regression_model.pkl"))
    save_pickle(encoder, os.path.join(GENDER_MODELS_DIR, "regression_label_encoder.pkl"))

    logging.info(f"Model and artifacts saved to {GENDER_MODELS_DIR}")


def main():
    cfg = Config(**vars(load_config("logistic regression model")))

    df = pd.DataFrame(load_csv_dataset(cfg.dataset_path, cfg.size, cfg.balanced))
    X_raw, y_raw = df["name"], df["sex"]
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
    evaluate_proba(y_test, y_proba, cfg.threshold, class_names=encoder.classes_)

    if cfg.save:
        save_artifacts(model, encoder)


if __name__ == "__main__":
    main()
