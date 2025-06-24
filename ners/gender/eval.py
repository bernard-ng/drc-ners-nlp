import argparse
import json
import os

import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)
from tensorflow.keras.preprocessing.sequence import pad_sequences

from misc import GENDER_MODELS_DIR, load_csv_dataset, save_json_dataset, load_pickle, GENDER_RESULT_DIR


def load_dataset(path="names.csv", size=None):
    """
    Loads a dataset from a CSV file, processes it to remove missing values
    and standardizes the case and formatting of specific columns.

    :param path: The path to the CSV file containing the dataset. Defaults to "names.csv".
    :type path: str
    :param size: The number of rows to load from the dataset. If None, the whole dataset is loaded.
    :type size: Optional[int]
    :return: A pandas DataFrame with the processed dataset where missing values in the
             'name' and 'sex' columns are removed, and the text in these columns is
             converted to lowercase and stripped of leading/trailing whitespace.
    :rtype: pandas.DataFrame
    """
    df = pd.DataFrame(load_csv_dataset(path, size)).dropna(subset=["name", "sex"])
    df["name"] = df["name"].str.lower().str.strip()
    df["sex"] = df["sex"].str.lower().str.strip()
    return df


def evaluate_logreg(df, threshold):
    """
    Evaluates a logistic regression model with the given DataFrame and threshold. The function loads
    a pre-trained model and label encoder, transforms the input data into the required format, and
    performs predictions. It returns the true labels, predicted labels, predicted probabilities, and
    the encoder class labels.

    :param df: Input data containing a column "name" for names to evaluate and a column "sex"
        for true labels.
        Type: pandas.DataFrame

    :param threshold: Threshold value used for classifying the predictions. Probabilities greater
        than or equal to this value are classified into the positive class.
        Type: float

    :return: A tuple containing:
        - y_true: True labels after encoding.
        - y_pred: Predicted binary class labels based on the threshold.
        - proba[:, 1]: Probability values for the positive class.
        - encoder.classes_: Labels used by the label encoder.
        Type: tuple (numpy.ndarray, int, numpy.ndarray, numpy.ndarray)
    """
    model = load_pickle(os.path.join(GENDER_MODELS_DIR, "regression_model.pkl"))
    encoder = load_pickle(os.path.join(GENDER_MODELS_DIR, "regression_label_encoder.pkl"))

    X = df["name"].tolist()
    y_true = encoder.transform(df["sex"])
    proba = model.predict_proba(X)
    y_pred = 1 if proba[:, 1] >= threshold else 0
    return y_true, y_pred, proba[:, 1], encoder.classes_


def evaluate_lstm(df, threshold, max_len=6):
    """
    Evaluates the predictions of a pre-trained BiLSTM model on the given dataset and
    returns the true labels, predicted labels, prediction probabilities, and class names.

    :param df: Input DataFrame containing the data for evaluation.
               The DataFrame must have two columns: "name" containing
               the input text data and "sex" containing the true labels.
    :type df: Pandas.DataFrame
    :param threshold: Decision threshold for determining binary classification
                      outcome based on model's prediction probabilities.
    :type threshold: Float
    :param max_len: The maximum length of input sequences. Used to pad or truncate
                    tokenized sequences. Default value is 6.
    :type max_len: Int
    :return: A tuple containing the following elements:
             - y_true: The true labels from the input DataFrame.
             - y_pred: The predicted binary labels according to the decision threshold.
             - proba: Prediction probabilities for the positive class, as output by the model.
             - encoder.classes_: An array of class names corresponding to the label encoding.
    :rtype: Tuple
    """
    model = tf.keras.models.load_model(os.path.join(GENDER_MODELS_DIR, "lstm_model.keras"))
    tokenizer = load_pickle(os.path.join(GENDER_MODELS_DIR, "lstm_tokenizer.pkl"))
    encoder = load_pickle(os.path.join(GENDER_MODELS_DIR, "lstm_label_encoder.pkl"))

    sequences = tokenizer.texts_to_sequences(df["name"])
    X = pad_sequences(sequences, maxlen=max_len, padding="post")
    y_true = encoder.transform(df["sex"])
    proba = model.predict(X)
    y_pred = 1 if proba[:, 1] >= threshold else 0
    return y_true, y_pred, proba[:, 1], encoder.classes_


def evaluate_transformer(df, threshold, max_len=6):
    """
    Evaluates the transformer model for gender prediction. The function loads a pre-trained
    transformer model, tokenizer, and label encoder. It processes the input dataframe by
    tokenizing and padding the "name" column and encodes the "sex" column to numerical format.
    The function then predicts the probabilities for the given names using the transformer model
    and generates predictions based on the specified threshold.

    :param df: Pandas DataFrame containing a "name" column with strings to be evaluated
        and a "sex" column with corresponding target labels.
    :type df: Pd.DataFrame
    :param threshold: Threshold value used to determine binary classification labels
        from predicted probabilities.
    :type threshold: Float
    :param max_len: Maximum length for padded sequences, default is 6.
    :type max_len: Int, optional
    :return: A tuple containing the ground truth labels, predicted labels, predicted
        probabilities for the positive class, and a list of the label classes.
    :rtype: Tuple
    """
    model = tf.keras.models.load_model(os.path.join(GENDER_MODELS_DIR, "transformer.keras"))
    tokenizer = load_pickle(os.path.join(GENDER_MODELS_DIR, "transformer_tokenizer.pkl"))
    encoder = load_pickle(os.path.join(GENDER_MODELS_DIR, "transformer_label_encoder.pkl"))

    sequences = tokenizer.texts_to_sequences(df["name"])
    X = pad_sequences(sequences, maxlen=max_len, padding="post")
    y_true = encoder.transform(df["sex"])
    proba = model.predict(X)
    y_pred = 1 if proba[:, 1] >= threshold else 0
    return y_true, y_pred, proba[:, 1], encoder.classes_


def compute_metrics(y_true, y_pred, y_proba, class_names):
    """
    Computes classification metrics for given true and predicted labels, along with
    class probabilities and class names. The function calculates accuracy, precision,
    recall, F1 score, and confusion matrix for evaluating model performance.

    :param y_true: Ground truth (correct) labels.
    :type y_true: list or numpy.ndarray
    :param y_pred: Predicted labels, as returned by a classifier.
    :type y_pred: list or numpy.ndarray
    :param y_proba: Predicted probabilities for positive class.
    :type y_proba: list or numpy.ndarray
    :param class_names: Names of the classes corresponding to labels in the confusion
        matrix.
    :type class_names: numpy.ndarray
    :return: A dictionary containing computed accuracy, precision, recall, F1 score,
        and confusion matrix with labels and matrix elements.
    :rtype: dict
    """
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": acc,
        "precision": pr,
        "recall": rc,
        "f1": f1,
        "confusion_matrix": {
            "labels": class_names.tolist(),
            "matrix": cm
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate gender prediction model")
    parser.add_argument("--model", choices=["logreg", "lstm", "transformer"], required=True)
    parser.add_argument("--dataset", default="names.csv")
    parser.add_argument("--size", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    df = load_dataset(args.dataset, args.size)

    if args.model == "logreg":
        y_true, y_pred, y_proba, classes = evaluate_logreg(df, args.threshold)
    elif args.model == "lstm":
        y_true, y_pred, y_proba, classes = evaluate_lstm(df, args.threshold)
    elif args.model == "transformer":
        y_true, y_pred, y_proba, classes = evaluate_transformer(df, args.threshold)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    results = compute_metrics(y_true, y_pred, y_proba, classes)
    save_json_dataset(results, os.path.join(GENDER_RESULT_DIR, f'{args.model}_eval'))


if __name__ == "__main__":
    main()
