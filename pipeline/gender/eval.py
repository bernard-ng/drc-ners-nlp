import argparse
import os

import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)
from tensorflow.keras.preprocessing.sequence import pad_sequences

from misc import GENDER_MODELS_DIR, load_csv_dataset, save_json_dataset, load_pickle, GENDER_RESULT_DIR


def evaluate_logreg(df, threshold):
    """
    Evaluates a logistic regression model with the given DataFrame and threshold. The function loads
    a pre-trained model and label encoder, transforms the input data into the required format, and
    performs predictions. It returns the true labels, predicted labels, predicted probabilities, and
    the encoder class labels.
    """
    model = load_pickle(os.path.join(GENDER_MODELS_DIR, "regression_model.pkl"))
    encoder = load_pickle(os.path.join(GENDER_MODELS_DIR, "regression_label_encoder.pkl"))

    X = df["name"].tolist()
    y_true = encoder.transform(df["sex"])
    proba = model.predict_proba(X)
    y_pred = (proba[:, 1] >= threshold).astype(int)
    return y_true, y_pred, proba[:, 1], encoder.classes_


def evaluate_lstm(df, threshold, max_len=6):
    """
    Evaluates the predictions of a pre-trained BiLSTM model on the given dataset and
    returns the true labels, predicted labels, prediction probabilities, and class names.
    """
    model = tf.keras.models.load_model(os.path.join(GENDER_MODELS_DIR, "lstm_model.keras"))
    tokenizer = load_pickle(os.path.join(GENDER_MODELS_DIR, "lstm_tokenizer.pkl"))
    encoder = load_pickle(os.path.join(GENDER_MODELS_DIR, "lstm_label_encoder.pkl"))

    sequences = tokenizer.texts_to_sequences(df["name"])
    X = pad_sequences(sequences, maxlen=max_len, padding="post")
    y_true = encoder.transform(df["sex"])
    proba = model.predict(X)
    y_pred = (proba[:, 1] >= threshold).astype(int)
    return y_true, y_pred, proba[:, 1], encoder.classes_


def evaluate_transformer(df, threshold, max_len=6):
    """
    Evaluates the transformer model for gender prediction. The function loads a pre-trained
    transformer model, tokenizer, and label encoder. It processes the input dataframe by
    tokenizing and padding the "name" column and encodes the "sex" column to numerical format.
    The function then predicts the probabilities for the given names using the transformer model
    and generates predictions based on the specified threshold.
    """
    model = tf.keras.models.load_model(os.path.join(GENDER_MODELS_DIR, "transformer.keras"))
    tokenizer = load_pickle(os.path.join(GENDER_MODELS_DIR, "transformer_tokenizer.pkl"))
    encoder = load_pickle(os.path.join(GENDER_MODELS_DIR, "transformer_label_encoder.pkl"))

    sequences = tokenizer.texts_to_sequences(df["name"])
    X = pad_sequences(sequences, maxlen=max_len, padding="post")
    y_true = encoder.transform(df["sex"])
    proba = model.predict(X)
    y_pred = (proba[:, 1] >= threshold).astype(int)
    return y_true, y_pred, proba[:, 1], encoder.classes_


def compute_metrics(y_true, y_pred, y_proba, class_names):
    """
    Computes classification metrics for given true and predicted labels, along with
    class probabilities and class names. The function calculates accuracy, precision,
    recall, F1 score, and confusion matrix for evaluating model performance.
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
    parser.add_argument("--dataset", default="names_evaluation.csv", help="Path to the dataset CSV file")
    parser.add_argument("--size", type=int, help="Number of rows to load from the dataset")
    parser.add_argument("--balanced", action="store_true", help="Load balanced dataset")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for classification")
    args = parser.parse_args()

    df = load_csv_dataset(args.dataset, args.size, args.balanced)

    model_funcs = {
        "logreg": evaluate_logreg,
        "lstm": evaluate_lstm,
        "transformer": evaluate_transformer,
    }
    try:
        y_true, y_pred, y_proba, classes = model_funcs[args.model](df, args.threshold)
    except KeyError:
        raise ValueError(f"Unknown model: {args.model}")

    results = compute_metrics(y_true, y_pred, y_proba, classes)
    save_json_dataset(results, os.path.join(GENDER_RESULT_DIR, f'{args.model}_eval'))


if __name__ == "__main__":
    main()
