import argparse
import os
from typing import List

import tensorflow as tf
from sklearn.pipeline import Pipeline
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from misc import GENDER_MODELS_DIR, load_pickle


def predict_logreg(names: List[str], threshold: float):
    """
    Predict gender labels for given names using a logistic regression model.

    The function takes in a list of names and predicts the gender labels
    based on a logistic regression model. A probabilistic threshold is used
    to classify the names into one of the defined labels.

    :param names:
        A list of names for which the gender needs to be predicted. Each
        name must be a string.
    :param threshold:
        A float value representing the threshold for classification. Names
        with predicted probabilities greater than or equal to this value
        will be classified into the positive class.
    :return:
        A tuple containing the predicted gender labels and their
        corresponding probabilities. The first element of the tuple is a
        list of predicted labels, while the second element is an array of
        probability scores for each label.
    """
    model_path = os.path.join(GENDER_MODELS_DIR, "regression_model.pkl")
    encoder_path = os.path.join(GENDER_MODELS_DIR, "regression_label_encoder.pkl")

    model: Pipeline = load_pickle(model_path)
    label_encoder = load_pickle(encoder_path)

    X = [name.lower().strip() for name in names]
    proba = model.predict_proba(X)
    pred = (proba[:, 1] >= threshold).astype(int)
    labels = label_encoder.inverse_transform(pred)
    return labels, proba


def predict_lstm(names: List[str], threshold: float, max_len=6):
    """
    Predicts gender labels and probabilities for a list of names using a pre-trained BiLSTM model.

    The function loads the model, tokenizer, and label encoder, performs preprocessing on the input
    names, and then uses the loaded model to predict gender probabilities. Based on the threshold
    value, it determines the predicted gender labels.

    :param names: List of names to be classified.
    :type names: List[str]
    :param threshold: Probability threshold for classifying gender. If the predicted probability for the
        'positive' class is greater than or equal to this threshold, it is classified accordingly.
    :type threshold: float
    :param max_len: Maximum length for name sequences. Names longer than this will be truncated, and shorter
        ones will be padded. Default value is 6.
    :type max_len: int, optional

    :return: A tuple containing predicted labels and associated probabilities. Labels are the predicted gender
        categories, and probabilities are the prediction scores for each input name.
    :rtype: Tuple[numpy.ndarray, numpy.ndarray]
    """
    model_path = os.path.join(GENDER_MODELS_DIR, "lstm_model.keras")
    tokenizer_path = os.path.join(GENDER_MODELS_DIR, "lstm_tokenizer.pkl")
    encoder_path = os.path.join(GENDER_MODELS_DIR, "lstm_label_encoder.pkl")

    model = tf.keras.models.load_model(model_path)
    tokenizer: Tokenizer = load_pickle(tokenizer_path)
    label_encoder = load_pickle(encoder_path)

    X = tokenizer.texts_to_sequences([n.lower().strip() for n in names])
    X = pad_sequences(X, maxlen=max_len, padding="post")
    proba = model.predict(X)
    pred = (proba[:, 1] >= threshold).astype(int)
    labels = label_encoder.inverse_transform(pred)
    return labels, proba


def predict_transformer(names: List[str], threshold: float, max_len=6):
    """
    Predicts gender labels for the provided names using a pre-trained transformer model.

    This function loads a pre-trained transformer model along with its tokenizer and label
    encoder, converts input names into tokenized sequences, and processes them to generate
    gender predictions. The function returns the predicted labels and the associated
    probabilities for each sample.

    :param names: List of names to predict gender labels for.
    :type names: List[str]
    :param threshold: Threshold value to determine the prediction class. Probability values
       above or equal to the threshold will be assigned to one class, and those below to
       another.
    :type threshold: float
    :param max_len: Maximum length for the sequences. Names will be truncated or padded to
       this length during processing, default is 6.
    :type max_len: int, optional
    :return: A tuple containing two elements: a list of predicted gender labels as strings
       and a NumPy array of probabilities for each gender class (where the first index
       corresponds to one class, and the second index corresponds to another).
    :rtype: Tuple[List[str], numpy.ndarray]
    """
    model_path = os.path.join(GENDER_MODELS_DIR, "transformer.keras")
    tokenizer_path = os.path.join(GENDER_MODELS_DIR, "transformer_tokenizer.pkl")
    encoder_path = os.path.join(GENDER_MODELS_DIR, "transformer_label_encoder.pkl")

    model = tf.keras.models.load_model(model_path)
    tokenizer: Tokenizer = load_pickle(tokenizer_path)
    label_encoder = load_pickle(encoder_path)

    X = tokenizer.texts_to_sequences([n.lower().strip() for n in names])
    X = pad_sequences(X, maxlen=max_len, padding="post")
    proba = model.predict(X)
    pred = (proba[:, 1] >= threshold).astype(int)
    labels = label_encoder.inverse_transform(pred)
    return labels, proba


def main():
    parser = argparse.ArgumentParser(description="Predict gender from names using trained model")
    parser.add_argument("--model", choices=["logreg", "lstm", "transformer"], required=True)
    parser.add_argument("--name", nargs="+", required=True, help="One or more names")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for classification")
    args = parser.parse_args()

    model = args.model
    names = args.name
    threshold = args.threshold

    if model == "logreg":
        labels, proba = predict_logreg(names, threshold)
    elif model == "lstm":
        labels, proba = predict_lstm(names, threshold)
    elif model == "transformer":
        labels, proba = predict_transformer(names, threshold)
    else:
        raise ValueError(f"Unsupported model type: {model}")

    for i, name in enumerate(names):
        p_female = proba[i][0]
        p_male = proba[i][1]
        print(f"{name} → {labels[i]} | P(f): {p_female:.2f} | P(m): {p_male:.2f}")


if __name__ == "__main__":
    main()
