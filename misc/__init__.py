import csv
import io
import json
import os
import pickle
from typing import Optional

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'dataset')

MODELS_DIR = os.path.join(ROOT_DIR, 'models')
GENDER_MODELS_DIR = os.path.join(MODELS_DIR, 'gender')
GENDER_RESULT_DIR = os.path.join(ROOT_DIR, 'gender', 'results')

NER_MODELS_DIR = os.path.join(MODELS_DIR, 'ner')
NER_RESULT_DIR = os.path.join(ROOT_DIR, 'ner', 'results')


def clean_spacing(filename: str) -> Optional[str]:
    try:
        with open(os.path.join(DATA_DIR, filename), 'r', encoding='utf8') as f:
            content = f.read()
            return content.translate(str.maketrans({'\00': ' ', ' ': ' '}))
    except Exception as e:
        return None


def load_json_dataset(path: str) -> list:
    print(f">> Loading JSON dataset from {path}")
    with open(os.path.join(DATA_DIR, path), "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv_dataset(data: list, path: str) -> None:
    print(f">> Saving CSV dataset to {path}")
    with open(os.path.join(DATA_DIR, path), "w", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


def load_csv_dataset(path: str, limit: int = None) -> list:
    print(f">> Loading CSV dataset from {path}")
    data = []
    encodings = ['utf-8', 'utf-16', 'latin1']

    for enc in encodings:
        try:
            with open(os.path.join(DATA_DIR, path), "r", encoding=enc, errors="replace") as f:
                raw_text = f.read().replace('\x00', '')

            csv_buffer = io.StringIO(raw_text)
            reader = csv.DictReader(csv_buffer)
            print(f">> Detected fieldnames: {reader.fieldnames}")

            for row in reader:
                data.append(row)
                if limit and len(data) >= limit:
                    break
            print(f">> Successfully loaded with encoding: {enc}")
            return data
        except Exception as e:
            print(f">> Failed with encoding: {enc}, error: {e}")

    raise UnicodeDecodeError("load_csv_dataset", path, 0, 0, "Unable to decode file with common encodings.")


def save_json_dataset(data: list, path: str) -> None:
    print(f">> Saving JSON dataset to {path}")
    with open(os.path.join(DATA_DIR, path), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(',', ':'))


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
