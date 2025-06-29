import csv
import io
import json
import os
import pickle
from typing import Optional
from typing import List, Dict

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'dataset')

MODELS_DIR = os.path.join(ROOT_DIR, 'models')
GENDER_MODELS_DIR = os.path.join(MODELS_DIR, 'gender')
GENDER_RESULT_DIR = os.path.join(ROOT_DIR, 'gender', 'results')

NER_MODELS_DIR = os.path.join(MODELS_DIR, 'ner')
NER_RESULT_DIR = os.path.join(ROOT_DIR, 'ner', 'results')


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


def load_csv_dataset(path: str, limit: int = None, balanced: bool = False) -> List[Dict[str, str]]:
    print(f">> Loading CSV dataset from {path}")

    file_path = os.path.join(DATA_DIR, path)
    with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        raw_text = f.read().replace('\x00', '')

    reader = csv.DictReader(io.StringIO(raw_text))
    print(f">> Detected fieldnames: {reader.fieldnames}")

    if balanced:
        by_sex = {'m': [], 'f': []}
        for row in reader:
            sex = row.get("sex", "").lower()
            if sex in by_sex:
                by_sex[sex].append(row)
        min_len = min(len(by_sex['m']), len(by_sex['f']))
        if limit:
            min_len = min(min_len, limit // 2)
        data = by_sex['m'][:min_len] + by_sex['f'][:min_len]
    else:
        data = []
        for i, row in enumerate(reader):
            data.append(row)
            if limit and i + 1 >= limit:
                break

    print(">> Successfully loaded with UTF-8 encoding")
    return data


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
