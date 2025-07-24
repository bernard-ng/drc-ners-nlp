import csv
import io
import json
import logging
import os
import pickle
from typing import List, Dict

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'dataset')

MODELS_DIR = os.path.join(ROOT_DIR, 'models')
GENDER_MODELS_DIR = os.path.join(MODELS_DIR, 'gender')
GENDER_RESULT_DIR = os.path.join(ROOT_DIR, 'gender', 'results')

NER_MODELS_DIR = os.path.join(MODELS_DIR, 'ner')
NER_RESULT_DIR = os.path.join(ROOT_DIR, 'ner', 'results')

REGION_MAPPING = {
    # Kinshasa
    "kinshasa": ("KINSHASA", "KINSHASA"),
    "kinshasa-centre": ("KINSHASA", "KINSHASA"),
    "kinshasa-est": ("KINSHASA", "KINSHASA"),
    "kinshasa-funa": ("KINSHASA", "KINSHASA"),
    "kinshasa-lukunga": ("KINSHASA", "KINSHASA"),
    "kinshasa-mont-amba": ("KINSHASA", "KINSHASA"),
    "kinshasa-ouest": ("KINSHASA", "KINSHASA"),
    "kinshasa-plateau": ("KINSHASA", "KINSHASA"),
    "kinshasa-tshangu": ("KINSHASA", "KINSHASA"),

    # Bas-Congo → Kongo-Central → BAS-CONGO
    "bas-congo": ("KONGO-CENTRAL", "BAS-CONGO"),
    "bas-congo-1": ("KONGO-CENTRAL", "BAS-CONGO"),
    "bas-congo-2": ("KONGO-CENTRAL", "BAS-CONGO"),
    "kongo-central": ("KONGO-CENTRAL", "BAS-CONGO"),
    "kongo-central-1": ("KONGO-CENTRAL", "BAS-CONGO"),
    "kongo-central-2": ("KONGO-CENTRAL", "BAS-CONGO"),
    "kongo-central-3": ("KONGO-CENTRAL", "BAS-CONGO"),

    # Kwilu, Kwango, Mai-Ndombe → BANDUNDU
    "bandundu": ("BANDUNDU", "BANDUNDU"),
    "bandundu-1": ("BANDUNDU", "BANDUNDU"),
    "bandundu-2": ("BANDUNDU", "BANDUNDU"),
    "bandundu-3": ("BANDUNDU", "BANDUNDU"),
    "kwilu": ("KWILU", "BANDUNDU"),
    "kwilu-1": ("KWILU", "BANDUNDU"),
    "kwilu-2": ("KWILU", "BANDUNDU"),
    "kwilu-3": ("KWILU", "BANDUNDU"),
    "kwango": ("KWANGO", "BANDUNDU"),
    "kwango-1": ("KWANGO", "BANDUNDU"),
    "kwango-2": ("KWANGO", "BANDUNDU"),
    "mai-ndombe": ("MAI-NDOMBE", "BANDUNDU"),
    "mai-ndombe-1": ("MAI-NDOMBE", "BANDUNDU"),
    "mai-ndombe-2": ("MAI-NDOMBE", "BANDUNDU"),
    "mai-ndombe-3": ("MAI-NDOMBE", "BANDUNDU"),

    # Katanga → HAUT-KATANGA, HAUT-LOMAMI, LUALABA, TANGANYIKA
    "haut-katanga": ("HAUT-KATANGA", "KATANGA"),
    "haut-katanga-1": ("HAUT-KATANGA", "KATANGA"),
    "haut-katanga-2": ("HAUT-KATANGA", "KATANGA"),
    "haut-lomami": ("HAUT-LOMAMI", "KATANGA"),
    "haut-lomami-1": ("HAUT-LOMAMI", "KATANGA"),
    "haut-lomami-2": ("HAUT-LOMAMI", "KATANGA"),
    "lualaba": ("LUALABA", "KATANGA"),
    "lualaba-1": ("LUALABA", "KATANGA"),
    "lualaba-2": ("LUALABA", "KATANGA"),
    "lualaba-74-corrige-922a": ("LUALABA", "KATANGA"),
    "tanganyika": ("TANGANYIKA", "KATANGA"),
    "tanganyika-1": ("TANGANYIKA", "KATANGA"),
    "tanganyika-2": ("TANGANYIKA", "KATANGA"),

    # Equateur → MONGALA, NORD-UBANGI, SUD-UBANGI, TSHUAPA
    "equateur": ("EQUATEUR", "EQUATEUR"),
    "equateur-1": ("EQUATEUR", "EQUATEUR"),
    "equateur-2": ("EQUATEUR", "EQUATEUR"),
    "equateur-3": ("EQUATEUR", "EQUATEUR"),
    "equateur-4": ("EQUATEUR", "EQUATEUR"),
    "equateur-5": ("EQUATEUR", "EQUATEUR"),
    "mongala": ("MONGALA", "EQUATEUR"),
    "mongala-1": ("MONGALA", "EQUATEUR"),
    "mongala-2": ("MONGALA", "EQUATEUR"),
    "nord-ubangi": ("NORD-UBANGI", "EQUATEUR"),
    "nord-ubangi-1": ("NORD-UBANGI", "EQUATEUR"),
    "nord-ubangi-2": ("NORD-UBANGI", "EQUATEUR"),
    "sud-ubangi": ("SUD-UBANGI", "EQUATEUR"),
    "sud-ubangi-1": ("SUD-UBANGI", "EQUATEUR"),
    "sud-ubangi-2": ("SUD-UBANGI", "EQUATEUR"),
    "tshuapa": ("TSHUAPA", "EQUATEUR"),
    "tshuapa-1": ("TSHUAPA", "EQUATEUR"),
    "tshuapa-2": ("TSHUAPA", "EQUATEUR"),

    # Province-Orientale
    "province-orientale": ("PROVINCE-ORIENTALE", "PROVINCE-ORIENTALE"),
    "province-orientale-1": ("PROVINCE-ORIENTALE", "PROVINCE-ORIENTALE"),
    "province-orientale-2": ("PROVINCE-ORIENTALE", "PROVINCE-ORIENTALE"),
    "province-orientale-3": ("PROVINCE-ORIENTALE", "PROVINCE-ORIENTALE"),
    "province-orientale-4": ("PROVINCE-ORIENTALE", "PROVINCE-ORIENTALE"),
    "haut-uele": ("HAUT-UELE", "PROVINCE-ORIENTALE"),
    "haut-uele-1": ("HAUT-UELE", "PROVINCE-ORIENTALE"),
    "haut-uele-2": ("HAUT-UELE", "PROVINCE-ORIENTALE"),
    "bas-uele": ("BAS-UELE", "PROVINCE-ORIENTALE"),
    "ituri": ("ITURI", "PROVINCE-ORIENTALE"),
    "ituri-1": ("ITURI", "PROVINCE-ORIENTALE"),
    "ituri-2": ("ITURI", "PROVINCE-ORIENTALE"),
    "ituri-3": ("ITURI", "PROVINCE-ORIENTALE"),
    "tshopo": ("TSHOPO", "PROVINCE-ORIENTALE"),
    "tshopo-1": ("TSHOPO", "PROVINCE-ORIENTALE"),
    "tshopo-2": ("TSHOPO", "PROVINCE-ORIENTALE"),

    # Kasaï
    "kasai-1": ("KASAÏ", "KASAÏ-OCCIDENTAL"),
    "kasai-2": ("KASAÏ", "KASAÏ-OCCIDENTAL"),
    "kasai-ce": ("KASAÏ", "KASAÏ-OCCIDENTAL"),
    "kasai-central": ("KASAÏ-CENTRAL", "KASAÏ-OCCIDENTAL"),
    "kasai-central-1": ("KASAÏ-CENTRAL", "KASAÏ-OCCIDENTAL"),
    "kasai-central-2": ("KASAÏ-CENTRAL", "KASAÏ-OCCIDENTAL"),
    "kasai-occidental": ("KASAÏ-CENTRAL", "KASAÏ-OCCIDENTAL"),
    "kasai-occidental-1": ("KASAÏ-CENTRAL", "KASAÏ-OCCIDENTAL"),
    "kasai-occidental-2": ("KASAÏ-CENTRAL", "KASAÏ-OCCIDENTAL"),
    "kasai-oriental": ("KASAÏ-ORIENTAL", "KASAÏ-ORIENTAL"),
    "kasai-oriental-1": ("KASAÏ-ORIENTAL", "KASAÏ-ORIENTAL"),
    "kasai-oriental-2": ("KASAÏ-ORIENTAL", "KASAÏ-ORIENTAL"),
    "kasai-oriental-3": ("KASAÏ-ORIENTAL", "KASAÏ-ORIENTAL"),
    "kasai-orientale": ("KASAÏ-ORIENTAL", "KASAÏ-ORIENTAL"),
    "lomami": ("LOMAMI", "KASAÏ-ORIENTAL"),
    "lomami-1": ("LOMAMI", "KASAÏ-ORIENTAL"),
    "lomami-2": ("LOMAMI", "KASAÏ-ORIENTAL"),
    "sankuru": ("SANKURU", "KASAÏ-ORIENTAL"),
    "sankuru-1": ("SANKURU", "KASAÏ-ORIENTAL"),
    "sankuru-2": ("SANKURU", "KASAÏ-ORIENTAL"),

    # Nord-Kivu
    "nord-kivu": ("NORD-KIVU", "NORD-KIVU"),
    "nord-kivu-1": ("NORD-KIVU", "NORD-KIVU"),
    "nord-kivu-2": ("NORD-KIVU", "NORD-KIVU"),
    "nord-kivu-3": ("NORD-KIVU", "NORD-KIVU"),

    # Sud-Kivu
    "sud-kivu": ("SUD-KIVU", "SUD-KIVU"),
    "sud-kivu-1": ("SUD-KIVU", "SUD-KIVU"),
    "sud-kivu-2": ("SUD-KIVU", "SUD-KIVU"),
    "sud-kivu-3": ("SUD-KIVU", "SUD-KIVU"),

    # Maniema
    "maniema": ("MANIEMA", "MANIEMA"),
    "maniema-1": ("MANIEMA", "MANIEMA"),
    "maniema-2": ("MANIEMA", "MANIEMA"),

    # Divers
    "hors-frontieres": ("AUTRES", "AUTRES"),
    "lukaya": ("AUTRES", "AUTRES"),
    "recours": ("AUTRES", "AUTRES"),
    "junacyc": ("AUTRES", "AUTRES"),
    "junacyp": ("AUTRES", "AUTRES"),
    "junacyc-lualaba-corrige": ("LUALABA", "KATANGA"),
    "options-techniques-toutes-les-provinces-et-hors-frontieres": ("AUTRES", "AUTRES"),
    "region": ("AUTRES", "AUTRES"),
}

logging.basicConfig(level=logging.INFO, format=">> %(message)s")

def load_json_dataset(path: str) -> list:
    logging.info(f"Loading JSON dataset from {path}")
    with open(os.path.join(DATA_DIR, path), "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv_dataset(data: list, path: str) -> None:
    logging.info(f"Saving CSV dataset to {path}")
    with open(os.path.join(DATA_DIR, path), "w", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


def load_csv_dataset(path: str, limit: int = None, balanced: bool = False) -> List[Dict[str, str]]:
    logging.info(f"Loading CSV dataset from {path}")

    file_path = os.path.join(DATA_DIR, path)
    with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        raw_text = f.read().replace('\x00', '')

    reader = csv.DictReader(io.StringIO(raw_text))
    logging.info(f"Detected fieldnames: {reader.fieldnames}")

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

    logging.info("Successfully loaded with UTF-8 encoding")
    return data


def save_json_dataset(data: list, path: str) -> None:
    logging.info(f"Saving JSON dataset to {path}")
    with open(os.path.join(DATA_DIR, path), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(',', ':'))


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_prompt() -> str:
    with open(os.path.join(ROOT_DIR, 'prompt.txt'), 'r') as f:
        return f.read()
