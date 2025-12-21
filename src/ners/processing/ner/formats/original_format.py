from typing import Dict

import pandas as pd

from ners.processing.ner.formats import BaseNameFormatter


class OriginalFormatter(BaseNameFormatter):
    def transform(self, row: pd.Series) -> Dict:
        # Handle NaN values properly - use None instead of converting to "nan"
        probable_native = row["probable_native"] if pd.notna(row["probable_native"]).all() else None
        probable_surname = row["probable_surname"] if pd.notna(row["probable_surname"]).all() else None

        native_str = str(probable_native) if probable_native is not None else ""
        surname_str = str(probable_surname) if probable_surname is not None else ""

        native_parts = self.parse_native_components(native_str) if native_str else []

        # Keep original order: native components + surname
        full_name = f"{native_str} {surname_str}".strip()

        return {
            "name": full_name,
            "probable_native": probable_native,
            "identified_name": probable_native,
            "probable_surname": probable_surname,
            "identified_surname": probable_surname,
            "ner_entities": str(self.create_ner_tags(full_name, native_parts, surname_str)),
            "transformation_type": self.transformation_type,
            **self.compute_numeric_features(full_name),
        }

    @property
    def transformation_type(self) -> str:
        return "original"
