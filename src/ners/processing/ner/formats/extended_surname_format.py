import random
from typing import Dict
import pandas as pd

from ners.processing.ner.formats import BaseNameFormatter, is_nonempty


class ExtendedSurnameFormatter(BaseNameFormatter):
    def transform(self, row: pd.Series) -> Dict:
        native_raw = row.get("probable_native", None)
        surname_raw = row.get("probable_surname", None)

        native_parts = self.parse_native_components(native_raw)
        native_text = self._to_str(native_raw)
        original_surname = self._to_str(surname_raw)

        additional_surname = random.choice(self.additional_surnames)
        combined_surname = f"{additional_surname} {original_surname}".strip()
        full_name = f"{native_text} {combined_surname}".strip()

        return {
            "name": full_name,
            "probable_native": native_text,
            "identified_name": native_text,
            "probable_surname": combined_surname,
            "identified_surname": combined_surname,
            "ner_entities": str(self.create_ner_tags(full_name, native_parts, combined_surname)),
            "transformation_type": self.transformation_type,
            **self.compute_numeric_features(full_name),
        }

    @property
    def transformation_type(self) -> str:
        return "extended_surname"
