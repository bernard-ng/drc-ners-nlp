from typing import Dict
import pandas as pd

from ners.processing.ner.formats import BaseNameFormatter


class PositionFlippedFormatter(BaseNameFormatter):
    def transform(self, row: pd.Series) -> Dict:
        native_raw = row.get("probable_native", None)
        surname_raw = row.get("probable_surname", None)

        native_parts = self.parse_native_components(native_raw)
        native_text = self._to_str(native_raw)
        surname = self._to_str(surname_raw)

        # Flip order: surname + native components
        full_name = f"{surname} {native_text}".strip()

        return {
            "name": full_name,
            "probable_native": native_text,
            "identified_name": native_text,
            "probable_surname": surname,
            "identified_surname": surname,
            "ner_entities": str(self.create_ner_tags(full_name, native_parts, surname)),
            "transformation_type": self.transformation_type,
            **self.compute_numeric_features(full_name),
        }

    @property
    def transformation_type(self) -> str:
        return "position_flipped"
