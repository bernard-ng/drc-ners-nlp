from typing import Dict

import pandas as pd

from ners.processing.ner.formats import BaseNameFormatter


class PositionFlippedFormatter(BaseNameFormatter):
    def transform(self, row: pd.Series) -> Dict:
        native_parts = self.parse_native_components(row["probable_native"])
        surname = row["probable_surname"] if pd.notna(row["probable_surname"]) else ""

        # Flip order: surname + native components
        full_name = f"{surname} {row['probable_native']}".strip()

        return {
            "name": full_name,
            "probable_native": row["probable_native"],
            "identified_name": row["probable_native"],
            "probable_surname": surname,
            "identified_surname": surname,
            "ner_entities": str(self.create_ner_tags(full_name, native_parts, surname)),
            "transformation_type": self.transformation_type,
            **self.compute_numeric_features(full_name),
        }

    @property
    def transformation_type(self) -> str:
        return "position_flipped"
