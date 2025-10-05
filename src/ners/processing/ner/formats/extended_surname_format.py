import random
from typing import Dict

import pandas as pd

from ners.processing.ner.formats import BaseNameFormatter


class ExtendedSurnameFormatter(BaseNameFormatter):
    def transform(self, row: pd.Series) -> Dict:
        native_parts = self.parse_native_components(row["probable_native"])
        original_surname = (
            row["probable_surname"] if pd.notna(row["probable_surname"]) else ""
        )

        # Add random additional surname
        additional_surname = random.choice(self.additional_surnames)
        combined_surname = f"{additional_surname} {original_surname}".strip()
        full_name = f"{row['probable_native']} {combined_surname}".strip()

        return {
            "name": full_name,
            "probable_native": row["probable_native"],
            "identified_name": row["probable_native"],
            "probable_surname": combined_surname,
            "identified_surname": combined_surname,
            "ner_entities": str(
                self.create_ner_tags(full_name, native_parts, combined_surname)
            ),
            "transformation_type": self.transformation_type,
            **self.compute_numeric_features(full_name),
        }

    @property
    def transformation_type(self) -> str:
        return "extended_surname"
