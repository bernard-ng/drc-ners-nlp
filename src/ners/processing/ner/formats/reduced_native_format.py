from typing import Dict
import pandas as pd

from ners.processing.ner.formats import BaseNameFormatter, is_nonempty


class ReducedNativeFormatter(BaseNameFormatter):
    def transform(self, row: pd.Series) -> Dict:
        native_raw = row.get("probable_native", None)
        surname_raw = row.get("probable_surname", None)

        native_parts = self.parse_native_components(native_raw)
        native_text = self._to_str(native_raw)
        surname = self._to_str(surname_raw)

        # Keep only first native component + surname
        reduced_native = native_parts[0] if native_parts else native_text
        full_name = f"{reduced_native} {surname}".strip()

        return {
            "name": full_name,
            "probable_native": reduced_native,
            "identified_name": reduced_native,
            "probable_surname": surname,
            "identified_surname": surname,
            "ner_entities": str(self.create_ner_tags(full_name, [reduced_native], surname)),
            "transformation_type": self.transformation_type,
            **self.compute_numeric_features(full_name),
        }

    @property
    def transformation_type(self) -> str:
        return "reduced_native"
