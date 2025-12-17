from typing import Dict
import pandas as pd

from ners.processing.ner.formats import BaseNameFormatter, is_nonempty


class NativeOnlyFormatter(BaseNameFormatter):
    def transform(self, row: pd.Series) -> Dict:
        native_raw = row.get("probable_native", None)

        native_parts = self.parse_native_components(native_raw)
        native_text = self._to_str(native_raw)

        full_name = native_text

        return {
            "name": full_name,
            "probable_native": native_text,
            "identified_name": native_text,
            "probable_surname": "",
            "identified_surname": "",
            "ner_entities": str(self.create_ner_tags(full_name, native_parts, "")),
            "transformation_type": self.transformation_type,
            **self.compute_numeric_features(full_name),
        }

    @property
    def transformation_type(self) -> str:
        return "native_only"
