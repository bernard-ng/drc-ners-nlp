from typing import Dict

import pandas as pd

from ners.processing.ner.formats import BaseNameFormatter


class NativeOnlyFormatter(BaseNameFormatter):
    def transform(self, row: pd.Series) -> Dict:
        native_parts = self.parse_native_components(row["probable_native"])  # type: ignore

        # Only native components
        full_name = row["probable_native"]

        return {
            "name": full_name,
            "probable_native": row["probable_native"],
            "identified_name": row["probable_native"],
            "probable_surname": "",
            "identified_surname": "",
            "ner_entities": str(self.create_ner_tags(full_name, native_parts, "")),  # type: ignore
            "transformation_type": self.transformation_type,
            **self.compute_numeric_features(full_name),  # type: ignore
        }

    @property
    def transformation_type(self) -> str:
        return "native_only"
