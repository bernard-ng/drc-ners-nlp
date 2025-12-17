import random
from typing import Dict
import pandas as pd

from ners.processing.ner.formats import BaseNameFormatter


class ConnectorFormatter(BaseNameFormatter):
    def transform(self, row: pd.Series) -> Dict:
        native_raw = row.get("probable_native", None)
        surname_raw = row.get("probable_surname", None)

        native_parts = self.parse_native_components(native_raw)
        native_text = self._to_str(native_raw)
        surname = self._to_str(surname_raw)

        connector = random.choice(self.connectors)

        if len(native_parts) > 1:
            connected_native = f" {connector} ".join(native_parts)
        else:
            # if native_text empty, keep it safe
            connected_native = f"{native_text} {connector} {native_text}".strip()

        full_name = f"{connected_native} {surname}".strip()

        return {
            "name": full_name,
            "probable_native": connected_native,
            "identified_name": connected_native,
            "probable_surname": surname,
            "identified_surname": surname,
            "ner_entities": str(self.create_ner_tags(full_name, native_parts, surname)),
            "transformation_type": self.transformation_type,
            **self.compute_numeric_features(full_name),
        }

    @property
    def transformation_type(self) -> str:
        return "connector_added"
