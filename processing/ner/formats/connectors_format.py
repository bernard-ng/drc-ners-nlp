import random
from typing import Dict

import pandas as pd

from processing.ner.formats import BaseNameFormatter


class ConnectorFormatter(BaseNameFormatter):
    def transform(self, row: pd.Series) -> Dict:
        native_parts = self.parse_native_components(row['probable_native'])
        surname = row['probable_surname'] if pd.notna(row['probable_surname']) else ''
        connector = random.choice(self.connectors)

        # Connect native parts with a random connector
        if len(native_parts) > 1:
            connected_native = f" {connector} ".join(native_parts)
            full_name = f"{connected_native} {surname}".strip()
        else:
            connected_native = f"{row['probable_native']} {connector} {row['probable_native']}".strip()
            full_name = f"{connected_native} {surname}".strip()

        return {
            'name': full_name,
            'probable_native': connected_native,
            'identify_name': connected_native,
            'probable_surname': surname,
            'identify_surname': surname,
            'ner_entities': str(self.create_ner_tags(full_name, native_parts, surname)),
            'transformation_type': self.transformation_type,
            **self.compute_derived_attributes(full_name)
        }

    @property
    def transformation_type(self) -> str:
        return 'connector_added'
