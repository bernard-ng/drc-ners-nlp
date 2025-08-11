from typing import Dict

import pandas as pd

from processing.ner.formats import BaseNameFormatter


class ReducedNativeFormatter(BaseNameFormatter):
    def transform(self, row: pd.Series) -> Dict:
        native_parts = self.parse_native_components(row['probable_native'])
        surname = row['probable_surname'] if pd.notna(row['probable_surname']) else ''

        # Keep only first native component + surname
        reduced_native = native_parts[0] if len(native_parts) > 1 else row['probable_native']
        full_name = f"{reduced_native} {surname}".strip()

        return {
            'name': full_name,
            'probable_native': reduced_native,
            'identify_name': reduced_native,
            'probable_surname': surname,
            'identify_surname': surname,
            'ner_entities': str(self.create_ner_tags(full_name, [reduced_native], surname)),
            'transformation_type': self.transformation_type,
            **self.compute_derived_attributes(full_name)
        }

    @property
    def transformation_type(self) -> str:
        return 'reduced_native'
