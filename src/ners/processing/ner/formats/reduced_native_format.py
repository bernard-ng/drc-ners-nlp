from typing import Dict, Any, List, cast
import pandas as pd

from ners.processing.ner.formats import BaseNameFormatter


class ReducedNativeFormatter(BaseNameFormatter):
    def transform(self, row: pd.Series) -> Dict[str, Any]:
        # On extrait les valeurs en s'assurant qu'elles sont traitées comme des types simples
        raw_native = row.get("probable_native", "")
        raw_surname = row.get("probable_surname", "")

        # Correction erreur pd.notna : conversion explicite en bool
        has_surname = bool(pd.notna(raw_surname))
        surname = str(raw_surname) if has_surname else ""

        # Analyse des composants natifs
        native_parts = cast(List[str], self.parse_native_components(raw_native))
        
        # Keep only first native component + surname
        # On s'assure que reduced_native est bien un str
        reduced_native = str(native_parts[0] if len(native_parts) > 0 else raw_native)
        
        full_name = f"{reduced_native} {surname}".strip()

        return {
            "name": full_name,
            "probable_native": reduced_native,
            "identified_name": reduced_native,
            "probable_surname": surname,
            "identified_surname": surname,
            "ner_entities": str(
                self.create_ner_tags(
                    full_name, 
                    [reduced_native], # Liste de strings attendue
                    surname           # String attendu
                )
            ),
            "transformation_type": self.transformation_type,
            **cast(Dict[str, Any], self.compute_numeric_features(full_name)),
        }

    @property
    def transformation_type(self) -> str:
        return "reduced_native"