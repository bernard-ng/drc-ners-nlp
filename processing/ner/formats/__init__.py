from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

import pandas as pd

from processing.steps.feature_extraction_step import NameCategory


class BaseNameFormatter(ABC):
    """
    Base class for name formatting transformations.
    Contains common logic for NER tagging and attribute computation.
    """

    def __init__(self, connectors: List[str] = None, additional_surnames: List[str] = None):
        self.connectors = connectors or ["wa", "ya", "ka", "ba"]
        self.additional_surnames = additional_surnames or [
            "jean",
            "paul",
            "marie",
            "joseph",
            "pierre",
            "claude",
            "andre",
            "michel",
            "robert",
        ]

    @classmethod
    def parse_native_components(cls, native_str: str) -> List[str]:
        """Parse native name string into individual components"""
        if pd.isna(native_str) or not native_str:
            return []
        return native_str.strip().split()

    def create_ner_tags(
        self, text: str, native_parts: List[str], surname: str
    ) -> List[Tuple[int, int, str]]:
        """Create NER entity tags for transformed text"""
        entities = []
        current_pos = 0
        words = text.split()

        for word in words:
            start_pos = current_pos
            end_pos = current_pos + len(word)

            # Determine tag based on word content
            if word in native_parts or any(connector in word for connector in self.connectors):
                tag = "NATIVE"
            elif word == surname or word in self.additional_surnames:
                tag = "SURNAME"
            else:
                # Check if it's a compound native word or new surname
                if any(part in word for part in native_parts):
                    tag = "NATIVE"
                else:
                    tag = "SURNAME"

            entities.append((start_pos, end_pos, tag))
            current_pos = end_pos + 1  # +1 for space

        return entities

    @classmethod
    def compute_numeric_features(cls, name: str) -> Dict:
        """Compute all derived attributes for the transformed name"""
        words_count = len(name.split()) if name else 0
        length = len(name) if name else 0

        return {
            "words": words_count,
            "length": length,
            "identified_category": (
                NameCategory.SIMPLE.value if words_count == 3 else NameCategory.COMPOSE.value
            ),
        }

    @abstractmethod
    def transform(self, row: pd.Series) -> Dict:
        """Transform a row according to the specific format rules"""
        pass

    @property
    @abstractmethod
    def transformation_type(self) -> str:
        """Return the transformation type identifier"""
        pass
