from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Union, Optional, Any
import pandas as pd
import numpy as np

from ners.processing.steps.feature_extraction_step import NameCategory

TextLike = Union[str, pd.Series, np.ndarray, None]


def is_nonempty(value: Any) -> bool:
    if value is None:
        return False

    if isinstance(value, (pd.Series, pd.DataFrame)):
        return not value.empty

    if isinstance(value, np.ndarray):
        return value.size > 0

    if isinstance(value, str):
        return len(value.strip()) > 0

    try:
        return bool(value)
    except (ValueError, TypeError):
        return False


class BaseNameFormatter(ABC):
    def __init__(
        self,
        connectors: Optional[List[str]] = None,
        additional_surnames: Optional[List[str]] = None,
    ) -> None:
        self.connectors: List[str] = (
            connectors
            if connectors is not None
            else [
                "wa",
                "ya",
                "ka",
                "ba",
            ]
        )
        self.additional_surnames: List[str] = (
            additional_surnames
            if additional_surnames is not None
            else [
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
        )

    @classmethod
    def _to_str(cls, value: TextLike) -> str:
        if value is None:
            return ""

        if isinstance(value, pd.Series):
            if value.empty:
                return ""
            v = value.iloc[0]
            return "" if pd.isna(v) else str(v)

        if isinstance(value, np.ndarray):
            if value.size == 0:
                return ""
            v = value.flat[0]
            return "" if pd.isna(v) else str(v)

        if pd.isna(value):
            return ""

        return str(value)

    @classmethod
    def parse_native_components(cls, native_str: TextLike) -> List[str]:
        text = cls._to_str(native_str)
        if not text:
            return []
        return text.strip().split()

    def create_ner_tags(
        self, text: str, native_parts: List[str], surname: str
    ) -> List[Tuple[int, int, str]]:
        entities: List[Tuple[int, int, str]] = []
        current_pos = 0
        words = text.split()

        for word in words:
            start_pos = current_pos
            end_pos = current_pos + len(word)

            # Logique de tagging
            is_native = (
                word in native_parts
                or any(connector in word for connector in self.connectors)
                or any(part in word for part in native_parts)
            )

            if is_native:
                tag = "NATIVE"
            elif word == surname or word in self.additional_surnames:
                tag = "SURNAME"
            else:
                tag = "SURNAME"

            entities.append((start_pos, end_pos, tag))
            current_pos = end_pos + 1

        return entities

    @classmethod
    def compute_numeric_features(cls, name: TextLike) -> Dict[str, Any]:
        """Calcule les attributs numériques (safe pour Series/arrays)."""
        text = cls._to_str(name)
        words_count = len(text.split()) if text else 0
        length = len(text) if text else 0

        return {
            "words": words_count,
            "length": length,
            "identified_category": (
                NameCategory.SIMPLE.value
                if words_count == 3
                else NameCategory.COMPOSE.value
            ),
        }

    @abstractmethod
    def transform(self, row: pd.Series) -> Dict[str, Any]:
        """Transformation spécifique à implémenter dans les classes filles."""
        raise NotImplementedError

    @property
    @abstractmethod
    def transformation_type(self) -> str:
        """Identifiant du type de transformation."""
        raise NotImplementedError
