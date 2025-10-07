from typing import Union, Dict, Any, List
import ast
import json
import logging
import pandas as pd
from spacy.util import filter_spans


class NameTagger:
    def tag_name(
        self, name: str, probable_native: str, probable_surname: str
    ) -> Union[Dict[str, Any], None]:
        """Create a single NER training example using probable_native and probable_surname"""
        if not name or not probable_native or not probable_surname:
            return None

        name = name.strip()
        probable_native = probable_native.strip()
        probable_surname = probable_surname.strip()

        entities = []
        used_spans = []  # Track used character spans to prevent overlaps

        # Helper function to check if a span overlaps with any existing span
        def has_overlap(start, end):
            for used_start, used_end in used_spans:
                if not (end <= used_start or start >= used_end):
                    return True
            return False

        # Find positions of native names in the full name
        native_words = probable_native.split()
        name_lower = name.lower()  # Use lowercase for consistent searching
        processed_native_words = set()

        for native_word in native_words:
            native_word = native_word.strip()
            if len(native_word) < 2:  # Skip very short words
                continue

            native_word_lower = native_word.lower()

            # Skip if we've already processed this exact word
            if native_word_lower in processed_native_words:
                continue
            processed_native_words.add(native_word_lower)

            # Find the first occurrence of this native word that doesn't overlap
            start_pos = 0
            while True:
                pos = name_lower.find(
                    native_word_lower, start_pos
                )  # Case-insensitive search
                if pos == -1:
                    break

                # Calculate end position - make sure we only include the word itself
                end_pos = pos + len(native_word_lower)

                # Double-check that the extracted span matches exactly what we expect
                extracted_text = name[pos:end_pos]  # Get original case text
                if extracted_text.lower() != native_word_lower:
                    start_pos = pos + 1
                    continue

                # Check if this is a word boundary match and doesn't overlap
                if self._is_word_boundary_match(name, pos, end_pos) and not has_overlap(
                    pos, end_pos
                ):
                    entities.append((pos, end_pos, "NATIVE"))
                    used_spans.append((pos, end_pos))
                    break  # Only take the first non-overlapping occurrence

                start_pos = pos + 1

        # Find position of surname in the full name
        if probable_surname and len(probable_surname.strip()) >= 2:
            surname_lower = probable_surname.lower()

            # Find the first occurrence that doesn't overlap
            start_pos = 0
            while True:
                pos = name_lower.find(
                    surname_lower, start_pos
                )  # Case-insensitive search
                if pos == -1:
                    break

                # Calculate end position correctly - exact match only
                end_pos = pos + len(surname_lower)

                # Double-check that the extracted span matches exactly what we expect
                extracted_text = name[pos:end_pos]  # Get original case text
                if extracted_text.lower() != surname_lower:
                    start_pos = pos + 1
                    continue

                if self._is_word_boundary_match(name, pos, end_pos) and not has_overlap(
                    pos, end_pos
                ):
                    entities.append((pos, end_pos, "SURNAME"))
                    used_spans.append((pos, end_pos))
                    break

                start_pos = pos + 1

        if not entities:
            logging.warning(
                f"No valid entities found for name: '{name}' with native: '{probable_native}' and surname: '{probable_surname}'"
            )
            return None

        # Sort entities by position and validate
        entities.sort(key=lambda x: x[0])

        # Final validation - ensure no overlaps and valid spans
        validated_entities = []
        for start, end, label in entities:
            # Check bounds
            if not (0 <= start < end <= len(name)):
                logging.warning(
                    f"Invalid span bounds ({start}, {end}) for text length {len(name)}: '{name}'"
                )
                continue

            # Check for overlaps with already validated entities
            if any(
                start < v_end and end > v_start
                for v_start, v_end, _ in validated_entities
            ):
                logging.warning(
                    f"Overlapping span ({start}, {end}, '{label}') in '{name}'"
                )
                continue

            # CRITICAL VALIDATION: Check that the span contains only the expected word (no spaces)
            span_text = name[start:end]
            if not span_text or span_text != span_text.strip() or " " in span_text:
                logging.warning(
                    f"Span contains spaces or is empty ({start}, {end}) in '{name}': '{span_text}'"
                )
                continue

            validated_entities.append((start, end, label))

        if not validated_entities:
            logging.warning(f"No valid entities after validation for: '{name}'")
            return None

        # Convert to string format that matches the dataset
        entities_str = str(validated_entities)

        return {
            "entities": entities_str,
            "spans": validated_entities,  # Keep the original tuples for internal use
        }

    @classmethod
    def _is_word_boundary_match(cls, text: str, start: int, end: int) -> bool:
        """Check if the match is at word boundaries"""
        # Check character before start position
        if start > 0:
            prev_char = text[start - 1]
            if prev_char.isalnum():
                return False

        # Check character after end position
        if end < len(text):
            next_char = text[end]
            if next_char.isalnum():
                return False

        return True

    @classmethod
    def extract_entity_text(cls, name: str, entities_str: str) -> Dict[str, List[str]]:
        """Extract the actual text for each entity type"""
        result = {"NATIVE": [], "SURNAME": []}

        try:
            entities = ast.literal_eval(entities_str)

            for start, end, label in entities:
                if 0 <= start < end <= len(name):
                    span_text = name[start:end]
                    if label in result:
                        result[label].append(span_text)

        except (ValueError, SyntaxError, TypeError):
            pass

        return result

    @classmethod
    def parse(cls, entities_str: str) -> List[tuple]:
        """Parse entity strings from various formats.

        Supports formats:
        - [(start, end, label), ...]
        - [[start, end, label], ...]
        - [{"start": start, "end": end, "label": label}, ...]
        """
        if not entities_str or entities_str in ["[]", "", "nan"]:
            return []
        entities_str = str(entities_str).strip()
        try:
            if entities_str.startswith("[(") and entities_str.endswith(")]"):
                return ast.literal_eval(entities_str)
            elif entities_str.startswith("[[") and entities_str.endswith("]]"):
                return [tuple(e) for e in ast.literal_eval(entities_str)]
            elif entities_str.startswith("[{") and entities_str.endswith("}]"):
                return [
                    (e["start"], e["end"], e["label"]) for e in json.loads(entities_str)
                ]
            else:
                parsed = ast.literal_eval(entities_str)
                return [
                    tuple(e)
                    for e in parsed
                    if isinstance(e, (list, tuple)) and len(e) == 3
                ]
        except (ValueError, SyntaxError, json.JSONDecodeError):
            return []

    def parse_entities(self, series: pd.Series) -> pd.Series:
        """Vectorized parse of entity strings."""
        return series.map(self.parse)

    @classmethod
    def validate(cls, text: str, entities: List[tuple]) -> List[tuple]:
        """Advanced entity validation with overlap removal.

        This is more comprehensive than the basic validate_entities method.
        """
        if not entities or not text:
            return []
        text = str(text).strip()
        valid = []

        for ent in entities:
            if not isinstance(ent, (list, tuple)) or len(ent) != 3:
                continue
            start, end, label = ent
            try:
                start, end = int(start), int(end)
            except (ValueError, TypeError):
                continue
            if not isinstance(label, str):
                continue
            if not (0 <= start < end <= len(text)):
                continue
            if not text[start:end].strip():
                continue
            valid.append((start, end, label))

        if not valid:
            return []

        valid.sort(key=lambda x: (x[0], x[1]))

        # Remove overlaps
        filtered, last_end = [], -1
        for s, e, label in valid:
            if s >= last_end:
                filtered.append((s, e, label))
                last_end = e
        return filtered

    def validate_entities(
        self, texts: pd.Series, entities_series: pd.Series
    ) -> pd.Series:
        """Vectorized entity validation."""
        return pd.Series(map(self.validate, texts, entities_series), index=texts.index)

    @classmethod
    def create_docs(cls, nlp, texts: List[str], entities: List[List[tuple]]) -> List:
        """Batch create spaCy Docs from texts and entities."""
        docs = []
        for text, ents in zip(texts, entities):
            doc = nlp(text)
            spans = []
            for start, end, label in ents:
                span = doc.char_span(
                    start, end, label=label, alignment_mode="contract"
                ) or doc.char_span(start, end, label=label, alignment_mode="strict")
                if span:
                    spans.append(span)
            doc.ents = filter_spans(spans)
            docs.append(doc)
        return docs
