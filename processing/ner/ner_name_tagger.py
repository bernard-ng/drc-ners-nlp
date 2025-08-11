from typing import Union, Dict, Any, List
import logging


class NERNameTagger:
    def tag_name(self, name: str, probable_native: str, probable_surname: str) -> Union[Dict[str, Any], None]:
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
                pos = name_lower.find(native_word_lower, start_pos)  # Case-insensitive search
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
                if (self._is_word_boundary_match(name, pos, end_pos) and
                    not has_overlap(pos, end_pos)):
                    entities.append((pos, end_pos, 'NATIVE'))
                    used_spans.append((pos, end_pos))
                    break  # Only take the first non-overlapping occurrence

                start_pos = pos + 1

        # Find position of surname in the full name
        if probable_surname and len(probable_surname.strip()) >= 2:
            surname_lower = probable_surname.lower()

            # Find the first occurrence that doesn't overlap
            start_pos = 0
            while True:
                pos = name_lower.find(surname_lower, start_pos)  # Case-insensitive search
                if pos == -1:
                    break

                # Calculate end position correctly - exact match only
                end_pos = pos + len(surname_lower)

                # Double-check that the extracted span matches exactly what we expect
                extracted_text = name[pos:end_pos]  # Get original case text
                if extracted_text.lower() != surname_lower:
                    start_pos = pos + 1
                    continue

                if (self._is_word_boundary_match(name, pos, end_pos) and
                    not has_overlap(pos, end_pos)):
                    entities.append((pos, end_pos, 'SURNAME'))
                    used_spans.append((pos, end_pos))
                    break

                start_pos = pos + 1

        if not entities:
            logging.warning(f"No valid entities found for name: '{name}' with native: '{probable_native}' and surname: '{probable_surname}'")
            return None

        # Sort entities by position and validate
        entities.sort(key=lambda x: x[0])

        # Final validation - ensure no overlaps and valid spans
        validated_entities = []
        for start, end, label in entities:
            # Check bounds
            if not (0 <= start < end <= len(name)):
                logging.warning(f"Invalid span bounds ({start}, {end}) for text length {len(name)}: '{name}'")
                continue

            # Check for overlaps with already validated entities
            if any(start < v_end and end > v_start for v_start, v_end, _ in validated_entities):
                logging.warning(f"Overlapping span ({start}, {end}, '{label}') in '{name}'")
                continue

            # CRITICAL VALIDATION: Check that the span contains only the expected word (no spaces)
            span_text = name[start:end]
            if not span_text or span_text != span_text.strip() or ' ' in span_text:
                logging.warning(f"Span contains spaces or is empty ({start}, {end}) in '{name}': '{span_text}'")
                continue

            validated_entities.append((start, end, label))

        if not validated_entities:
            logging.warning(f"No valid entities after validation for: '{name}'")
            return None

        # Convert to string format that matches the dataset
        entities_str = str(validated_entities)

        return {
            "entities": entities_str,
            "spans": validated_entities  # Keep the original tuples for internal use
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
    def validate_entities(cls, name: str, entities_str: str) -> bool:
        """Validate that entity annotations are correct for a given name"""
        try:
            import ast
            entities = ast.literal_eval(entities_str)

            # Check for overlaps and valid bounds
            sorted_entities = sorted(entities, key=lambda x: x[0])

            for i, (start, end, label) in enumerate(sorted_entities):
                # Check bounds
                if not (0 <= start < end <= len(name)):
                    return False

                # Check for overlaps with next entity
                if i < len(sorted_entities) - 1:
                    next_start = sorted_entities[i + 1][0]
                    if end > next_start:
                        return False

                # Extract the text span and validate it's not empty
                span_text = name[start:end]
                if not span_text.strip():
                    return False

            return True
        except (ValueError, SyntaxError, TypeError):
            return False

    @classmethod
    def extract_entity_text(cls, name: str, entities_str: str) -> Dict[str, List[str]]:
        """Extract the actual text for each entity type"""
        result = {'NATIVE': [], 'SURNAME': []}

        try:
            import ast
            entities = ast.literal_eval(entities_str)

            for start, end, label in entities:
                if 0 <= start < end <= len(name):
                    span_text = name[start:end]
                    if label in result:
                        result[label].append(span_text)

        except (ValueError, SyntaxError, TypeError):
            pass

        return result
