import ast
import json
import logging
from pathlib import Path

import pandas as pd
import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans

from core.config import PipelineConfig
from core.utils import get_data_file_path


class NERDataBuilder:
    def __init__(self, config: PipelineConfig):
        self.config = config

    @classmethod
    def parse_entities(cls, entities_str):
        """Parse entity string (tuple format or JSON) into spaCy-style tuples."""
        if not entities_str or entities_str in ["[]", "", "nan"]:
            return []

        entities_str = str(entities_str).strip()

        # Handle different formats
        try:
            # Try to parse as Python literal (tuples or lists)
            if entities_str.startswith("[(") and entities_str.endswith(")]"):
                # Standard tuple format: [(0, 6, 'NATIVE'), ...]
                return ast.literal_eval(entities_str)
            elif entities_str.startswith("[[") and entities_str.endswith("]]"):
                # Nested list format: [[0, 6, 'NATIVE'], ...]
                nested_list = ast.literal_eval(entities_str)
                return [(start, end, label) for start, end, label in nested_list]
            elif entities_str.startswith("[{") and entities_str.endswith("}]"):
                # JSON format: [{"start": 0, "end": 6, "label": "NATIVE"}, ...]
                json_entities = json.loads(entities_str)
                return [(e["start"], e["end"], e["label"]) for e in json_entities]
            else:
                # Try general ast.literal_eval for other formats
                parsed = ast.literal_eval(entities_str)
                if isinstance(parsed, list):
                    # Convert any list format to tuples
                    result = []
                    for item in parsed:
                        if isinstance(item, (list, tuple)) and len(item) == 3:
                            result.append((item[0], item[1], item[2]))
                    return result

        except (ValueError, SyntaxError, json.JSONDecodeError) as e:
            logging.warning(f"Failed to parse entities: {entities_str} ({e})")
            return []

        logging.warning(f"Unknown entity format: {entities_str}")
        return []

    @classmethod
    def validate_entities(cls, entities, text):
        """Validate and sort entity tuples, removing overlaps and invalid spans."""
        if not entities or not text:
            return []

        text = str(text).strip()
        if not text:
            return []

        # Filter out invalid entities
        valid_entities = []
        for entity in entities:
            if not isinstance(entity, (list, tuple)) or len(entity) != 3:
                logging.warning(f"Invalid entity format: {entity}")
                continue

            start, end, label = entity

            # Ensure start/end are integers
            try:
                start = int(start)
                end = int(end)
            except (ValueError, TypeError):
                logging.warning(f"Invalid start/end positions: {entity}")
                continue

            # Ensure label is string
            if not isinstance(label, str):
                logging.warning(f"Invalid label type: {entity}")
                continue

            # Check bounds
            if not (0 <= start < end <= len(text)):
                logging.warning(f"Entity span out of bounds: {entity} for text '{text}' (length {len(text)})")
                continue

            # Check that span contains actual text
            span_text = text[start:end].strip()
            if not span_text:
                logging.warning(f"Empty span: {entity} in text '{text}'")
                continue

            valid_entities.append((start, end, label))

        if not valid_entities:
            return []

        # Sort by start position
        valid_entities.sort(key=lambda x: (x[0], x[1]))

        # Remove overlapping entities (keep the first one)
        filtered = []
        for start, end, label in valid_entities:
            # Check for overlap with already added entities
            has_overlap = False
            for e_start, e_end, _ in filtered:
                if not (end <= e_start or start >= e_end):
                    has_overlap = True
                    logging.warning(
                        f"Removing overlapping entity ({start}, {end}, '{label}') "
                        f"conflicts with ({e_start}, {e_end}) in '{text}'"
                    )
                    break

            if not has_overlap:
                filtered.append((start, end, label))

        return filtered

    @classmethod
    def create_doc(cls, text, entities, nlp):
        """Create a spaCy Doc object with entities added."""
        doc = nlp(text)
        ents = []

        for start, end, label in entities:
            span = doc.char_span(start, end, label=label, alignment_mode="contract") \
                   or doc.char_span(start, end, label=label, alignment_mode="strict")
            if span:
                ents.append(span)
            else:
                logging.warning(f"Could not create span ({start}, {end}, '{label}') in '{text}'")

        doc.ents = filter_spans(ents) if ents else []
        return doc

    def build(self, data: pd.DataFrame = None) -> int:
        """Build the dataset for NER training."""
        logging.info("Building dataset for NER training")
        try:
            df = pd.read_csv(get_data_file_path("names_featured.csv", self.config)) \
                if data is None \
                else data

            ner_df = df[df["ner_tagged"] == 1].copy()
            if ner_df.empty:
                logging.error("No NER tagged data found in the CSV")
                return 1

            logging.info(f"Found {len(ner_df)} NER tagged entries")
            nlp = spacy.blank("fr")
            doc_bin, training_data = DocBin(), []
            processed_count, skipped_count = 0, 0

            for _, row in ner_df.iterrows():
                text = str(row.get("name", "")).strip()
                if not text:
                    continue

                entities = self.parse_entities(row.get("ner_entities", "[]"))
                entities = self.validate_entities(entities, text)

                training_data.append((text, {"entities": entities}))
                try:
                    doc_bin.add(self.create_doc(text, entities, nlp))
                    processed_count += 1
                except Exception as e:
                    logging.error(f"Error processing '{text}': {e}")
                    skipped_count += 1

            if not training_data:
                logging.error("No valid training examples generated")
                return 1

            json_path = Path(self.config.paths.data_dir) / self.config.data.output_files["ner_data"]
            spacy_path = Path(self.config.paths.data_dir) / self.config.data.output_files["ner_spacy"]

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(training_data, f, ensure_ascii=False, indent=None)
            doc_bin.to_disk(spacy_path)

            logging.info(f"Processed: {processed_count}, Skipped: {skipped_count}")
            logging.info(f"Saved NER data in json format to {json_path}")
            logging.info(f"Saved NER data in spaCy format to {spacy_path}")
            return 0

        except Exception as e:
            logging.error(f"Failed to build NER dataset: {e}", exc_info=True)
            return 1
