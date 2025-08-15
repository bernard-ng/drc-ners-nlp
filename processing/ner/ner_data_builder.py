import ast
import json
import logging
from pathlib import Path

import pandas as pd
import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans

from core.config import PipelineConfig
from core.utils.data_loader import DataLoader


class NERDataBuilder:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.data_loader = DataLoader(config)

    @staticmethod
    def _parse_entities(series: pd.Series) -> pd.Series:
        """Vectorized parse of entity strings."""

        def _parse(entities_str):
            if not entities_str or entities_str in ["[]", "", "nan"]:
                return []
            entities_str = str(entities_str).strip()
            try:
                if entities_str.startswith("[(") and entities_str.endswith(")]"):
                    return ast.literal_eval(entities_str)
                elif entities_str.startswith("[[") and entities_str.endswith("]]"):
                    return [tuple(e) for e in ast.literal_eval(entities_str)]
                elif entities_str.startswith("[{") and entities_str.endswith("}]"):
                    return [(e["start"], e["end"], e["label"]) for e in json.loads(entities_str)]
                else:
                    parsed = ast.literal_eval(entities_str)
                    return [
                        tuple(e) for e in parsed if isinstance(e, (list, tuple)) and len(e) == 3
                    ]
            except (ValueError, SyntaxError, json.JSONDecodeError):
                return []

        return series.map(_parse)

    @staticmethod
    def _validate_entities(texts: pd.Series, entities_series: pd.Series) -> pd.Series:
        """Vectorized entity validation."""

        def _validate(text, entities):
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
            # remove overlaps
            filtered, last_end = [], -1
            for s, e, l in valid:
                if s >= last_end:
                    filtered.append((s, e, l))
                    last_end = e
            return filtered

        return pd.Series(map(_validate, texts, entities_series), index=texts.index)

    @staticmethod
    def _create_docs(nlp, texts, entities):
        """Batch create spaCy Docs."""
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

    def build(self) -> int:
        filepath = self.config.paths.get_data_path(self.config.data.output_files["engineered"])
        df = self.data_loader.load_csv_complete(filepath)
        df = df[["name", "ner_tagged", "ner_entities"]]

        # Filter early
        ner_df = df.loc[df["ner_tagged"] == 1, ["name", "ner_entities"]]
        if ner_df.empty:
            logging.error("No NER tagged data found")
            return 1

        total_rows = len(df)
        del df  # No need to keep in memory

        logging.info(f"Found {len(ner_df)} NER tagged entries")
        nlp = spacy.blank("fr")

        # Vectorized parsing + validation
        parsed_entities = self._parse_entities(ner_df["ner_entities"])
        validated_entities = self._validate_entities(ner_df["name"], parsed_entities)

        # Drop rows with no valid entities
        mask = validated_entities.map(bool)
        ner_df = ner_df.loc[mask]
        validated_entities = validated_entities.loc[mask]

        if ner_df.empty:
            logging.error("No valid training examples after validation")
            return 1

        # Prepare training data
        training_data = list(
            zip(ner_df["name"].tolist(), [{"entities": ents} for ents in validated_entities])
        )

        # Create spaCy DocBin in batch
        docs = self._create_docs(nlp, ner_df["name"].tolist(), validated_entities.tolist())
        doc_bin = DocBin(docs=docs)

        # Save
        json_path = self.config.paths.get_data_path(self.config.data.output_files["ner_data"])
        spacy_path = self.config.paths.get_data_path(self.config.data.output_files["ner_spacy"])

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(training_data, f, ensure_ascii=False, separators=(",", ":"))
        doc_bin.to_disk(spacy_path)

        logging.info(f"Processed: {len(training_data)}, Skipped: {total_rows - len(training_data)}")
        logging.info(f"Saved NER JSON to {json_path}")
        logging.info(f"Saved NER spacy to {spacy_path}")
        return 0
