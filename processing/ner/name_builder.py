import json
import logging

import spacy
from spacy.tokens import DocBin

from core.config import PipelineConfig
from core.utils.data_loader import DataLoader
from .name_tagger import NameTagger


class NameBuilder:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.data_loader = DataLoader(config)
        self.tagger = NameTagger()

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

        # Use NERNameTagger for parsing and validation
        parsed_entities = NameTagger.parse_entities(ner_df["ner_entities"])
        validated_entities = NameTagger.validate_entities(ner_df["name"], parsed_entities)

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

        # Use NERNameTagger to create spaCy DocBin
        docs = NameTagger.create_docs(nlp, ner_df["name"].tolist(), validated_entities.tolist())
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
