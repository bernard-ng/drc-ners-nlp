import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

import pandas as pd

from core.config.pipeline_config import PipelineConfig
from processing.steps import PipelineStep, NameAnnotation
from processing.ner.ner_name_model import NERNameModel


class NERAnnotationStep(PipelineStep):
    """NER annotation step using trained spaCy model for entity recognition"""

    def __init__(self, pipeline_config: PipelineConfig):
        # Create custom batch config for NER processing
        super().__init__("ner_annotation", pipeline_config)

        self.model_name = "drc_ner_model"
        self.model_path = pipeline_config.paths.models_dir / "drc_ner_model"
        self.ner_trainer = NERNameModel(pipeline_config)
        self.ner_config = pipeline_config.annotation.ner

        # Statistics
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_retry_attempts = 0

        # Load the model
        self._load_ner_model()

    def _load_ner_model(self) -> None:
        """Load the trained NER model"""
        try:
            if self.model_path.exists():
                logging.info(f"Loading NER model from {self.model_path}")
                self.ner_trainer.load(str(self.model_path))
                logging.info("NER model loaded successfully")
            else:
                logging.warning(f"NER model not found at {self.model_path}")
                logging.warning("NER annotation will be skipped. Train the model first.")
                self.ner_trainer.nlp = None
        except Exception as e:
            logging.error(f"Failed to load NER model: {e}")
            self.ner_trainer.nlp = None

    def analyze_name(self, name: str) -> Dict:
        """Analyze a name with retry logic"""
        if self.ner_trainer.nlp is None:
            return {
                "identified_name": None,
                "identified_surname": None,
                "annotated": 0,
                "processing_time": 0,
                "attempts": 0,
                "failed": True,
            }

        for attempt in range(self.ner_config.retry_attempts):
            try:
                start_time = time.time()

                # Get NER predictions
                prediction = self.ner_trainer.predict(name.lower())
                entities = prediction.get('entities', [])

                elapsed_time = time.time() - start_time

                # Extract native names and surnames from entities
                native_parts = []
                surname_parts = []

                for entity in entities:
                    if entity['label'] == 'NATIVE':
                        native_parts.append(entity['text'])
                    elif entity['label'] == 'SURNAME':
                        surname_parts.append(entity['text'])

                # Create annotation result in same format as LLM step
                annotation = NameAnnotation(
                    identified_name=" ".join(native_parts) if native_parts else None,
                    identified_surname=" ".join(surname_parts) if surname_parts else None
                )

                result = {
                    **annotation.model_dump(),
                    "annotated": 1,
                    "processing_time": elapsed_time,
                    "attempts": attempt + 1,
                }

                self.successful_requests += 1
                if attempt > 0:
                    self.total_retry_attempts += attempt

                return result

            except Exception as e:
                logging.warning(
                    f"Error analyzing '{name}' with NER (attempt {attempt + 1}/{self.ner_config.retry_attempts}): {e}"
                )

                # Small delay between retries
                if attempt < self.ner_config.retry_attempts - 1:
                    time.sleep(0.1)

        self.failed_requests += 1
        return {
            "identified_name": None,
            "identified_surname": None,
            "annotated": 0,
            "processing_time": 0,
            "attempts": self.ner_config.retry_attempts,
            "failed": True,
        }

    def process_batch(self, batch: pd.DataFrame, batch_id: int) -> pd.DataFrame:
        """Process batch with NER annotation using same logic as LLM step"""
        unannotated_mask = batch.get("annotated", 0) == 0
        unannotated_entries = batch[unannotated_mask]

        if unannotated_entries.empty:
            logging.info(f"Batch {batch_id}: No entries to annotate")
            return batch

        logging.info(f"Batch {batch_id}: Annotating {len(unannotated_entries)} entries with NER")

        batch = batch.copy()

        # Process with controlled concurrency
        max_workers = self.batch_config.max_workers

        if len(unannotated_entries) == 1 or max_workers == 1:
            # Sequential processing
            for idx, row in unannotated_entries.iterrows():
                result = self.analyze_name(row["name"])
                for field, value in result.items():
                    if field not in ["failed"]:
                        batch.loc[idx, field] = value
        else:
            # Concurrent processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {}

                for idx, row in unannotated_entries.iterrows():
                    future = executor.submit(self.analyze_name, row["name"])
                    future_to_idx[future] = idx

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        for field, value in result.items():
                            if field not in ["failed"]:
                                batch.loc[idx, field] = value
                    except Exception as e:
                        logging.error(f"Failed to process row {idx}: {e}")
                        batch.loc[idx, "annotated"] = 0

        # Ensure proper data types
        batch["annotated"] = pd.to_numeric(batch["annotated"], errors="coerce").fillna(0).astype("Int8")

        return batch
