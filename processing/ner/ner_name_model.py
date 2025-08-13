import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import spacy
from spacy.training import Example
from spacy.util import minibatch

from core.config.pipeline_config import PipelineConfig


class NERNameModel:
    """NER model trainer using spaCy for DRC names entity recognition"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.nlp = None
        self.ner = None
        self.model_path = None
        self.training_stats = {}

    def create_blank_model(self, language: str = "fr") -> None:
        """Create a blank spaCy model with NER pipeline"""
        logging.info(f"Creating blank {language} model for NER training")

        # Create blank model - French tokenizer works well for DRC names
        self.nlp = spacy.blank(language)

        # Add NER pipeline component
        if "ner" not in self.nlp.pipe_names:
            self.ner = self.nlp.add_pipe("ner")
        else:
            self.ner = self.nlp.get_pipe("ner")

        # Add our custom labels
        self.ner.add_label("NATIVE")
        self.ner.add_label("SURNAME")

        logging.info("Blank model created with NATIVE and SURNAME labels")

    @classmethod
    def load_data(cls, data_path: str) -> List[Tuple[str, Dict]]:
        """Load training data from JSON file - compatible with NERNameTagger output format"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Training data not found at {data_path}")

        logging.info(f"Loading training data from {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Validate and clean training data
        valid_data = []
        skipped_count = 0

        for i, item in enumerate(raw_data):
            try:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    logging.warning(
                        f"Skipping invalid training example format at index {i}: {item}"
                    )
                    skipped_count += 1
                    continue

                text, annotations = item

                # Validate text
                if not isinstance(text, str) or not text.strip():
                    logging.warning(f"Skipping invalid text at index {i}: {repr(text)}")
                    skipped_count += 1
                    continue

                # Handle different annotation formats from NERNameTagger
                if not isinstance(annotations, dict) or "entities" not in annotations:
                    logging.warning(f"Skipping invalid annotations at index {i}: {annotations}")
                    skipped_count += 1
                    continue

                entities_raw = annotations["entities"]

                # Parse entities - handle both string and list formats from tagger
                if isinstance(entities_raw, str):
                    # String format from tagger: "[(0, 6, 'NATIVE'), ...]"
                    try:
                        import ast

                        entities = ast.literal_eval(entities_raw)
                        if not isinstance(entities, list):
                            logging.warning(
                                f"Parsed entities is not a list at index {i}: {entities}"
                            )
                            skipped_count += 1
                            continue
                    except (ValueError, SyntaxError) as e:
                        logging.warning(
                            f"Failed to parse entity string at index {i}: {entities_raw} ({e})"
                        )
                        skipped_count += 1
                        continue
                elif isinstance(entities_raw, list):
                    # Already in list format
                    entities = entities_raw
                else:
                    logging.warning(
                        f"Skipping invalid entities format at index {i}: {entities_raw}"
                    )
                    skipped_count += 1
                    continue

                # Validate each entity
                valid_entities = []
                for entity in entities:
                    if not isinstance(entity, (list, tuple)) or len(entity) != 3:
                        logging.warning(f"Skipping invalid entity format in '{text}': {entity}")
                        continue

                    start, end, label = entity

                    # Validate entity components
                    if (
                        not isinstance(start, int)
                        or not isinstance(end, int)
                        or not isinstance(label, str)
                        or start >= end
                        or start < 0
                        or end > len(text)
                    ):
                        logging.warning(f"Skipping invalid entity bounds in '{text}': {entity}")
                        continue

                    # Check for overlaps with already validated entities
                    has_overlap = any(
                        start < v_end and end > v_start for v_start, v_end, _ in valid_entities
                    )

                    if has_overlap:
                        logging.warning(f"Skipping overlapping entity in '{text}': {entity}")
                        continue

                    # Validate that the span doesn't contain spaces (matching tagger validation)
                    span_text = text[start:end]
                    if not span_text or span_text != span_text.strip() or " " in span_text:
                        logging.warning(
                            f"Skipping entity with spaces in '{text}': {entity} -> '{span_text}'"
                        )
                        continue

                    valid_entities.append((start, end, label))

                if not valid_entities:
                    logging.warning(f"Skipping training example with no valid entities: '{text}'")
                    skipped_count += 1
                    continue

                # Sort entities by start position
                valid_entities.sort(key=lambda x: x[0])
                valid_data.append((text.strip(), {"entities": valid_entities}))

            except Exception as e:
                logging.error(f"Error processing training example at index {i}: {e}")
                skipped_count += 1
                continue

        logging.info(
            f"Loaded {len(valid_data)} valid training examples, skipped {skipped_count} invalid ones"
        )

        if not valid_data:
            raise ValueError("No valid training examples found in the data")

        return valid_data

    def train(
        self,
        data: List[Tuple[str, Dict]],
        epochs: int = 5,
        batch_size: int = 16,
        dropout_rate: float = 0.2,
    ) -> None:
        """Train the NER model"""
        logging.info(f"Starting NER training with {len(data)} examples")
        logging.info(
            f"Training parameters: epochs={epochs}, batch_size={batch_size}, dropout={dropout_rate}"
        )

        if self.nlp is None:
            raise ValueError("Model not initialized. Call create_blank_model() first.")

        # Initialize the model
        self.nlp.initialize()

        # Training loop
        losses_history = []

        for epoch in range(epochs):
            losses = {}

            # Create training examples
            examples = []
            for text, annotations in data:
                doc = self.nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)
                logging.info(
                    f"Training example: {text[:30]}... with entities {annotations.get('entities', [])}"
                )

            # Train in batches
            batches = minibatch(examples, size=batch_size)
            for batch in batches:
                self.nlp.update(
                    batch, losses=losses, drop=dropout_rate, sgd=self.nlp.create_optimizer()
                )
                logging.info(f"Training batch with {len(batch)} examples, current losses: {losses}")

            epoch_loss = losses.get("ner", 0)
            losses_history.append(epoch_loss)
            logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

        # Store training statistics
        self.training_stats = {
            "epochs": epochs,
            "final_loss": losses_history[-1] if losses_history else 0,
            "training_examples": len(data),
            "loss_history": losses_history,
            "batch_size": batch_size,
            "dropout_rate": dropout_rate,
        }

        logging.info(f"Training completed. Final loss: {self.training_stats['final_loss']:.4f}")

    def evaluate(self, test_data: List[Tuple[str, Dict]]) -> Dict[str, Any]:
        """Evaluate the trained model on test data"""
        if self.nlp is None:
            raise ValueError("Model not trained. Call train_model() first.")

        logging.info(f"Evaluating model on {len(test_data)} test examples")

        total_examples = len(test_data)
        correct_entities = 0
        predicted_entities = 0
        actual_entities = 0

        entity_stats = {
            "NATIVE": {"tp": 0, "fp": 0, "fn": 0},
            "SURNAME": {"tp": 0, "fp": 0, "fn": 0},
        }

        for text, annotations in test_data:
            # Get actual entities
            actual_ents = set()
            for start, end, label in annotations.get("entities", []):
                actual_ents.add((start, end, label))
                actual_entities += 1

            # Get predicted entities
            doc = self.nlp(text)
            predicted_ents = set()
            for ent in doc.ents:
                predicted_ents.add((ent.start_char, ent.end_char, ent.label_))
                predicted_entities += 1

            # Calculate matches
            matches = actual_ents.intersection(predicted_ents)
            correct_entities += len(matches)

            # Update per-label statistics
            for start, end, label in actual_ents:
                if (start, end, label) in predicted_ents:
                    entity_stats[label]["tp"] += 1
                else:
                    entity_stats[label]["fn"] += 1

            for start, end, label in predicted_ents:
                if (start, end, label) not in actual_ents:
                    entity_stats[label]["fp"] += 1

        # Calculate overall metrics
        precision = correct_entities / predicted_entities if predicted_entities > 0 else 0
        recall = correct_entities / actual_entities if actual_entities > 0 else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        )

        # Calculate per-label metrics
        label_metrics = {}
        for label, stats in entity_stats.items():
            tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
            label_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            label_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            label_f1 = (
                (2 * (label_precision * label_recall) / (label_precision + label_recall))
                if (label_precision + label_recall) > 0
                else 0
            )

            label_metrics[label] = {
                "precision": label_precision,
                "recall": label_recall,
                "f1_score": label_f1,
                "support": tp + fn,
            }

        evaluation_results = {
            "overall": {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "total_examples": total_examples,
                "correct_entities": correct_entities,
                "predicted_entities": predicted_entities,
                "actual_entities": actual_entities,
            },
            "by_label": label_metrics,
        }

        logging.info(f"NER Evaluation completed. Overall F1: {f1_score:.4f}")
        return evaluation_results

    def save(self, model_name: str = "drc_ner_model") -> str:
        """Save the trained model"""
        if self.nlp is None:
            raise ValueError("No model to save. Train a model first.")

        # Create model directory
        model_dir = self.config.paths.models_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save the model
        self.nlp.to_disk(model_dir)
        self.model_path = str(model_dir)

        # Save training statistics
        stats_path = model_dir / "training_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(self.training_stats, f, indent=2)

        logging.info(f"NER Model saved to {model_dir}")
        return self.model_path

    def load(self, model_path: str) -> None:
        """Load a trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        logging.info(f"Loading model from {model_path}")
        self.nlp = spacy.load(model_path)
        self.ner = self.nlp.get_pipe("ner")
        self.model_path = model_path

        # Load training statistics if available
        stats_path = Path(model_path) / "training_stats.json"
        if stats_path.exists():
            with open(stats_path, "r", encoding="utf-8") as f:
                self.training_stats = json.load(f)

        logging.info("NER Model loaded successfully")

    def predict(self, text: str) -> Dict[str, Any]:
        """Make predictions on a single text"""
        if self.nlp is None:
            raise ValueError("No model loaded. Load or train a model first.")

        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            entities.append(
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": getattr(ent, "score", None),  # If confidence scores are available
                }
            )

        return {"text": text, "entities": entities}
