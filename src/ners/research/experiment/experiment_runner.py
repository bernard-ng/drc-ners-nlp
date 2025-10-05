import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from ners.core.config import PipelineConfig
from ners.core.utils.data_loader import DataLoader
from ners.research.base_model import BaseModel
from ners.research.experiment import (
    ExperimentConfig,
    ExperimentStatus,
    calculate_metrics,
)
from ners.research.experiment.experiment_tracker import ExperimentTracker
from ners.research.model_registry import create_model


class ExperimentRunner:
    """Runs and manages experiments"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.tracker = ExperimentTracker(self.config)
        self.data_loader = DataLoader(self.config)

    def run_experiment(self, experiment_config: ExperimentConfig) -> str:
        """Run a single experiment and return experiment ID"""
        # Create experiment
        experiment_id = self.tracker.create_experiment(experiment_config)

        try:
            logging.info(f"Starting experiment: {experiment_id}")
            self.tracker.update_experiment(
                experiment_id, status=ExperimentStatus.RUNNING
            )

            # Load data
            filepath = self.config.paths.get_data_path(
                self.config.data.output_files["featured"]
            )
            df = self.data_loader.load_csv_complete(filepath)

            # Apply data filters if specified
            df = self._apply_data_filters(df, experiment_config)

            # Prepare target variable
            y = df[experiment_config.target_column]
            X = df

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=experiment_config.test_size,
                random_state=experiment_config.random_seed,
                stratify=y,
            )

            # Create and train model
            model = create_model(experiment_config)
            model.fit(X_train, y_train)

            # Make predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            # Calculate metrics
            train_metrics = calculate_metrics(
                y_train, train_pred, experiment_config.metrics
            )
            test_metrics = calculate_metrics(
                y_test, test_pred, experiment_config.metrics
            )

            # Cross-validation if requested
            cv_metrics = {}
            if experiment_config.cross_validation_folds > 1:
                cv_metrics = model.cross_validate(
                    X_train, y_train, experiment_config.cross_validation_folds
                )

            # Additional analysis
            conf_matrix = confusion_matrix(y_test, test_pred).tolist()
            feature_importance = model.get_feature_importance()

            # Create prediction examples
            prediction_examples = self._create_prediction_examples(
                X_test, y_test, test_pred, model, n_examples=10
            )

            # Calculate class distribution
            class_distribution = y.value_counts().to_dict()

            # Save model
            model_path = self._save_model(model, experiment_id)

            # Update experiment with results
            self.tracker.update_experiment(
                experiment_id,
                status=ExperimentStatus.COMPLETED,
                end_time=datetime.now(),
                model_path=str(model_path),
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                cv_metrics=cv_metrics,
                confusion_matrix=conf_matrix,
                feature_importance=feature_importance,
                prediction_examples=prediction_examples,
                train_size=len(X_train),
                test_size=len(X_test),
                class_distribution=class_distribution,
            )

            logging.info(f"Experiment {experiment_id} completed successfully")
            logging.info(f"Test accuracy: {test_metrics.get('accuracy', 'N/A'):.4f}")

            return experiment_id

        except Exception as e:
            logging.error(f"Experiment {experiment_id} failed: {str(e)}")
            self.tracker.update_experiment(
                experiment_id,
                status=ExperimentStatus.FAILED,
                end_time=datetime.now(),
                error_message=str(e),
            )
            raise

    def run_experiment_batch(self, experiments: List[ExperimentConfig]) -> List[str]:
        """Run multiple experiments"""
        experiment_ids = []

        for i, config in enumerate(experiments):
            logging.info(
                f"Running experiment {i + 1}/{len(experiments)}: {config.name}"
            )
            try:
                exp_id = self.run_experiment(config)
                experiment_ids.append(exp_id)
            except Exception as e:
                logging.error(f"Failed to run experiment {config.name}: {e}")
                continue

        return experiment_ids

    @classmethod
    def _apply_data_filters(
        cls, df: pd.DataFrame, config: ExperimentConfig
    ) -> pd.DataFrame:
        """Apply data filters specified in experiment config"""
        filtered_df = df.copy()

        # Apply training data filters
        if config.train_data_filter:
            for column, criteria in config.train_data_filter.items():
                if column in filtered_df.columns:
                    if isinstance(criteria, list):
                        filtered_df = filtered_df[filtered_df[column].isin(criteria)]
                    elif isinstance(criteria, dict):
                        if "min" in criteria:
                            filtered_df = filtered_df[
                                filtered_df[column] >= criteria["min"]
                            ]
                        if "max" in criteria:
                            filtered_df = filtered_df[
                                filtered_df[column] <= criteria["max"]
                            ]
                    else:
                        filtered_df = filtered_df[filtered_df[column] == criteria]

        return filtered_df

    @classmethod
    def _create_prediction_examples(
        cls,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        predictions: np.ndarray,
        model: BaseModel,
        n_examples: int = 10,
    ) -> List[Dict]:
        """Create prediction examples for analysis"""
        examples = []

        # Get both correct and incorrect predictions
        correct_mask = y_test == predictions
        incorrect_indices = X_test[~correct_mask].index[: n_examples // 2]
        correct_indices = X_test[correct_mask].index[: n_examples // 2]

        sample_indices = list(incorrect_indices) + list(correct_indices)

        for idx in sample_indices[:n_examples]:
            example = {
                "name": X_test.loc[idx, "name"] if "name" in X_test.columns else "N/A",
                "true_label": y_test.loc[idx],
                "predicted_label": predictions[X_test.index.get_loc(idx)],
                "correct": y_test.loc[idx] == predictions[X_test.index.get_loc(idx)],
            }

            # Add probability if available
            if model.architecture == "traditional":
                proba = model.predict_proba(X_test.loc[[idx]])
                example["prediction_confidence"] = float(proba.max())

            examples.append(example)

        return examples

    def _save_model(self, model: BaseModel, experiment_id: str) -> Path:
        """Save trained model"""
        model_dir = self.config.paths.models_dir / "experiments" / experiment_id
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "model.joblib"
        model.save(str(model_path))

        return model_path

    def load_experiment_model(self, experiment_id: str) -> Optional[BaseModel]:
        """Load a model from a completed experiment"""
        experiment = self.tracker.get_experiment(experiment_id)

        if experiment and experiment.model_path:
            try:
                # Load the saved model data Recreate the model instance using the saved config
                model_data = joblib.load(experiment.model_path)
                config = ExperimentConfig.from_dict(model_data["config"])
                model = create_model(config)

                # Restore the saved state
                model.model = model_data["model"]
                model.feature_extractor = model_data["feature_extractor"]
                model.label_encoder = model_data["label_encoder"]
                model.tokenizer = model_data.get("tokenizer")
                model.is_fitted = model_data["is_fitted"]
                model.training_history = model_data.get("training_history", {})
                model.learning_curve_data = model_data.get("learning_curve_data", {})

                # Restore vectorizers and encoders for models that use them (like XGBoost)
                if "vectorizers" in model_data and hasattr(model, "vectorizers"):
                    model.vectorizers = model_data["vectorizers"]
                if "label_encoders" in model_data and hasattr(model, "label_encoders"):
                    model.label_encoders = model_data["label_encoders"]

                return model

            except Exception as e:
                logging.error(
                    f"Failed to load model for experiment {experiment_id}: {e}"
                )
                return None

        return None

    def compare_experiments(
        self, experiment_ids: List[str], metric: str = "accuracy"
    ) -> pd.DataFrame:
        """Compare experiments and return analysis"""
        comparison_df = self.tracker.compare_experiments(experiment_ids)

        if f"test_{metric}" in comparison_df.columns:
            comparison_df = comparison_df.sort_values(f"test_{metric}", ascending=False)

        return comparison_df

    def get_feature_analysis(self, experiment_id: str) -> Optional[pd.DataFrame]:
        """Get feature importance analysis for an experiment"""
        experiment = self.tracker.get_experiment(experiment_id)

        if experiment and experiment.feature_importance:
            importance_df = pd.DataFrame(
                [
                    {"feature": feature, "importance": importance}
                    for feature, importance in experiment.feature_importance.items()
                ]
            )
            return importance_df.sort_values("importance", ascending=False)

        return None
