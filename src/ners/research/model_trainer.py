import json
import logging
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd

from ners.core.config import get_config
from ners.core.utils.data_loader import DataLoader
from ners.research.experiment import FeatureType, ExperimentConfig
from ners.research.experiment.experiment_runner import ExperimentRunner
from ners.research.experiment.experiment_tracker import ExperimentTracker
from ners.research.model_registry import MODEL_REGISTRY


class ModelTrainer:
    """Comprehensive model training and artifact management"""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.data_loader = DataLoader(self.config)
        self.experiment_runner = ExperimentRunner(self.config)
        self.experiment_tracker = ExperimentTracker(self.config)

        # Setup model artifacts directory
        self.models_dir = self.config.paths.models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def train_single_model(
        self,
        model_name: str,
        model_type: str = "logistic_regression",
        features: List[str] = None,
        model_params: Dict[str, Any] = None,
        tags: List[str] = None,
        save_artifacts: bool = True,
    ) -> str:
        """
        Train a single model and save its artifacts.
        Returns the experiment ID.
        """
        logging.info(f"Training {model_type} model: {model_name}")

        if features is None:
            features = ["full_name"]
        feature_types = [FeatureType(f) for f in features]

        # Prepare tags - combine default tags with template tags
        default_tags = ["training", model_type]
        experiment_tags = default_tags + (tags or [])

        # Create experiment configuration
        config = ExperimentConfig(
            name=model_name,
            description=f"Training {model_type} model with features: {', '.join(features)}",
            model_type=model_type,
            features=feature_types,
            model_params=model_params or {},
            tags=experiment_tags,
        )

        # Run experiment
        experiment_id = self.experiment_runner.run_experiment(config)
        experiment = self.experiment_tracker.get_experiment(experiment_id)

        if experiment and experiment.test_metrics:
            logging.info("Training completed successfully!")
            logging.info(f"Experiment ID: {experiment_id}")
            logging.info(
                f"Test Accuracy: {experiment.test_metrics.get('accuracy', 0):.4f}"
            )
            logging.info(f"Test F1-Score: {experiment.test_metrics.get('f1', 0):.4f}")

            if save_artifacts:
                self.save_model_artifacts(experiment_id)

        return experiment_id

    def train_multiple_models(
        self, base_name: str, model_configs: List[Dict[str, Any]], save_all: bool = True
    ) -> List[str]:
        """
        Train multiple models with different configurations.
        """
        logging.info(f"Training {len(model_configs)} models...")

        experiment_ids = []

        for i, config in enumerate(model_configs):
            model_name = f"{base_name}_{config['model_type']}_{i + 1}"

            try:
                exp_id = self.train_single_model(
                    model_name=model_name,
                    model_type=config["model_type"],
                    features=config.get("features", ["full_name"]),
                    model_params=config.get("model_params", {}),
                    save_artifacts=save_all,
                )
                experiment_ids.append(exp_id)

            except Exception as e:
                logging.error(f"Failed to train {model_name}: {e}")
                continue

        logging.info(f"Completed training {len(experiment_ids)} models successfully")
        return experiment_ids

    def save_model_artifacts(self, experiment_id: str) -> Dict[str, str]:
        """
        Save model artifacts in a structured way for easy loading.
        Returns paths to saved artifacts.
        """
        experiment = self.experiment_tracker.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        # Create model-specific directory
        model_dir = self.models_dir / experiment_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # Load the trained model
        trained_model = self.experiment_runner.load_experiment_model(experiment_id)
        if not trained_model:
            raise ValueError(f"Could not load model for experiment {experiment_id}")

        # Save complete model with joblib
        model_path = model_dir / "complete_model.joblib"
        trained_model.save(str(model_path))

        # Save model configuration
        config_path = model_dir / "model_config.json"
        with open(config_path, "w") as f:
            import json

            json.dump(experiment.config.to_dict(), f, indent=2)

        # Save experiment results
        results_path = model_dir / "experiment_results.json"
        with open(results_path, "w") as f:
            json.dump(experiment.to_dict(), f, indent=2, default=str)

        # Generate and save learning curves
        learning_curve_path = None
        training_history_path = None

        try:
            # Load data for learning curve generation
            data_path = self.config.paths.get_data_path(
                self.config.data.output_files["featured"]
            )
            if data_path.exists():
                df = self.data_loader.load_csv_complete(data_path)

                # Generate learning curve
                logging.info("Generating learning curve...")
                trained_model.generate_learning_curve(
                    df, df[experiment.config.target_column]
                )

                # Plot and save learning curve
                learning_curve_path = model_dir / "learning_curve.png"
                trained_model.plot_learning_curve(str(learning_curve_path))

                # Plot and save training history (for neural networks)
                if trained_model.training_history:
                    training_history_path = model_dir / "training_history.png"
                    trained_model.plot_training_history(str(training_history_path))

                # Save learning curve data as JSON
                learning_data_path = model_dir / "learning_curve_data.json"
                with open(learning_data_path, "w") as f:
                    json.dump(trained_model.learning_curve_data, f, indent=2)

                # Save training history data as JSON
                if trained_model.training_history:
                    history_data_path = model_dir / "training_history_data.json"
                    with open(history_data_path, "w") as f:
                        json.dump(trained_model.training_history, f, indent=2)

        except Exception as e:
            logging.warning(f"Could not generate learning curves: {e}")

        # Save artifacts metadata
        metadata = {
            "experiment_id": experiment_id,
            "model_name": experiment.config.name,
            "model_type": experiment.config.model_type,
            "features": [f.value for f in experiment.config.features],
            "training_date": datetime.now().isoformat(),
            "test_accuracy": experiment.test_metrics.get("accuracy", 0),
            "test_f1": experiment.test_metrics.get("f1", 0),
            "model_path": str(model_path),
            "config_path": str(config_path),
            "results_path": str(results_path),
            "learning_curve_plot": str(learning_curve_path)
            if learning_curve_path
            else None,
            "training_history_plot": str(training_history_path)
            if training_history_path
            else None,
            "has_learning_curve": bool(trained_model.learning_curve_data),
            "has_training_history": bool(trained_model.training_history),
        }

        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logging.info(f"Model artifacts saved to: {model_dir}")
        logging.info(f"   - Complete model: {model_path.name}")
        logging.info(f"   - Configuration: {config_path.name}")
        logging.info(f"   - Results: {results_path.name}")
        logging.info(f"   - Metadata: {metadata_path.name}")

        if learning_curve_path and learning_curve_path.exists():
            logging.info(f"   - Learning curve: {learning_curve_path.name}")

        if training_history_path and training_history_path.exists():
            logging.info(f"   - Training history: {training_history_path.name}")

        return {
            "model_dir": str(model_dir),
            "model_path": str(model_path),
            "config_path": str(config_path),
            "results_path": str(results_path),
            "metadata_path": str(metadata_path),
            "learning_curve_plot": str(learning_curve_path)
            if learning_curve_path
            else None,
            "training_history_plot": str(training_history_path)
            if training_history_path
            else None,
        }

    def load_trained_model(self, experiment_id: str):
        """
        Load a previously trained model from artifacts.
        """
        model_dir = self.models_dir / experiment_id
        model_path = model_dir / "complete_model.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifacts not found for experiment {experiment_id}"
            )

        # Load the model class dynamically
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        model_type = metadata["model_type"]
        model_class = MODEL_REGISTRY[model_type]

        # Load the complete model
        loaded_model = model_class.load(str(model_path))

        logging.info(f"Loaded model: {metadata['model_name']}")
        logging.info(f"   Type: {model_type}")
        logging.info(f"   Accuracy: {metadata['test_accuracy']:.4f}")

        return loaded_model

    def list_saved_models(self) -> pd.DataFrame:
        """
        List all saved model artifacts.
        """
        models_data = []

        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                metadata_path = model_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                        models_data.append(metadata)
                    except Exception as e:
                        logging.warning(
                            f"Could not read metadata for {model_dir.name}: {e}"
                        )

        if not models_data:
            logging.info("No saved models found.")
            return pd.DataFrame()

        df = pd.DataFrame(models_data)

        # Format the display
        display_columns = [
            "model_name",
            "model_type",
            "features",
            "test_accuracy",
            "test_f1",
            "training_date",
        ]
        available_columns = [col for col in display_columns if col in df.columns]

        return df[available_columns].sort_values("training_date", ascending=False)
