#!.venv/bin/python3
import logging
import argparse

from research.model_trainer import ModelTrainer


def train_baseline_models():
    """
    Quick function to train all baseline models and save artifacts.
    """
    logger = logging.getLogger(__name__)
    logger.info("Training Baseline Models with Artifact Saving")

    trainer = ModelTrainer()

    # Define baseline model configurations
    baseline_configs = [
        {
            "model_type": "logistic_regression",
            "features": ["full_name"],
            "model_params": {"ngram_range": [2, 5], "max_features": 10000},
        },
        {
            "model_type": "logistic_regression",
            "features": ["native_name"],
            "model_params": {"ngram_range": [2, 4], "max_features": 5000},
        },
        {
            "model_type": "logistic_regression",
            "features": ["surname"],
            "model_params": {"ngram_range": [2, 4], "max_features": 5000},
        },
        {
            "model_type": "random_forest",
            "features": ["name_length", "word_count", "province"],
            "model_params": {"n_estimators": 100, "max_depth": 10},
        },
        {
            "model_type": "svm",
            "features": ["full_name"],
            "model_params": {"kernel": "rbf", "C": 1.0},
        },
        {"model_type": "naive_bayes", "features": ["full_name"], "model_params": {"alpha": 1.0}},
    ]

    # Train all baseline models
    experiment_ids = trainer.train_multiple_models("baseline", baseline_configs)

    # Show summary
    logger.info(f"\n Training Summary:")
    for exp_id in experiment_ids:
        experiment = trainer.experiment_tracker.get_experiment(exp_id)
        if experiment:
            acc = experiment.test_metrics.get("accuracy", 0)
            logger.info(f"   {experiment.config.name}: {acc:.4f} accuracy")

    return experiment_ids


def train_neural_networks():
    """
    Train neural network models with proper parameters.
    """

    logging.info("Training Neural Network Models")

    trainer = ModelTrainer()

    neural_configs = [
        {
            "model_type": "lstm",
            "features": ["full_name"],
            "model_params": {
                "embedding_dim": 64,
                "lstm_units": 32,
                "epochs": 10,
                "batch_size": 64,
                "max_len": 6,
            },
        },
        {
            "model_type": "cnn",
            "features": ["full_name"],
            "model_params": {
                "embedding_dim": 64,
                "filters": 64,
                "kernel_size": 3,
                "epochs": 10,
                "batch_size": 64,
                "max_len": 20,  # Character level
            },
        },
        {
            "model_type": "transformer",
            "features": ["full_name"],
            "model_params": {
                "embedding_dim": 64,
                "transformer_num_heads": 2,
                "epochs": 10,
                "batch_size": 64,
                "max_len": 6,
            },
        },
    ]

    experiment_ids = trainer.train_multiple_models("neural_networks", neural_configs)
    return experiment_ids


def main():
    """
    Main training script with different options.
    """

    parser = argparse.ArgumentParser(description="Train DRC Names Models")
    parser.add_argument(
        "--mode",
        choices=["baseline", "neural", "list"],
        default="list",
        help="Training mode",
    )
    parser.add_argument("--model-type", type=str, help="Specific model type to train")
    parser.add_argument("--name", type=str, help="Model name")

    args = parser.parse_args()

    trainer = ModelTrainer()

    if args.mode == "baseline":
        train_baseline_models()

    elif args.mode == "neural":
        train_neural_networks()

    elif args.mode == "list":
        logging.info("📋 Saved Models:")
        saved_models = trainer.list_saved_models()
        if not saved_models.empty:
            logging.info(saved_models.to_string(index=False))
        else:
            logging.info("No saved models found.")

    elif args.model_type and args.name:
        # Train specific model
        trainer.train_single_model(
            model_name=args.name, model_type=args.model_type, features=["full_name"]
        )


if __name__ == "__main__":
    main()
