#!.venv/bin/python3
import argparse
import sys
from pathlib import Path
import json
import pandas as pd
import logging

from core.config import get_config, setup_logging
from research.experiment import ExperimentConfig
from research.experiment.experiment_tracker import ExperimentTracker
from research.experiment.feature_extractor import FeatureType
from research.experiment.experiment_builder import ExperimentBuilder
from research.experiment.experiment_runner import ExperimentRunner
from research.model_registry import list_available_models


def create_experiment_from_args(args) -> ExperimentConfig:
    """Create experiment configuration from command line arguments"""

    features = []
    if args.features:
        for feature_name in args.features:
            try:
                features.append(FeatureType(feature_name))
            except ValueError:
                logging.warning(f"Unknown feature type '{feature_name}', skipping")

    if not features:
        features = [FeatureType.FULL_NAME]  # Default

    # Parse model parameters
    model_params = {}
    if args.model_params:
        try:
            model_params = json.loads(args.model_params)
        except json.JSONDecodeError:
            logging.warning("Invalid JSON for model parameters, using defaults")

    # Parse feature parameters
    feature_params = {}
    if args.feature_params:
        try:
            feature_params = json.loads(args.feature_params)
        except json.JSONDecodeError:
            logging.warning("Invalid JSON for feature parameters, using defaults")

    # Parse data filters
    train_filter = None
    if args.train_filter:
        try:
            train_filter = json.loads(args.train_filter)
        except json.JSONDecodeError:
            logging.warning("Invalid JSON for train filter, ignoring")

    return ExperimentConfig(
        name=args.name,
        description=args.description or "",
        tags=args.tags or [],
        model_type=args.model_type,
        model_params=model_params,
        features=features,
        feature_params=feature_params,
        train_data_filter=train_filter,
        target_column=args.target,
        test_size=args.test_size,
        random_seed=args.seed,
        cross_validation_folds=args.cv_folds,
        metrics=args.metrics or ["accuracy", "precision", "recall", "f1"],
    )


def run_single_experiment(args):
    """Run a single experiment"""

    config = create_experiment_from_args(args)
    runner = ExperimentRunner()
    experiment_id = runner.run_experiment(config)

    logging.info(f"Experiment completed: {experiment_id}")

    # Show results
    experiment = runner.tracker.get_experiment(experiment_id)
    if experiment:
        logging.info("Results:")
        for metric, value in experiment.test_metrics.items():
            logging.info(f"  Test {metric}: {value:.4f}")

        if experiment.cv_metrics:
            logging.info("Cross-validation:")
            for metric, value in experiment.cv_metrics.items():
                if not metric.endswith("_std"):
                    std_key = f"{metric}_std"
                    std_val = experiment.cv_metrics.get(std_key, 0)
                    logging.info(f"  CV {metric}: {value:.4f} ± {std_val:.4f}")


def run_baseline_experiments(args):
    """Run baseline experiments"""
    logger = logging.getLogger(__name__)

    builder = ExperimentBuilder()
    experiments = builder.create_baseline_experiments()

    runner = ExperimentRunner()
    experiment_ids = runner.run_experiment_batch(experiments)

    logging.info(f"Completed {len(experiment_ids)} baseline experiments")

    # Show comparison
    if experiment_ids:
        comparison = runner.compare_experiments(experiment_ids)
        logging.info("Baseline Results Comparison:")
        logging.info(
            comparison[["name", "model_type", "features", "test_accuracy"]].to_string(index=False)
        )


def run_ablation_study(args):
    """Run feature ablation study"""

    builder = ExperimentBuilder()
    experiments = builder.create_feature_ablation_study()

    runner = ExperimentRunner()
    experiment_ids = runner.run_experiment_batch(experiments)

    logging.info(f"Completed {len(experiment_ids)} ablation experiments")

    # Show results
    if experiment_ids:
        comparison = runner.compare_experiments(experiment_ids)
        logging.info("Ablation Study Results:")
        logging.info(comparison[["name", "test_accuracy", "test_f1"]].to_string(index=False))


def run_component_study(args):
    """Run name component study"""

    builder = ExperimentBuilder()
    experiments = builder.create_name_component_study()

    runner = ExperimentRunner()
    experiment_ids = runner.run_experiment_batch(experiments)

    logging.info(f"Completed {len(experiment_ids)} component study experiments")

    # Show results
    if experiment_ids:
        comparison = runner.compare_experiments(experiment_ids)
        logging.info("Name Component Study Results:")
        logging.info(
            comparison[["name", "test_accuracy", "test_precision", "test_recall"]].to_string(
                index=False
            )
        )


def list_experiments(args):
    """List experiments with optional filtering"""

    tracker = ExperimentTracker()

    # Apply filters
    filters = {}
    if args.status:
        from research.experiment import ExperimentStatus

        filters["status"] = ExperimentStatus(args.status)
    if args.model_type:
        filters["model_type"] = args.model_type
    if args.tags:
        filters["tags"] = args.tags

    experiments = tracker.list_experiments(**filters)

    if not experiments:
        logging.info("No experiments found matching criteria")
        return

    # Create summary table
    rows = []
    for exp in experiments:
        row = {
            "ID": exp.experiment_id[:12] + "...",
            "Name": exp.config.name,
            "Model": exp.config.model_type,
            "Status": exp.status.value,
            "Test Acc": f"{exp.test_metrics.get('accuracy', 0):.4f}" if exp.test_metrics else "N/A",
            "Start Time": exp.start_time.strftime("%Y-%m-%d %H:%M"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    logging.info(df.to_string(index=False))


def show_experiment_details(args):
    """Show detailed results for an experiment"""

    tracker = ExperimentTracker()
    experiment = tracker.get_experiment(args.experiment_id)

    if not experiment:
        logging.error(f"Experiment not found: {args.experiment_id}")
        return

    logging.info("=== Experiment Details ===")
    logging.info(f"ID: {experiment.experiment_id}")
    logging.info(f"Name: {experiment.config.name}")
    logging.info(f"Description: {experiment.config.description}")
    logging.info(f"Model Type: {experiment.config.model_type}")
    logging.info(f"Features: {', '.join([f.value for f in experiment.config.features])}")
    logging.info(f"Status: {experiment.status.value}")
    logging.info(f"Start Time: {experiment.start_time}")
    logging.info(f"End Time: {experiment.end_time}")

    if experiment.test_metrics:
        logging.info("=== Test Metrics ===")
        for metric, value in experiment.test_metrics.items():
            logging.info(f"{metric}: {value:.4f}")

    if experiment.cv_metrics:
        logging.info("=== Cross-Validation Metrics ===")
        for metric, value in experiment.cv_metrics.items():
            if not metric.endswith("_std"):
                std_key = f"{metric}_std"
                std_val = experiment.cv_metrics.get(std_key, 0)
                logging.info(f"{metric}: {value:.4f} ± {std_val:.4f}")

    if experiment.feature_importance:
        logging.info("=== Top 10 Feature Importances ===")
        sorted_features = sorted(
            experiment.feature_importance.items(), key=lambda x: x[1], reverse=True
        )
        for feature, importance in sorted_features[:10]:
            logging.info(f"{feature}: {importance:.4f}")

    if experiment.prediction_examples:
        logging.info("=== Prediction Examples ===")
        for i, example in enumerate(experiment.prediction_examples[:5]):
            correct = "✓" if example["correct"] else "✗"
            logging.info(
                f"{i + 1}. {example['name']} -> True: {example['true_label']}, "
                f"Pred: {example['predicted_label']} {correct}"
            )


def compare_experiments_cmd(args):
    """Compare multiple experiments"""

    runner = ExperimentRunner()
    comparison = runner.compare_experiments(args.experiment_ids)

    if comparison.empty:
        logging.info("No experiments found for comparison")
        return

    logging.info("=== Experiment Comparison ===")

    # Show key columns
    key_columns = ["name", "model_type", "features", "test_accuracy", "test_f1"]
    available_columns = [col for col in key_columns if col in comparison.columns]

    logging.info(comparison[available_columns].to_string(index=False))


def export_results(args):
    """Export experiment results"""

    tracker = ExperimentTracker()
    output_path = tracker.export_results(Path(args.output) if args.output else None)

    logging.info(f"Results exported to: {output_path}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="DRC Names Research Experiment Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Setup logging
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Single experiment command
    exp_parser = subparsers.add_parser("run", help="Run a single experiment")
    exp_parser.add_argument("--name", required=True, help="Experiment name")
    exp_parser.add_argument("--description", help="Experiment description")
    exp_parser.add_argument(
        "--model-type",
        default="logistic_regression",
        choices=list_available_models(),
        help="Model type",
    )
    exp_parser.add_argument(
        "--features", nargs="+", choices=[f.value for f in FeatureType], help="Features to use"
    )
    exp_parser.add_argument("--model-params", help="Model parameters as JSON")
    exp_parser.add_argument("--feature-params", help="Feature parameters as JSON")
    exp_parser.add_argument("--train-filter", help="Training data filter as JSON")
    exp_parser.add_argument("--target", default="sex", help="Target column")
    exp_parser.add_argument("--test-size", type=float, default=0.2, help="Test set size")
    exp_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    exp_parser.add_argument("--cv-folds", type=int, default=5, help="CV folds")
    exp_parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["accuracy", "precision", "recall", "f1"],
        help="Metrics to calculate",
    )
    exp_parser.add_argument("--tags", nargs="+", help="Experiment tags")

    # Batch experiment commands
    subparsers.add_parser("baseline", help="Run baseline experiments")
    subparsers.add_parser("ablation", help="Run feature ablation study")
    subparsers.add_parser("components", help="Run name component study")

    # List experiments
    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument("--status", choices=["pending", "running", "completed", "failed"])
    list_parser.add_argument("--model-type", choices=list_available_models())
    list_parser.add_argument("--tags", nargs="+", help="Filter by tags")

    # Show experiment details
    detail_parser = subparsers.add_parser("show", help="Show experiment details")
    detail_parser.add_argument("experiment_id", help="Experiment ID")

    # Compare experiments
    compare_parser = subparsers.add_parser("compare", help="Compare experiments")
    compare_parser.add_argument("experiment_ids", nargs="+", help="Experiment IDs to compare")

    # Export results
    export_parser = subparsers.add_parser("export", help="Export results to CSV")
    export_parser.add_argument("--output", help="Output file path")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Setup logging
    config = get_config()
    if args.verbose:
        config.logging.level = "DEBUG"
    setup_logging(config)

    # Execute command
    try:
        if args.command == "run":
            run_single_experiment(args)
        elif args.command == "baseline":
            run_baseline_experiments(args)
        elif args.command == "ablation":
            run_ablation_study(args)
        elif args.command == "components":
            run_component_study(args)
        elif args.command == "list":
            list_experiments(args)
        elif args.command == "show":
            show_experiment_details(args)
        elif args.command == "compare":
            compare_experiments_cmd(args)
        elif args.command == "export":
            export_results(args)

        return 0

    except Exception as e:
        logging.error(f"Command failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
