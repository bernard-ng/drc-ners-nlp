#!.venv/bin/python3
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from core.config import setup_config
from research.experiment.experiment_runner import ExperimentRunner
from research.experiment.experiment_tracker import ExperimentTracker


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

    config = setup_config(env="development")
    runner = ExperimentRunner(config)
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
    """Main CLI entry point with unified configuration loading"""
    parser = argparse.ArgumentParser(
        description="DRC Names Research Experiment Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global arguments
    parser.add_argument("--config", type=Path, help="Path to configuration file")
    parser.add_argument(
        "--env", type=str, default="development",
        help="Environment name (default: development)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List experiments
    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument("--status", choices=["pending", "running", "completed", "failed"])
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

    try:
        # Load configuration and setup logging
        config = setup_config(config_path=args.config, env=args.env)

        # Override log level if verbose requested
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        # Execute command
        command_map = {
            "list": list_experiments,
            "show": show_experiment_details,
            "compare": compare_experiments_cmd,
            "export": export_results,
        }
        handler = command_map.get(args.command)
        if handler:
            handler(args)

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
