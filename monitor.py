#!.venv/bin/python3
import argparse
import sys
from pathlib import Path

from core.config import setup_config_and_logging
from processing.monitoring.data_analyzer import DatasetAnalyzer
from processing.monitoring.pipeline_monitor import PipelineMonitor


def main():
    parser = argparse.ArgumentParser(
        description="Monitor and manage the DRC names processing pipeline"
    )
    parser.add_argument("--config", type=Path, help="Path to configuration file")
    parser.add_argument(
        "--env", type=str, default="development",
        help="Environment name (default: development)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show pipeline status")
    status_parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed information including failed batch IDs",
    )

    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean checkpoint files")
    clean_parser.add_argument(
        "--step",
        type=str,
        choices=["data_cleaning", "feature_extraction", "llm_annotation", "data_splitting"],
        help="Clean specific step (default: all)",
    )
    clean_parser.add_argument(
        "--keep-last", type=int, default=1, help="Number of recent checkpoints to keep (default: 1)"
    )
    clean_parser.add_argument("--force", action="store_true", help="Clean without confirmation")

    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset pipeline step")
    reset_parser.add_argument(
        "step",
        type=str,
        choices=["data_cleaning", "feature_extraction", "llm_annotation", "data_splitting"],
        help="Step to reset",
    )
    reset_parser.add_argument("--force", action="store_true", help="Reset without confirmation")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze dataset")
    analyze_parser.add_argument(
        "--file",
        type=str,
        default="names_featured.csv",
        help="Dataset file to analyze (default: names_featured.csv)",
    )

    # Checkpoint info command
    info_parser = subparsers.add_parser("info", help="Show checkpoint information")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        # Load configuration and setup logging
        config = setup_config_and_logging(config_path=args.config, env=args.env)

        monitor = PipelineMonitor()

        if args.command == "status":
            monitor.print_status(detailed=args.detailed)

        elif args.command == "clean":
            checkpoint_info = monitor.count_checkpoint_files()
            print(f"Current checkpoint storage: {checkpoint_info['total_size_mb']:.1f} MB")

            if not args.force:
                response = input("Are you sure you want to clean checkpoints? (y/N): ")
                if response.lower() != "y":
                    print("Cancelled")
                    return 0

            if args.step:
                monitor.clean_step_checkpoints(args.step, args.keep_last)
            else:
                for step in monitor.steps:
                    monitor.clean_step_checkpoints(step, args.keep_last)

            print("Checkpoint cleaning completed")

        elif args.command == "reset":
            if not args.force:
                response = input(
                    f"Are you sure you want to reset {args.step}? This will delete all checkpoints. (y/N): "
                )
                if response.lower() != "y":
                    print("Cancelled")
                    return 0

            monitor.reset_step(args.step)
            print(f"Reset completed for {args.step}")

        elif args.command == "analyze":
            # Use configured data directory
            data_dir = config.paths.data_dir
            filepath = data_dir / args.file

            if not filepath.exists():
                print(f"File not found: {filepath}")
                return 1

            analyzer = DatasetAnalyzer(str(filepath))

            if not analyzer.load_data():
                return 1

            completion_stats = analyzer.analyze_completion()

            print(f"\n=== Dataset Analysis: {args.file} ===")
            print(f"Total rows: {completion_stats['total_rows']:,}")
            print(f"Annotated: {completion_stats['annotated_rows']:,} ({completion_stats['annotation_percentage']:.1f}%)")
            print(f"Unannotated: {completion_stats['unannotated_rows']:,}")
            print(
                f"Complete names: {completion_stats['complete_names']:,} ({completion_stats['completeness_percentage']:.1f}%)"
            )

        elif args.command == "info":
            checkpoint_info = monitor.count_checkpoint_files()

            print(f"\n=== Checkpoint Information ===")
            print(f"Total storage: {checkpoint_info['total_size_mb']:.1f} MB")
            print()

            for step in monitor.steps:
                step_info = checkpoint_info[step]
                print(f"{step.replace('_', ' ').title()}:")
                print(f"  Files: {step_info['files']}")
                print(f"  Size: {step_info['size_mb']:.1f} MB")
                print()

        return 0

    except Exception as e:
        print(f"Monitor command failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
