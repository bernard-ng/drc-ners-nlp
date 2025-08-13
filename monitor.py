#!.venv/bin/python3
import argparse
import sys
import traceback
from pathlib import Path

from core.config import setup_config
from processing.monitoring.pipeline_monitor import PipelineMonitor


def main():
    choices = [
        "data_cleaning",
        "feature_extraction",
        "ner_annotation",
        "llm_annotation",
        "data_splitting",
    ]

    parser = argparse.ArgumentParser(description="DRC NERS Processing Monitoring")
    parser.add_argument("--config", type=Path, help="Path to configuration file")
    parser.add_argument("--env", type=str, default="development", help="Environment")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean checkpoint files")
    clean_parser.add_argument("--step", type=str, choices=choices, help="default: all")
    clean_parser.add_argument("--keep-last", type=int, default=1, help="(default: 1)")
    clean_parser.add_argument("--force", action="store_true", help="Clean without confirmation")

    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset pipeline step")
    reset_parser.add_argument("--step", type=str, choices=choices, help="(default: all)")
    reset_parser.add_argument("--all", action="store_true", help="Reset all steps")
    reset_parser.add_argument("--force", action="store_true", help="Reset without confirmation")
    args = parser.parse_args()

    try:
        setup_config(config_path=args.config, env=args.env)
        monitor = PipelineMonitor()

        if not args.command:
            parser.print_help()
            monitor.print_status(detailed=True)
            return 1

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

            if args.step:
                monitor.reset_step(args.step)
            else:
                for step in monitor.steps:
                    monitor.reset_step(step)

            print(f"Reset completed")

    except Exception as e:
        print(f"Monitoring failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
