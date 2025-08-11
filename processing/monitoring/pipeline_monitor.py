import json
import logging
import shutil
from datetime import datetime
from typing import Optional, Dict

from core.config.config_manager import ConfigManager
from core.config.project_paths import ProjectPaths


class PipelineMonitor:
    """Monitor and manage pipeline execution"""

    def __init__(self, paths: Optional[ProjectPaths] = None):
        if paths is None:
            # Use default configuration if none provided
            config_manager = ConfigManager()
            paths = config_manager.default_paths

        self.paths = paths
        self.checkpoint_dir = paths.checkpoints_dir
        self.steps = ["data_cleaning", "feature_extraction", "ner_annotation", "llm_annotation", "data_splitting"]

    def get_step_status(self, step_name: str) -> Dict:
        """Get status of a specific pipeline step"""
        step_dir = self.checkpoint_dir / step_name
        state_file = step_dir / "pipeline_state.json"

        if not state_file.exists():
            return {
                "step": step_name,
                "status": "not_started",
                "processed_batches": 0,
                "total_batches": 0,
                "failed_batches": 0,
                "completion_percentage": 0.0,
            }

        try:
            with open(state_file, "r") as f:
                state = json.load(f)

            processed = state.get("processed_batches", 0)
            total = state.get("total_batches", 0)
            failed = len(state.get("failed_batches", []))

            if total == 0:
                completion = 0.0
                status = "not_started"
            elif processed >= total:
                completion = 100.0
                status = "completed" if failed == 0 else "completed_with_errors"
            else:
                completion = (processed / total) * 100
                status = "in_progress"

            return {
                "step": step_name,
                "status": status,
                "processed_batches": processed,
                "total_batches": total,
                "failed_batches": failed,
                "completion_percentage": completion,
                "last_checkpoint": state.get("last_checkpoint"),
                "failed_batch_ids": state.get("failed_batches", []),
            }

        except Exception as e:
            logging.error(f"Error reading state for {step_name}: {e}")
            return {"step": step_name, "status": "error", "error": str(e)}

    def get_pipeline_status(self) -> Dict:
        """Get overall pipeline status"""
        step_statuses = {}
        overall_status = "not_started"
        total_completion = 0.0

        for step in self.steps:
            status = self.get_step_status(step)
            step_statuses[step] = status

            if status["status"] == "error":
                overall_status = "error"
            elif status["status"] in ["in_progress"]:
                overall_status = "in_progress"
            elif status["status"] == "completed_with_errors":
                overall_status = "completed_with_errors"

            total_completion += status.get("completion_percentage", 0)

        avg_completion = total_completion / len(self.steps)

        if avg_completion >= 100 and overall_status not in ["error", "completed_with_errors"]:
            overall_status = "completed"

        return {
            "overall_status": overall_status,
            "overall_completion": avg_completion,
            "steps": step_statuses,
            "timestamp": datetime.now().isoformat(),
        }

    def print_status(self, detailed: bool = False):
        """Print pipeline status in a human-readable format"""
        status = self.get_pipeline_status()

        print("\n=== Pipeline Status ===")
        print(f"Overall Status: {status['overall_status'].upper()}")
        print(f"Overall Completion: {status['overall_completion']:.1f}%")
        print(f"Last Updated: {status['timestamp']}")
        print()

        for step_name, step_status in status["steps"].items():
            print(f"{step_name.replace('_', ' ').title()}:")
            print(f"  Status: {step_status['status']}")
            print(f"  Progress: {step_status['completion_percentage']:.1f}%")
            print(f"  Batches: {step_status['processed_batches']}/{step_status['total_batches']}")

            if step_status["failed_batches"] > 0:
                print(f"  Failed Batches: {step_status['failed_batches']}")

                if detailed and "failed_batch_ids" in step_status:
                    print(f"  Failed Batch IDs: {step_status['failed_batch_ids']}")

            print()

    def count_checkpoint_files(self) -> Dict:
        """Count checkpoint files for each step"""
        counts = {}
        total_size = 0

        for step in self.steps:
            step_dir = self.checkpoint_dir / step
            if step_dir.exists():
                csv_files = list(step_dir.glob("*.csv"))
                step_size = sum(f.stat().st_size for f in csv_files)
                counts[step] = {"files": len(csv_files), "size_mb": step_size / (1024 * 1024)}
                total_size += step_size
            else:
                counts[step] = {"files": 0, "size_mb": 0}

        counts["total_size_mb"] = total_size / (1024 * 1024)
        return counts

    def clean_step_checkpoints(self, step_name: str, keep_last: int = 1):
        """Clean checkpoint files for a specific step"""
        step_dir = self.checkpoint_dir / step_name

        if not step_dir.exists():
            logging.info(f"No checkpoints found for {step_name}")
            return

        csv_files = sorted(step_dir.glob("batch_*.csv"))

        if len(csv_files) <= keep_last:
            logging.info(f"Only {len(csv_files)} checkpoint files for {step_name}, keeping all")
            return

        files_to_delete = csv_files[:-keep_last] if keep_last > 0 else csv_files

        for file_path in files_to_delete:
            try:
                file_path.unlink()
                logging.info(f"Deleted {file_path}")
            except Exception as e:
                logging.error(f"Failed to delete {file_path}: {e}")

    def reset_step(self, step_name: str):
        """Reset a pipeline step by removing its checkpoints and state"""
        step_dir = self.checkpoint_dir / step_name

        if step_dir.exists():
            try:
                shutil.rmtree(step_dir)
                logging.info(f"Reset step: {step_name}")
            except Exception as e:
                logging.error(f"Failed to reset {step_name}: {e}")
        else:
            logging.info(f"Step {step_name} has no checkpoints to reset")
