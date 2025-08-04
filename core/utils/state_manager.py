import json
import logging
from typing import Dict, Any

from core.config.pipeline_config import PipelineConfig


class StateManager:
    """Manage pipeline state and checkpoints"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.checkpoints_dir = self.config.paths.checkpoints_dir

    def save_state(self, state: Dict[str, Any], state_name: str) -> None:
        """Save pipeline state"""
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        state_file = self.checkpoints_dir / f"{state_name}.json"

        with open(state_file, "w") as f:
            json.dump(state, f, indent=2, default=str)

        logging.debug(f"Saved state to {state_file}")

    def load_state(self, state_name: str) -> Dict[str, Any]:
        """Load pipeline state"""
        state_file = self.checkpoints_dir / f"{state_name}.json"

        if not state_file.exists():
            return {}

        with open(state_file, "r") as f:
            return json.load(f)

    def clear_state(self, state_name: str) -> None:
        """Clear pipeline state"""
        state_file = self.checkpoints_dir / f"{state_name}.json"

        if state_file.exists():
            state_file.unlink()
            logging.info(f"Cleared state: {state_name}")
