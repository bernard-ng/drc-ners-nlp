from core.config.pipeline_config import PipelineConfig


class PromptManager:
    """Manage prompts for LLM operations"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.prompts_dir = self.config.paths.configs_dir / "prompts"

    def load_prompt(self, prompt_name: str = "default") -> str:
        """Load a prompt template"""
        prompt_file = self.prompts_dir / f"{prompt_name}.txt"

        if not prompt_file.exists():
            # Fallback to root directory
            fallback_file = self.config.paths.root_dir / "prompt.txt"
            if fallback_file.exists():
                prompt_file = fallback_file
            else:
                raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
