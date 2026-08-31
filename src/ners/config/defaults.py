from pathlib import Path


DEFAULT_DATASET_PATH = Path("data/dataset/names.csv")
DEFAULT_RESEARCH_TEMPLATES_PATH = Path("config/research_templates.yaml")
DEFAULT_RESEARCH_MODELS_DIR = Path("data/models")
DEFAULT_RESEARCH_OUTPUTS_DIR = Path("data/outputs")
DEFAULT_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "macro_f1",
    "mcc",
)
