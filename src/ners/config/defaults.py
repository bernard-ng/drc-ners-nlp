from pathlib import Path


DEFAULT_DATASET_PATH = Path("data/dataset/names.csv")
DEFAULT_EXPERIMENT_TEMPLATES_PATH = Path("config/experiment_templates.yaml")
DEFAULT_EXPERIMENT_MODELS_DIR = Path("data/models")
DEFAULT_EXPERIMENT_OUTPUTS_DIR = Path("data/outputs")
DEFAULT_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "macro_f1",
    "mcc",
)
