from dataclasses import field
from typing import Dict

from pydantic import BaseModel


class DataConfig(BaseModel):
    """Data handling configuration"""

    input_file: str = "names.csv"
    output_files: Dict[str, str] = field(
        default_factory=lambda: {
            "featured": "names_featured.csv",
            "evaluation": "names_evaluation.csv",
            "males": "names_males.csv",
            "females": "names_females.csv",
        }
    )
    split_evaluation: bool = True
    split_by_gender: bool = True
    evaluation_fraction: float = 0.2
    random_seed: int = 42
