from dataclasses import field
from typing import Dict, Optional

from pydantic import BaseModel


class DataConfig(BaseModel):
    """Data handling configuration"""

    input_file: str = "names.csv"
    output_files: Dict[str, str] = field(
        default_factory=lambda: {
            "featured": "names_featured.csv",
            "evaluation": "names_evaluation.csv",
            "engineered": "names_engineered.csv",
            "males": "names_males.csv",
            "females": "names_females.csv",
            "ner_data": "names_ner.json",
            "ner_spacy": "names_ner.spacy",
        }
    )
    split_evaluation: bool = False
    split_by_province: bool = True
    split_by_gender: bool = True
    split_ner_data: bool = True
    evaluation_fraction: float = 0.2
    random_seed: int = 42

    # Dataset size limiting options
    max_dataset_size: Optional[int] = None
    balance_by_sex: bool = False
