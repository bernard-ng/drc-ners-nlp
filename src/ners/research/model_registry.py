from typing import List

from ners.research.base_model import BaseModel
from ners.research.experiment import ExperimentConfig
from ners.research.models.bigru_model import BiGRUModel
from ners.research.models.cnn_model import CNNModel
from ners.research.models.ensemble_model import EnsembleModel
from ners.research.models.lightgbm_model import LightGBMModel
from ners.research.models.logistic_regression_model import LogisticRegressionModel
from ners.research.models.lstm_model import LSTMModel
from ners.research.models.naive_bayes_model import NaiveBayesModel
from ners.research.models.random_forest_model import RandomForestModel
from ners.research.models.svm_model import SVMModel
from ners.research.models.transformer_model import TransformerModel
from ners.research.models.xgboost_model import XGBoostModel

MODEL_REGISTRY = {
    "bigru": BiGRUModel,
    "cnn": CNNModel,
    "ensemble": EnsembleModel,
    "lightgbm": LightGBMModel,
    "logistic_regression": LogisticRegressionModel,
    "lstm": LSTMModel,
    "naive_bayes": NaiveBayesModel,
    "random_forest": RandomForestModel,
    "svm": SVMModel,
    "transformer": TransformerModel,
    "xgboost": XGBoostModel,
}


def create_model(config: ExperimentConfig) -> BaseModel:
    """Factory function to create models"""
    model_class = MODEL_REGISTRY.get(config.model_type)

    if model_class is None:
        raise ValueError(f"Unknown model type: {config.model_type}")

    return model_class(config)


def list_available_models() -> List[str]:
    """List all available model types"""
    return list(MODEL_REGISTRY.keys())
