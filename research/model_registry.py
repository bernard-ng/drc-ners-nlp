from typing import List

from research.base_model import BaseModel
from research.experiment import ExperimentConfig
from research.models.bigru_model import BiGRUModel
from research.models.cnn_model import CNNModel
from research.models.ensemble_model import EnsembleModel
from research.models.lightgbm_model import LightGBMModel
from research.models.logistic_regression_model import LogisticRegressionModel
from research.models.lstm_model import LSTMModel
from research.models.naive_bayes_model import NaiveBayesModel
from research.models.random_forest_model import RandomForestModel
from research.models.svm_model import SVMModel
from research.models.transformer_model import TransformerModel
from research.models.xgboost_model import XGBoostModel

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
