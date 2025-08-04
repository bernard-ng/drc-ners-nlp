import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from research.experiment import ExperimentConfig
from research.traditional_model import TraditionalModel


class EnsembleModel(TraditionalModel):
    """Ensemble model combining multiple base models"""

    @property
    def architecture(self) -> str:
        """Return the architecture type"""
        return "ensemble"

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.base_models = []
        self.model_weights = None

    def build_model(self) -> BaseEstimator:
        params = self.config.model_params
        base_model_types = params.get(
            "base_models", ["logistic_regression", "random_forest", "naive_bayes"]
        )

        # Create base models with simplified configs
        estimators = []
        for model_type in base_model_types:
            if model_type == "logistic_regression":
                model = Pipeline(
                    [
                        (
                            "vectorizer",
                            CountVectorizer(analyzer="char", ngram_range=(2, 4), max_features=5000),
                        ),
                        (
                            "classifier",
                            LogisticRegression(max_iter=1000, random_state=self.config.random_seed),
                        ),
                    ]
                )
                estimators.append((f"logistic_regression", model))

            elif model_type == "random_forest":
                model = Pipeline(
                    [
                        (
                            "vectorizer",
                            TfidfVectorizer(analyzer="char", ngram_range=(2, 3), max_features=3000),
                        ),
                        (
                            "classifier",
                            RandomForestClassifier(
                                n_estimators=50, random_state=self.config.random_seed
                            ),
                        ),
                    ]
                )
                estimators.append((f"rf", model))

            elif model_type == "naive_bayes":
                model = Pipeline(
                    [
                        (
                            "vectorizer",
                            CountVectorizer(analyzer="char", ngram_range=(1, 3), max_features=4000),
                        ),
                        ("classifier", MultinomialNB()),
                    ]
                )
                estimators.append((f"nb", model))

        voting_type = params.get("voting", "soft")  # 'hard' or 'soft'
        return VotingClassifier(estimators=estimators, voting=voting_type)

    def prepare_features(self, X: pd.DataFrame) -> np.ndarray:
        text_features = []

        for feature_type in self.config.features:
            if feature_type.value in X.columns:
                text_features.append(X[feature_type.value].astype(str))

        if len(text_features) == 1:
            return text_features[0].values
        else:
            combined = text_features[0].astype(str)
            for feature in text_features[1:]:
                combined = combined + " " + feature.astype(str)
            return combined.values
