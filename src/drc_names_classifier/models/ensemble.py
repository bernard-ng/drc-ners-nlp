import numpy as np
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from drc_names_classifier.config import ExperimentConfig
from drc_names_classifier.models.sklearn import SklearnModel
from drc_names_classifier.utils import full_name_array


class EnsembleModel(SklearnModel):
    """Voting ensemble combining complementary full-name models."""

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.base_models = []
        self.model_weights = None

    def build_model(self) -> BaseEstimator:
        params = self.config.model_params
        base_model_types = params.get(
            "base_models", ["logistic_regression", "random_forest", "naive_bayes"]
        )

        estimators = []
        for model_type in base_model_types:
            if model_type == "logistic_regression":
                model = Pipeline(
                    [
                        (
                            "vectorizer",
                            CountVectorizer(
                                analyzer="char", ngram_range=(2, 4), max_features=5000
                            ),
                        ),
                        (
                            "classifier",
                            LogisticRegression(
                                max_iter=1000, random_state=self.config.random_seed
                            ),
                        ),
                    ]
                )
                estimators.append(("logistic_regression", model))

            elif model_type == "random_forest":
                model = Pipeline(
                    [
                        (
                            "vectorizer",
                            TfidfVectorizer(
                                analyzer="char", ngram_range=(2, 3), max_features=3000
                            ),
                        ),
                        (
                            "classifier",
                            RandomForestClassifier(
                                n_estimators=50, random_state=self.config.random_seed
                            ),
                        ),
                    ]
                )
                estimators.append(("rf", model))

            elif model_type == "naive_bayes":
                model = Pipeline(
                    [
                        (
                            "vectorizer",
                            CountVectorizer(
                                analyzer="char", ngram_range=(1, 3), max_features=4000
                            ),
                        ),
                        ("classifier", MultinomialNB()),
                    ]
                )
                estimators.append(("nb", model))

        voting_type = params.get("voting", "soft")
        return VotingClassifier(
            estimators=estimators, voting=voting_type, n_jobs=params.get("n_jobs", -1)
        )

    def prepare_features(self, X: pl.DataFrame) -> np.ndarray:
        return full_name_array(X)
