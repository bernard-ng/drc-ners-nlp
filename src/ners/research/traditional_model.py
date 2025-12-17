import logging
from abc import abstractmethod
from typing import Dict, Any, List, Optional, cast

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import LabelEncoder

from ners.research.base_model import BaseModel
from ners.research.experiment.feature_extractor import FeatureExtractor


class TraditionalModel(BaseModel):
    """Base class for traditional ML models (scikit-learn compatible)"""

    @property
    def architecture(self) -> str:
        return "traditional"

    @abstractmethod
    def build_model(self) -> BaseEstimator:
        """Build and return the sklearn model instance"""
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """Fit the traditional ML model"""
        logging.info(f"Training {self.__class__.__name__}")

        if self.model is None:
            self.model = self.build_model()

        model = cast(BaseEstimator, self.model)

        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor(
                self.config.features, self.config.feature_params
            )

        fe = cast(FeatureExtractor, self.feature_extractor)
        features_df = fe.extract_features(X)
        X_prepared = self.prepare_features(features_df)

        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()

        le = cast(LabelEncoder, self.label_encoder)
        y_encoded = le.fit_transform(y)

        if X_prepared.ndim == 1:
            logging.info(
                f"Fitting model with {X_prepared.shape[0]} samples (text features)"
            )
        else:
            logging.info(
                f"Fitting model with {X_prepared.shape[0]} samples and {X_prepared.shape[1]} features"
            )

        try:
            if isinstance(X_prepared, pd.DataFrame):
                logging.info(X_prepared.iloc[0].to_dict())
            elif isinstance(X_prepared, np.ndarray):
                logging.info(X_prepared[0])
        except Exception:
            pass

        logging.info(f"Model parameters: {self.config.model_params}")

        model.fit(X_prepared, y_encoded)
        self.is_fitted = True
        self.training_history = {}

        return self

    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5
    ) -> Dict[str, float]:
        """Run stratified cross-validation on the model"""
        if self.model is None or self.feature_extractor is None or self.label_encoder is None:
            raise ValueError("Model, FeatureExtractor, and LabelEncoder must be initialized.")

        model = cast(BaseEstimator, self.model)
        fe = cast(FeatureExtractor, self.feature_extractor)
        le = cast(LabelEncoder, self.label_encoder)

        features_df = fe.extract_features(X)
        X_prepared = self.prepare_features(features_df)
        y_encoded = le.transform(y)

        cv = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=self.config.random_seed
        )

        results: Dict[str, float] = {}

        accuracy_scores = cross_val_score(
            model, X_prepared, y_encoded, cv=cv, scoring="accuracy"
        )
        results["accuracy"] = accuracy_scores.mean()
        results["accuracy_std"] = accuracy_scores.std()

        for metric in ["precision", "recall", "f1"]:
            if metric in self.config.metrics:
                scores = cross_val_score(
                    model,
                    X_prepared,
                    y_encoded,
                    cv=cv,
                    scoring=f"{metric}_weighted",
                )
                results[metric] = scores.mean()
                results[f"{metric}_std"] = scores.std()

        return results

    def generate_learning_curve(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_sizes: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Generate learning curve data for the model"""
        logging.info(f"Generating learning curve for {self.__class__.__name__}")

        actual_train_sizes = train_sizes or [0.1, 0.25, 0.5, 0.75, 1.0]

        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor(
                self.config.features, self.config.feature_params
            )

        fe = cast(FeatureExtractor, self.feature_extractor)
        features_df = fe.extract_features(X)
        X_prepared = self.prepare_features(features_df)

        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()

        le = cast(LabelEncoder, self.label_encoder)
        y_encoded = le.fit_transform(y)

        try:
            train_sizes_abs, train_scores, val_scores = learning_curve(
                self.build_model(),
                X_prepared,
                y_encoded,
                train_sizes=actual_train_sizes,
                cv=3,
                scoring="accuracy",
            )

            learning_curve_data = {
                "train_sizes": train_sizes_abs.tolist(),
                "train_scores": train_scores.mean(axis=1).tolist(),
                "val_scores": val_scores.mean(axis=1).tolist(),
                "train_scores_std": train_scores.std(axis=1).tolist(),
                "val_scores_std": val_scores.std(axis=1).tolist(),
            }
        except Exception as e:
            logging.warning(f"Could not generate learning curve: {e}")
            return {}

        self.learning_curve_data = learning_curve_data
        return learning_curve_data
