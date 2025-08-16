import logging
from abc import abstractmethod
from typing import Dict, Any, List

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import LabelEncoder

from research.base_model import BaseModel
from research.experiment.feature_extractor import FeatureExtractor


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

        # Build model if not already built
        if self.model is None:
            self.model = self.build_model()

        # Setup feature extraction
        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor(
                self.config.features, self.config.feature_params
            )

        # Extract and prepare features
        features_df = self.feature_extractor.extract_features(X)
        X_prepared = self.prepare_features(features_df)

        # Encode labels
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = self.label_encoder.transform(y)

        # Train model
        if len(X_prepared.shape) == 1:
            # For text-based features (like LogisticRegression with vectorization)
            logging.info(f"Fitting model with {X_prepared.shape[0]} samples (text features)")
        else:
            # For numerical features
            logging.info(
                f"Fitting model with {X_prepared.shape[0]} samples and {X_prepared.shape[1]} features"
            )

        self.model.fit(X_prepared, y_encoded)
        self.is_fitted = True

        return self

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> Dict[str, float]:
        features_df = self.feature_extractor.extract_features(X)
        X_prepared = self.prepare_features(features_df)
        y_encoded = self.label_encoder.transform(y)

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.config.random_seed)

        # Calculate different metrics
        results = {}

        # Accuracy
        accuracy_scores = cross_val_score(
            self.model, X_prepared, y_encoded, cv=cv, scoring="accuracy"
        )
        results["accuracy"] = accuracy_scores.mean()
        results["accuracy_std"] = accuracy_scores.std()

        # Precision, Recall, F1
        for metric in ["precision", "recall", "f1"]:
            if metric in self.config.metrics:
                scores = cross_val_score(
                    self.model, X_prepared, y_encoded, cv=cv, scoring=f"{metric}_weighted"
                )
                results[metric] = scores.mean()
                results[f"{metric}_std"] = scores.std()

        return results

    def generate_learning_curve(
            self, X: pd.DataFrame, y: pd.Series, train_sizes: List[float] = None
    ) -> Dict[str, Any]:
        """Generate learning curve data for the model"""
        logging.info(f"Generating learning curve for {self.__class__.__name__}")

        if train_sizes is None:
            train_sizes = [0.1, 0.25, 0.5, 0.75, 1.0]

        # Prepare features
        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor(
                self.config.features, self.config.feature_params
            )

        features_df = self.feature_extractor.extract_features(X)
        X_prepared = self.prepare_features(features_df)

        # Encode labels
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = self.label_encoder.transform(y)

        try:
            train_sizes_abs, train_scores, val_scores = learning_curve(
                self.build_model(),
                X_prepared,
                y_encoded,
                train_sizes=train_sizes,
                cv=3,  # Use 3-fold CV for speed
                scoring="accuracy",
                random_state=self.config.random_seed,
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
