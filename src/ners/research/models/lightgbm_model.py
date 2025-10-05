import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

from ners.research.traditional_model import TraditionalModel


class LightGBMModel(TraditionalModel):
    """LightGBM with engineered features"""

    def __init__(self, config):
        super().__init__(config)
        # Store vectorizers and encoders to ensure consistent feature space
        self.vectorizers = {}
        self.label_encoders = {}
        self.feature_columns = []

    def build_model(self) -> BaseEstimator:
        params = self.config.model_params

        # Optional GPU acceleration
        use_gpu = bool(params.get("use_gpu", False))
        device = params.get("device", "gpu" if use_gpu else "cpu")
        gpu_platform_id = params.get("gpu_platform_id", None)
        gpu_device_id = params.get("gpu_device_id", None)

        # Leaf-wise boosted trees excel on sparse/categorical mixes; binary objective
        # and parallelism improve training speed for this task.
        return lgb.LGBMClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", -1),
            learning_rate=params.get("learning_rate", 0.1),
            num_leaves=params.get("num_leaves", 31),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            random_state=self.config.random_seed,
            objective=params.get("objective", "binary"),
            n_jobs=params.get("n_jobs", -1),
            verbose=params.get("verbose", -1),
            device=device,
            gpu_platform_id=gpu_platform_id,
            gpu_device_id=gpu_device_id,
            force_row_wise=params.get("force_row_wise", True),
        )

    def prepare_features(self, X: pd.DataFrame) -> pd.DataFrame | np.ndarray:
        features = []
        columns: list[str] = []

        for feature_type in self.config.features:
            if feature_type.value in X.columns:
                column = X[feature_type.value]

                if feature_type.value in ["name_length", "word_count"]:
                    # Numerical features
                    arr = column.fillna(0).values.reshape(-1, 1)
                    features.append(arr)
                    columns.append(feature_type.value)
                elif feature_type.value in ["full_name", "native_name", "surname"]:
                    # Character-level features for names
                    feature_key = f"vectorizer_{feature_type.value}"

                    if feature_key not in self.vectorizers:
                        # First time - create and fit vectorizer
                        self.vectorizers[feature_key] = CountVectorizer(
                            analyzer="char", ngram_range=(2, 3), max_features=50
                        )
                        vec = self.vectorizers[feature_key]
                        char_features = vec.fit_transform(
                            column.fillna("").astype(str)
                        ).toarray()
                        vocab_names = list(vec.get_feature_names_out())
                    else:
                        # Subsequent times - use existing vectorizer
                        vec = self.vectorizers[feature_key]
                        char_features = vec.transform(
                            column.fillna("").astype(str)
                        ).toarray()
                        vocab_names = list(vec.get_feature_names_out())

                    features.append(char_features)
                    # Prefix with feature name to avoid collisions
                    columns.extend(
                        [f"char_{feature_type.value}_{n}" for n in vocab_names]
                    )
                else:
                    # Categorical features
                    feature_key = f"encoder_{feature_type.value}"

                    if feature_key not in self.label_encoders:
                        # First time - create and fit encoder
                        self.label_encoders[feature_key] = LabelEncoder()
                        encoded = self.label_encoders[feature_key].fit_transform(
                            column.fillna("unknown").astype(str)
                        )
                    else:
                        # Subsequent times - use existing encoder
                        # Handle unseen labels by mapping them to a default value
                        column_clean = column.fillna("unknown").astype(str)

                        # Get the classes the encoder was trained on
                        known_classes = set(self.label_encoders[feature_key].classes_)

                        # Map unseen values to "unknown" if it exists, otherwise to the first class
                        if "unknown" in known_classes:
                            default_class = "unknown"
                        else:
                            default_class = self.label_encoders[feature_key].classes_[0]

                        # Replace unseen values with default
                        column_mapped = column_clean.apply(
                            lambda x: x if x in known_classes else default_class
                        )

                        encoded = self.label_encoders[feature_key].transform(
                            column_mapped
                        )

                    features.append(encoded.reshape(-1, 1))
                    columns.append(f"cat_{feature_type.value}")
        if not features:
            return pd.DataFrame(index=X.index)

        matrix = np.hstack(features)
        # Persist column order for consistency
        self.feature_columns = columns
        return pd.DataFrame(matrix, index=X.index, columns=columns)
