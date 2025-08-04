from typing import List

from research.experiment import ExperimentConfig
from research.experiment.feature_extractor import FeatureType


class ExperimentBuilder:
    """Helper class to build experiment configurations"""

    @staticmethod
    def create_baseline_experiments() -> List[ExperimentConfig]:
        """Create a set of baseline experiments for comparison"""

        return [
            # Full name experiments
            ExperimentConfig(
                name="baseline_logistic_regression_fullname",
                description="Logistic regression with full name",
                model_type="logistic_regression",
                features=[FeatureType.FULL_NAME],
                tags=["baseline", "fullname"],
            ),
            # Native name only
            ExperimentConfig(
                name="baseline_logistic_regression_native",
                description="Logistic regression with native name only",
                model_type="logistic_regression",
                features=[FeatureType.NATIVE_NAME],
                tags=["baseline", "native"],
            ),
            # Surname only
            ExperimentConfig(
                name="baseline_logistic_regression_surname",
                description="Logistic regression with surname only",
                model_type="logistic_regression",
                features=[FeatureType.SURNAME],
                tags=["baseline", "surname"],
            ),
            # Random Forest with engineered features
            ExperimentConfig(
                name="baseline_rf_engineered",
                description="Random Forest with engineered features",
                model_type="random_forest",
                features=[FeatureType.NAME_LENGTH, FeatureType.WORD_COUNT, FeatureType.PROVINCE],
                tags=["baseline", "engineered"],
            ),
        ]

    @staticmethod
    def create_feature_ablation_study() -> List[ExperimentConfig]:
        """Create experiments for feature ablation study"""
        base_features = [
            FeatureType.FULL_NAME,
            FeatureType.NAME_LENGTH,
            FeatureType.WORD_COUNT,
            FeatureType.PROVINCE,
        ]

        experiments = []

        # Test removing each feature one by one
        for i, feature_to_remove in enumerate(base_features):
            remaining_features = [f for f in base_features if f != feature_to_remove]

            experiments.append(
                ExperimentConfig(
                    name=f"ablation_remove_{feature_to_remove.value}",
                    description=f"Ablation study: removed {feature_to_remove.value}",
                    model_type="logistic_regression",
                    features=remaining_features,
                    tags=["ablation", feature_to_remove.value],
                )
            )

        return experiments

    @staticmethod
    def create_name_component_study() -> List[ExperimentConfig]:
        """Create experiments to study different name components"""
        experiments = []

        name_components = [
            (FeatureType.FIRST_WORD, "first_word"),
            (FeatureType.LAST_WORD, "last_word"),
            (FeatureType.NATIVE_NAME, "native_name"),
            (FeatureType.SURNAME, "surname"),
            (FeatureType.NAME_BEGINNINGS, "name_beginnings"),
            (FeatureType.NAME_ENDINGS, "name_endings"),
        ]

        for feature, name in name_components:
            experiments.append(
                ExperimentConfig(
                    name=f"component_study_{name}",
                    description=f"Study of {name} for gender prediction",
                    model_type="logistic_regression",
                    features=[feature],
                    tags=["component_study", name],
                )
            )

        return experiments

    @staticmethod
    def create_province_specific_study() -> List[ExperimentConfig]:
        """Create experiments for province-specific analysis"""
        provinces = ["kinshasa", "bas-congo", "bandundu", "katanga"]  # Add more as needed

        experiments = []

        for province in provinces:
            experiments.append(
                ExperimentConfig(
                    name=f"province_study_{province}",
                    description=f"Gender prediction for {province} province only",
                    model_type="logistic_regression",
                    features=[FeatureType.FULL_NAME],
                    train_data_filter={"province": province},
                    tags=["province_study", province],
                )
            )

        return experiments
