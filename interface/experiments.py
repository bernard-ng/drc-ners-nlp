from typing import List, Dict, Any

import streamlit as st

from core.utils.region_mapper import RegionMapper
from research.experiment import ExperimentConfig, ExperimentStatus
from research.experiment.experiment_builder import ExperimentBuilder
from research.experiment.experiment_runner import ExperimentRunner
from research.experiment.experiment_tracker import ExperimentTracker
from research.experiment.feature_extractor import FeatureType
from research.model_registry import list_available_models


class Experiments:
    """Handles experiment management interface"""

    def __init__(self, config, experiment_tracker: ExperimentTracker, experiment_runner: ExperimentRunner):
        self.config = config
        self.experiment_tracker = experiment_tracker
        self.experiment_runner = experiment_runner

    def index(self):
        """Main experiments page"""
        st.header("Experiment Management")
        tab1, tab2, tab3 = st.tabs(["New Experiment", "Experiment List", "Batch Experiments"])

        with tab1:
            self.show_experiment_creation()

        with tab2:
            self.show_experiment_list()

        with tab3:
            self.show_batch_experiments()

    def show_experiment_creation(self):
        """Show interface for creating new experiments"""
        st.subheader("Create New Experiment")

        with st.form("new_experiment"):
            col1, col2 = st.columns(2)

            with col1:
                exp_name = st.text_input("Experiment Name", placeholder="e.g., native_name_gender_prediction")
                description = st.text_area("Description", placeholder="Brief description of the experiment")
                model_type = st.selectbox("Model Type", list_available_models())

                # Feature selection
                feature_options = [f.value for f in FeatureType]
                selected_features = st.multiselect("Features to Use", feature_options, default=["full_name"])

            with col2:
                # Model parameters
                st.write("**Model Parameters**")
                model_params = {}
                if model_type == "logistic_regression":
                    ngram_min = st.number_input("N-gram Min", 1, 5, 2)
                    ngram_max = st.number_input("N-gram Max", 2, 8, 5)
                    max_features = st.number_input("Max Features", 1000, 50000, 10000)
                    model_params = {
                        "ngram_range": [ngram_min, ngram_max],
                        "max_features": max_features,
                    }
                elif model_type == "random_forest":
                    n_estimators = st.number_input("Number of Trees", 10, 500, 100)
                    max_depth = st.number_input("Max Depth", 1, 20, 10)
                    model_params = {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth if max_depth > 0 else None,
                    }

                # Training parameters
                st.write("**Training Parameters**")
                test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2)
                cv_folds = st.number_input("Cross-Validation Folds", 3, 10, 5)

                tags = st.text_input("Tags (comma-separated)", placeholder="e.g., baseline, feature_study")

            # Advanced options
            with st.expander("Advanced Options"):
                # Data filters
                st.write("**Data Filters**")
                filter_province = st.selectbox(
                    "Filter by Province (optional)",
                    ["None"] + RegionMapper().get_provinces(),
                )

                min_words = st.number_input("Minimum Word Count", 0, 10, 0)
                max_words = st.number_input("Maximum Word Count (0 = no limit)", 0, 20, 0)

            submitted = st.form_submit_button("Create and Run Experiment", type="primary")

            if submitted:
                self._handle_experiment_submission(
                    exp_name, description, model_type, selected_features, model_params,
                    test_size, cv_folds, tags, filter_province, min_words, max_words
                )

    def _handle_experiment_submission(self, exp_name: str, description: str, model_type: str,
                                      selected_features: List[str], model_params: Dict[str, Any],
                                      test_size: float, cv_folds: int, tags: str,
                                      filter_province: str, min_words: int, max_words: int):
        """Handle experiment form submission"""
        if not exp_name:
            st.error("Please provide an experiment name")
            return

        if not selected_features:
            st.error("Please select at least one feature")
            return

        try:
            # Prepare data filters
            train_filter = {}
            if filter_province != "None":
                train_filter["province"] = filter_province
            if min_words > 0:
                train_filter["words"] = {"min": min_words}
            if max_words > 0:
                if "words" in train_filter:
                    train_filter["words"]["max"] = max_words
                else:
                    train_filter["words"] = {"max": max_words}

            # Create experiment config
            features = [FeatureType(f) for f in selected_features]
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]

            config = ExperimentConfig(
                name=exp_name,
                description=description,
                tags=tag_list,
                model_type=model_type,
                model_params=model_params,
                features=features,
                train_data_filter=train_filter if train_filter else None,
                test_size=test_size,
                cross_validation_folds=cv_folds,
            )

            # Run experiment
            with st.spinner("Running experiment..."):
                experiment_id = self.experiment_runner.run_experiment(config)

            st.success(f"Experiment completed successfully!")
            st.info(f"Experiment ID: `{experiment_id}`")

            # Show results
            experiment = self.experiment_tracker.get_experiment(experiment_id)
            if experiment and experiment.test_metrics:
                st.write("**Results:**")
                for metric, value in experiment.test_metrics.items():
                    st.metric(metric.title(), f"{value:.4f}")

        except Exception as e:
            st.error(f"Error running experiment: {e}")

    def show_experiment_list(self):
        """Show list of all experiments with filtering"""
        st.subheader("All Experiments")

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            status_filter = st.selectbox(
                "Filter by Status", ["All", "completed", "running", "failed", "pending"]
            )

        with col2:
            model_filter = st.selectbox("Filter by Model", ["All"] + list_available_models())

        with col3:
            tag_filter = st.text_input("Filter by Tags (comma-separated)")

        # Get and filter experiments
        experiments = self._get_filtered_experiments(status_filter, model_filter, tag_filter)

        if not experiments:
            st.info("No experiments found matching the filters.")
            return

        # Display experiments
        for i, exp in enumerate(experiments):
            with st.expander(
                    f"{exp.config.name} - {exp.status.value} - {exp.start_time.strftime('%Y-%m-%d %H:%M')}"
            ):
                self._display_experiment_details(exp, i)

    def _get_filtered_experiments(self, status_filter: str, model_filter: str, tag_filter: str):
        """Get experiments with applied filters"""
        experiments = self.experiment_tracker.list_experiments()

        # Apply filters
        if status_filter != "All":
            experiments = [e for e in experiments if e.status == ExperimentStatus(status_filter)]

        if model_filter != "All":
            experiments = [e for e in experiments if e.config.model_type == model_filter]

        if tag_filter:
            tags = [tag.strip() for tag in tag_filter.split(",")]
            experiments = [e for e in experiments if any(tag in e.config.tags for tag in tags)]

        return experiments

    def _display_experiment_details(self, exp, index: int):
        """Display details for a single experiment"""
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write(f"**Model:** {exp.config.model_type}")
            st.write(f"**Features:** {', '.join([f.value for f in exp.config.features])}")
            st.write(f"**Tags:** {', '.join(exp.config.tags)}")

        with col2:
            if exp.test_metrics:
                for metric, value in exp.test_metrics.items():
                    st.metric(metric.title(), f"{value:.4f}")

        with col3:
            st.write(f"**Train Size:** {exp.train_size:,}")
            st.write(f"**Test Size:** {exp.test_size:,}")

            if st.button(f"View Details", key=f"details_{index}"):
                st.session_state.selected_experiment = exp.experiment_id
                st.rerun()

        if exp.config.description:
            st.write(f"**Description:** {exp.config.description}")

    def show_batch_experiments(self):
        """Show interface for running batch experiments"""
        st.subheader("Batch Experiments")
        st.write("Run multiple experiments with different parameter combinations.")

        # Parameter sweep configuration
        with st.form("batch_experiments"):
            st.write("**Parameter Sweep Configuration**")

            col1, col2 = st.columns(2)

            with col1:
                base_name = st.text_input("Base Experiment Name", "parameter_sweep")
                model_types = st.multiselect(
                    "Model Types", list_available_models(), default=["logistic_regression"]
                )

                # N-gram ranges for logistic regression
                st.write("**Logistic Regression Parameters**")
                ngram_ranges = st.text_area(
                    "N-gram Ranges (one per line, format: min,max)", "2,4\n2,5\n3,6"
                )

            with col2:
                feature_combinations = st.multiselect(
                    "Feature Combinations",
                    [f.value for f in FeatureType],
                    default=["full_name", "native_name", "surname"],
                )

                test_sizes = st.text_input("Test Sizes (comma-separated)", "0.15,0.2,0.25")

                tags = st.text_input("Common Tags", "parameter_sweep,batch")

            if st.form_submit_button("🚀 Run Batch Experiments"):
                self.run_batch_experiments(
                    base_name, model_types, ngram_ranges, feature_combinations, test_sizes, tags
                )

    def run_batch_experiments(self, base_name: str, model_types: List[str], ngram_ranges: str,
                              feature_combinations: List[str], test_sizes: str, tags: str):
        """Run batch experiments with parameter combinations"""
        with st.spinner("Running batch experiments..."):
            try:
                experiments = []

                # Parse parameters
                ngram_list = []
                for line in ngram_ranges.strip().split("\n"):
                    if "," in line:
                        min_val, max_val = map(int, line.split(","))
                        ngram_list.append([min_val, max_val])

                test_size_list = [float(x.strip()) for x in test_sizes.split(",")]
                tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]

                # Generate experiment combinations
                exp_count = 0
                for model_type in model_types:
                    for feature_combo in feature_combinations:
                        for test_size in test_size_list:
                            if model_type == "logistic_regression":
                                for ngram_range in ngram_list:
                                    exp_name = f"{base_name}_{model_type}_{feature_combo}_{ngram_range[0]}_{ngram_range[1]}_{test_size}"

                                    config = ExperimentConfig(
                                        name=exp_name,
                                        description=f"Batch experiment: {model_type} with {feature_combo}",
                                        model_type=model_type,
                                        features=[FeatureType(feature_combo)],
                                        model_params={"ngram_range": ngram_range},
                                        test_size=test_size,
                                        tags=tag_list,
                                    )
                                    experiments.append(config)
                                    exp_count += 1
                            else:
                                exp_name = f"{base_name}_{model_type}_{feature_combo}_{test_size}"

                                config = ExperimentConfig(
                                    name=exp_name,
                                    description=f"Batch experiment: {model_type} with {feature_combo}",
                                    model_type=model_type,
                                    features=[FeatureType(feature_combo)],
                                    test_size=test_size,
                                    tags=tag_list,
                                )
                                experiments.append(config)
                                exp_count += 1

                # Run experiments
                experiment_ids = self.experiment_runner.run_experiment_batch(experiments)

                st.success(f"Completed {len(experiment_ids)} batch experiments")

                # Show summary
                if experiment_ids:
                    comparison = self.experiment_runner.compare_experiments(experiment_ids)
                    st.write("**Batch Results Summary:**")
                    st.dataframe(
                        comparison[["name", "model_type", "test_accuracy"]],
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"Error running batch experiments: {e}")

    def run_baseline_experiments(self):
        """Run baseline experiments"""
        with st.spinner("Running baseline experiments..."):
            try:
                builder = ExperimentBuilder()
                experiments = builder.create_baseline_experiments()
                experiment_ids = self.experiment_runner.run_experiment_batch(experiments)

                st.success(f"Completed {len(experiment_ids)} baseline experiments")

                # Show quick comparison
                if experiment_ids:
                    comparison = self.experiment_runner.compare_experiments(experiment_ids)
                    st.write("**Results Summary:**")
                    st.dataframe(
                        comparison[["name", "model_type", "test_accuracy"]],
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"Error running baseline experiments: {e}")

    def run_ablation_study(self):
        """Run feature ablation study"""
        with st.spinner("Running ablation study..."):
            try:
                builder = ExperimentBuilder()
                experiments = builder.create_feature_ablation_study()
                experiment_ids = self.experiment_runner.run_experiment_batch(experiments)

                st.success(f"Completed {len(experiment_ids)} ablation experiments")

            except Exception as e:
                st.error(f"Error running ablation study: {e}")

    def run_component_study(self):
        """Run name component study"""
        with st.spinner("Running component study..."):
            try:
                builder = ExperimentBuilder()
                experiments = builder.create_name_component_study()
                experiment_ids = self.experiment_runner.run_experiment_batch(experiments)

                st.success(f"Completed {len(experiment_ids)} component experiments")

            except Exception as e:
                st.error(f"Error running component study: {e}")

    def run_province_study(self):
        """Run province-specific study"""
        with st.spinner("Running province study..."):
            try:
                builder = ExperimentBuilder()
                experiments = builder.create_province_specific_study()
                experiment_ids = self.experiment_runner.run_experiment_batch(experiments)

                st.success(f"Completed {len(experiment_ids)} province experiments")

            except Exception as e:
                st.error(f"Error running province study: {e}")
