from typing import List, Dict

import streamlit as st

from core.config.pipeline_config import PipelineConfig
from research.experiment import ExperimentConfig, ExperimentStatus
from research.experiment.experiment_builder import ExperimentBuilder
from research.experiment.experiment_runner import ExperimentRunner
from research.experiment.experiment_tracker import ExperimentTracker
from research.experiment.feature_extractor import FeatureType
from research.model_registry import list_available_models


class Experiments:
    def __init__(
        self,
        config: PipelineConfig,
        experiment_tracker: ExperimentTracker,
        experiment_runner: ExperimentRunner,
    ):
        self.config = config
        self.experiment_tracker = experiment_tracker
        self.experiment_runner = experiment_runner
        self.experiment_builder = ExperimentBuilder(config)

    def index(self):
        st.title("Experiments")

        tab1, tab2, tab3 = st.tabs(["Templates", "Experiments", "Batch Experiments"])

        with tab1:
            self.show_template_experiments()

        with tab2:
            self.show_experiment_list()

        with tab3:
            self.show_batch_experiments()

    def show_template_experiments(self):
        """Show interface for running predefined template experiments"""
        st.subheader("Template Experiments")
        st.write("Run predefined experiments based on research templates.")

        try:
            available_experiments = self.experiment_builder.get_templates()

            # Create tabs for different experiment types
            exp_tabs = st.tabs(["Baseline", "Advanced", "Feature Studies", "Hyperparameter Tuning"])

            with exp_tabs[0]:
                self._show_experiments_by_type(available_experiments["baseline"], "baseline")

            with exp_tabs[1]:
                self._show_experiments_by_type(available_experiments["advanced"], "advanced")

            with exp_tabs[2]:
                self._show_experiments_by_type(
                    available_experiments["feature_study"], "feature_study"
                )

            with exp_tabs[3]:
                self._show_experiments_by_type(available_experiments["tuning"], "tuning")

        except Exception as e:
            st.error(f"Error loading experiment templates: {e}")
            st.info(
                "Make sure the research templates file exists at `config/research_templates.yaml`"
            )

    def _show_experiments_by_type(self, experiments: List[Dict], experiment_type: str):
        """Show experiments for a specific type"""
        if not experiments:
            st.info(f"No {experiment_type} experiments available in templates.")
            return

        st.write(f"**{experiment_type.title()} Experiments**")

        # Show available experiments
        for i, exp_template in enumerate(experiments):
            exp_name = exp_template.get("name", f"Experiment {i + 1}")
            exp_description = exp_template.get("description", "No description available")

            with st.expander(f"📊 {exp_name} - {exp_description}"):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.json(exp_template)

                with col2:
                    if st.button(f"🚀 Run Experiment", key=f"run_{experiment_type}_{i}"):
                        self._run_template_experiment(exp_template)

    def _run_template_experiment(self, exp_template: Dict):
        """Run a template experiment"""
        try:
            with st.spinner(f"Running {exp_template.get('name')}..."):
                # Create experiment config from template
                experiment_config = self.experiment_builder.from_template(exp_template)

                # Run the experiment
                experiment_id = self.experiment_runner.run_experiment(experiment_config)
                st.success(f"Experiment '{experiment_config.name}' completed successfully!")
                st.info(f"Experiment ID: `{experiment_id}`")

                # Show results
                experiment = self.experiment_tracker.get_experiment(experiment_id)
                if experiment and experiment.test_metrics:
                    st.write("**Results:**")
                    col1, col2, col3 = st.columns(3)

                    metrics = list(experiment.test_metrics.items())
                    for i, (metric, value) in enumerate(metrics):
                        with [col1, col2, col3][i % 3]:
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

    @classmethod
    def _display_experiment_details(cls, exp, index: int):
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

        # Add option to run template batch experiments
        batch_type = st.radio("Batch Type", ["Template Batch", "Custom Parameter Sweep"])

        if batch_type == "Template Batch":
            self._show_template_batch_experiments()
        else:
            self._show_custom_batch_experiments()

    def _show_template_batch_experiments(self):
        """Show interface for running batch experiments from templates"""
        st.write("**Run Multiple Template Experiments**")

        try:
            available_experiments = self.experiment_builder.get_templates()

            # Select experiment types to run
            experiment_types = st.multiselect(
                "Select Experiment Types",
                ["baseline", "advanced", "feature_study", "tuning"],
                default=["baseline"],
            )

            if experiment_types:
                selected_experiments = []

                for exp_type in experiment_types:
                    experiments = available_experiments.get(exp_type, [])
                    if experiments:
                        st.write(f"**{exp_type.title()} Experiments:**")
                        exp_names = [
                            exp.get("name", f"Exp {i}") for i, exp in enumerate(experiments)
                        ]
                        selected_names = st.multiselect(
                            f"Select {exp_type} experiments", exp_names, key=f"select_{exp_type}"
                        )

                        for name in selected_names:
                            for exp in experiments:
                                if exp.get("name") == name:
                                    selected_experiments.append(exp)

                if st.button("🚀 Run Selected Template Experiments"):
                    self._run_template_batch_experiments(selected_experiments)

        except Exception as e:
            st.error(f"Error loading templates for batch experiments: {e}")

    def _run_template_batch_experiments(self, selected_experiments: List[Dict]):
        """Run batch experiments from templates"""
        if not selected_experiments:
            st.warning("No experiments selected")
            return

        with st.spinner(f"Running {len(selected_experiments)} template experiments..."):
            try:
                experiment_configs = []
                for exp_template in selected_experiments:
                    config = self.experiment_builder.from_template(exp_template)
                    experiment_configs.append(config)

                # Run batch experiments
                experiment_ids = self.experiment_runner.run_experiment_batch(experiment_configs)

                st.success(f"Completed {len(experiment_ids)} template experiments!")

                # Show summary
                if experiment_ids:
                    comparison = self.experiment_runner.compare_experiments(experiment_ids)
                    st.write("**Template Batch Results:**")
                    st.dataframe(
                        comparison[["name", "model_type", "test_accuracy"]],
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"Error running template batch experiments: {e}")

    def _show_custom_batch_experiments(self):
        """Show interface for custom parameter sweep experiments"""
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

            if st.form_submit_button("🚀 Run Parameter Sweep"):
                self.run_batch_experiments(
                    base_name, model_types, ngram_ranges, feature_combinations, test_sizes, tags
                )

    def run_batch_experiments(
        self,
        base_name: str,
        model_types: List[str],
        ngram_ranges: str,
        feature_combinations: List[str],
        test_sizes: str,
        tags: str,
    ):
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
