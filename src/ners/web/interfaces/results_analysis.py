from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ners.research.experiment.experiment_runner import ExperimentRunner
from ners.research.experiment.experiment_tracker import ExperimentTracker


class ResultsAnalysis:
    def __init__(
        self,
        config,
        experiment_tracker: ExperimentTracker,
        experiment_runner: ExperimentRunner,
    ):
        self.config = config
        self.experiment_tracker = experiment_tracker
        self.experiment_runner = experiment_runner

    def index(self):
        st.title("Results & Analysis")
        tab1, tab2, tab3 = st.tabs(
            ["Experiment Comparison", "Performance Analysis", "Model Analysis"]
        )

        with tab1:
            self.show_experiment_comparison()

        with tab2:
            self.show_performance_analysis()

        with tab3:
            self.show_model_analysis()

    def show_experiment_comparison(self):
        """Show experiment comparison interface"""
        st.subheader("Compare Experiments")

        experiments = self.experiment_tracker.list_experiments()
        completed_experiments = [
            e for e in experiments if e.status.value == "completed"
        ]

        if not completed_experiments:
            st.warning("No completed experiments found.")
            return

        # Experiment selection
        exp_options = {
            f"{exp.config.name} ({exp.experiment_id[:8]})": exp.experiment_id
            for exp in completed_experiments
        }

        selected_exp_names = st.multiselect(
            "Select Experiments to Compare",
            list(exp_options.keys()),
            default=list(exp_options.keys())[: min(5, len(exp_options))],
        )

        if not selected_exp_names:
            st.info("Please select experiments to compare.")
            return

        selected_exp_ids = [exp_options[name] for name in selected_exp_names]

        # Generate comparison
        comparison_df = self.experiment_runner.compare_experiments(selected_exp_ids)

        if comparison_df.empty:
            st.error("No data available for comparison.")
            return

        self._display_comparison_table(comparison_df)
        self._display_comparison_charts(comparison_df)

    def _display_comparison_table(self, comparison_df: pd.DataFrame):
        """Display comparison table"""
        st.write("**Experiment Comparison Table**")

        # Select columns to display
        metric_columns = [
            col
            for col in comparison_df.columns
            if col.startswith("test_") or col.startswith("cv_")
        ]
        display_columns = ["name", "model_type", "features"] + metric_columns
        available_columns = [
            col for col in display_columns if col in comparison_df.columns
        ]

        st.dataframe(comparison_df[available_columns], use_container_width=True)

    def _display_comparison_charts(self, comparison_df: pd.DataFrame):
        """Display comparison charts"""
        st.write("**Performance Comparison**")

        if "test_accuracy" in comparison_df.columns:
            fig = px.bar(
                comparison_df,
                x="name",
                y="test_accuracy",
                color="model_type",
                title="Test Accuracy Comparison",
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        # Metric comparison across multiple metrics
        metric_columns = [
            col
            for col in comparison_df.columns
            if col.startswith("test_") or col.startswith("cv_")
        ]

        if len(metric_columns) > 1:
            metric_to_plot = st.selectbox(
                "Select Metric for Detailed Comparison", metric_columns
            )

            if metric_to_plot in comparison_df.columns:
                fig = px.bar(
                    comparison_df,
                    x="name",
                    y=metric_to_plot,
                    color="model_type",
                    title=f"{metric_to_plot.replace('_', ' ').title()} Comparison",
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

    def show_performance_analysis(self):
        """Show performance analysis across experiments"""
        st.subheader("Performance Analysis")

        experiments = self.experiment_tracker.list_experiments()
        completed_experiments = [
            e for e in experiments if e.status.value == "completed" and e.test_metrics
        ]

        if not completed_experiments:
            st.warning("No completed experiments with metrics found.")
            return

        # Prepare data for analysis
        analysis_data = self._prepare_analysis_data(completed_experiments)
        analysis_df = pd.DataFrame(analysis_data)

        self._display_performance_trends(analysis_df)
        self._display_model_comparison(analysis_df)
        self._display_top_experiments(analysis_df)

    def _prepare_analysis_data(self, completed_experiments: List) -> List[dict]:
        """Prepare data for performance analysis"""
        analysis_data = []
        for exp in completed_experiments:
            row = {
                "experiment_id": exp.experiment_id,
                "name": exp.config.name,
                "model_type": exp.config.model_type,
                "feature_count": len(exp.config.features),
                "features": ", ".join([f.value for f in exp.config.features]),
                "train_size": exp.train_size,
                "test_size": exp.test_size,
                **exp.test_metrics,
            }
            analysis_data.append(row)
        return analysis_data

    def _display_performance_trends(self, analysis_df: pd.DataFrame):
        """Display performance trend charts"""
        col1, col2 = st.columns(2)

        with col1:
            # Accuracy vs Training Size
            if (
                "accuracy" in analysis_df.columns
                and "train_size" in analysis_df.columns
            ):
                fig = px.scatter(
                    analysis_df,
                    x="train_size",
                    y="accuracy",
                    color="model_type",
                    hover_data=["name"],
                    title="Accuracy vs Training Size",
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Feature Count vs Performance
            if (
                "accuracy" in analysis_df.columns
                and "feature_count" in analysis_df.columns
            ):
                fig = px.scatter(
                    analysis_df,
                    x="feature_count",
                    y="accuracy",
                    color="model_type",
                    hover_data=["name"],
                    title="Accuracy vs Number of Features",
                )
                st.plotly_chart(fig, use_container_width=True)

    def _display_model_comparison(self, analysis_df: pd.DataFrame):
        """Display model type comparison"""
        if "accuracy" in analysis_df.columns:
            model_performance = (
                analysis_df.groupby("model_type")["accuracy"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=model_performance["model_type"],
                    y=model_performance["mean"],
                    error_y=dict(type="data", array=model_performance["std"].fillna(0)),
                    name="Accuracy",
                )
            )

            st.plotly_chart(fig, use_container_width=True)

    def _display_top_experiments(self, analysis_df: pd.DataFrame):
        """Display top-performing experiments"""
        if "accuracy" in analysis_df.columns:
            top_n = st.slider("Select Top N Experiments", 3, 20, 5)
            top_experiments = analysis_df.nlargest(top_n, "accuracy")

            st.write("**Top Performing Experiments:**")
            st.dataframe(
                top_experiments[
                    [
                        "name",
                        "model_type",
                        "features",
                        "train_size",
                        "test_size",
                        "accuracy",
                    ]
                ],
                use_container_width=True,
            )

    def show_model_analysis(self):
        """Show detailed model analysis interface"""
        st.subheader("Model Analysis")

        experiments = self.experiment_tracker.list_experiments()
        completed_experiments = [
            e for e in experiments if e.status.value == "completed"
        ]

        if not completed_experiments:
            st.warning("No completed experiments found for analysis.")
            return

        # Model selection
        exp_options = {
            f"{exp.config.name} ({exp.experiment_id[:8]})": exp.experiment_id
            for exp in completed_experiments
        }
        selected_exp_name = st.selectbox(
            "Select Model for Analysis", list(exp_options.keys())
        )
        if not selected_exp_name:
            return

        exp_id = exp_options[selected_exp_name]
        experiment = self.experiment_tracker.get_experiment(exp_id)

        if not experiment or not experiment.test_metrics:
            st.warning("Selected experiment has no evaluation metrics.")
            return

        # Display detailed metrics
        st.write("**Detailed Metrics:**")
        st.json(experiment.test_metrics)
