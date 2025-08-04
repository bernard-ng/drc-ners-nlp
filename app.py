#!.venv/bin/python3
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.config import get_config
from core.utils import get_data_file_path
from core.utils.data_loader import DataLoader
from core.utils.region_mapper import RegionMapper
from processing.monitoring.pipeline_monitor import PipelineMonitor
from research.experiment import ExperimentConfig
from research.experiment.experiment_builder import ExperimentBuilder
from research.experiment.experiment_runner import ExperimentRunner
from research.experiment.experiment_tracker import ExperimentTracker
from research.experiment.feature_extractor import FeatureType
from research.model_registry import list_available_models
from web.dashboard import Dashboard
from web.data_overview import DataOverview
from web.data_processing import DataProcessing

# Page configuration
st.set_page_config(
    page_title="DRC Names NLP Pipeline",
    page_icon="🇨🇩",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_config():
    """Load application configuration"""
    return get_config()


@st.cache_data
def load_dataset(file_path: str) -> pd.DataFrame:
    """Load dataset with caching"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()


class StreamlitApp:
    """Main Streamlit application class"""

    def __init__(self):
        self.config = load_config()
        self.data_loader = DataLoader(self.config)
        self.experiment_tracker = ExperimentTracker(self.config)
        self.experiment_runner = ExperimentRunner(self.config)
        self.pipeline_monitor = PipelineMonitor()

        # Initialize web components
        self.dashboard = Dashboard(self.config, self.experiment_tracker, self.experiment_runner)
        self.data_overview = DataOverview(self.config)
        self.data_processing = DataProcessing(self.config, self.pipeline_monitor)

        # Initialize session state
        if "current_experiment" not in st.session_state:
            st.session_state.current_experiment = None
        if "experiment_results" not in st.session_state:
            st.session_state.experiment_results = {}

    def run(self):
        st.title("🇨🇩 DRC NERS Pipeline")
        st.markdown("A comprehensive tool for Congolese name analysis and gender prediction")

        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigation",
            [
                "Dashboard",
                "Dataset Overview",
                "Data Processing",
                "Experiments",
                "Results & Analysis",
                "Predictions",
                "Configuration",
            ],
        )

        # Route to appropriate page
        if page == "Dashboard":
            self.dashboard.index()
        elif page == "Dataset Overview":
            self.data_overview.index()
        elif page == "Data Processing":
            self.data_processing.index()
        elif page == "Experiments":
            self.show_experiments()
        elif page == "Results & Analysis":
            self.show_results_analysis()
        elif page == "Predictions":
            self.show_predictions()
        elif page == "Configuration":
            self.show_configuration()

    def show_experiments(self):
        """Show experiment management interface"""
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
                if model_type == "logistic_regression":
                    ngram_min = st.number_input("N-gram Min", 1, 5, 2)
                    ngram_max = st.number_input("N-gram Max", 2, 8, 5)
                    max_features = st.number_input("Max Features", 1000, 50000, 10000)
                elif model_type == "random_forest":
                    n_estimators = st.number_input("Number of Trees", 10, 500, 100)
                    max_depth = st.number_input("Max Depth", 1, 20, 10)

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
                if not exp_name:
                    st.error("Please provide an experiment name")
                    return

                if not selected_features:
                    st.error("Please select at least one feature")
                    return

                # Build experiment configuration
                try:
                    # Prepare model parameters
                    model_params = {}
                    if model_type == "logistic_regression":
                        model_params = {
                            "ngram_range": [ngram_min, ngram_max],
                            "max_features": max_features,
                        }
                    elif model_type == "random_forest":
                        model_params = {
                            "n_estimators": n_estimators,
                            "max_depth": max_depth if max_depth > 0 else None,
                        }

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

        # Get experiments
        experiments = self.experiment_tracker.list_experiments()

        # Apply filters
        if status_filter != "All":
            from research.experiment import ExperimentStatus

            experiments = [e for e in experiments if e.status == ExperimentStatus(status_filter)]

        if model_filter != "All":
            experiments = [e for e in experiments if e.config.model_type == model_filter]

        if tag_filter:
            tags = [tag.strip() for tag in tag_filter.split(",")]
            experiments = [e for e in experiments if any(tag in e.config.tags for tag in tags)]

        if not experiments:
            st.info("No experiments found matching the filters.")
            return

        # Display experiments
        for i, exp in enumerate(experiments):
            with st.expander(
                    f"{exp.config.name} - {exp.status.value} - {exp.start_time.strftime('%Y-%m-%d %H:%M')}"
            ):
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

                    if st.button(f"View Details", key=f"details_{i}"):
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

    def show_results_analysis(self):
        """Show experiment results and analysis"""
        st.header("Results & Analysis")
        tab1, tab2, tab3 = st.tabs(["Experiment Comparison", "Performance Analysis", "Model Analysis"])

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
        completed_experiments = [e for e in experiments if e.status.value == "completed"]

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

        # Display comparison table
        st.write("**Experiment Comparison Table**")

        # Select columns to display
        metric_columns = [
            col for col in comparison_df.columns if col.startswith("test_") or col.startswith("cv_")
        ]
        display_columns = ["name", "model_type", "features"] + metric_columns
        available_columns = [col for col in display_columns if col in comparison_df.columns]

        st.dataframe(comparison_df[available_columns], use_container_width=True)

        # Visualization
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
        if len(metric_columns) > 1:
            metric_to_plot = st.selectbox("Select Metric for Detailed Comparison", metric_columns)

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

        analysis_df = pd.DataFrame(analysis_data)

        # Performance trends
        col1, col2 = st.columns(2)

        with col1:
            # Accuracy vs Training Size
            if "accuracy" in analysis_df.columns and "train_size" in analysis_df.columns:
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
            if "accuracy" in analysis_df.columns and "feature_count" in analysis_df.columns:
                fig = px.scatter(
                    analysis_df,
                    x="feature_count",
                    y="accuracy",
                    color="model_type",
                    hover_data=["name"],
                    title="Accuracy vs Number of Features",
                )
                st.plotly_chart(fig, use_container_width=True)

        # Model type comparison
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
                    error_y=dict(type="data", array=model_performance["std"]),
                    name="Average Accuracy",
                )
            )
            fig.update_layout(title="Average Accuracy by Model Type", yaxis_title="Accuracy")
            st.plotly_chart(fig, use_container_width=True)

        # Best experiments summary
        st.subheader("Top Performing Experiments")

        if "accuracy" in analysis_df.columns:
            top_experiments = analysis_df.nlargest(5, "accuracy")[
                ["name", "model_type", "features", "accuracy", "precision", "recall", "f1"]
            ]
            st.dataframe(top_experiments, use_container_width=True)

    def show_model_analysis(self):
        """Show detailed model analysis"""
        st.subheader("Model Analysis")

        experiments = self.experiment_tracker.list_experiments()
        completed_experiments = [e for e in experiments if e.status.value == "completed"]

        if not completed_experiments:
            st.warning("No completed experiments found.")
            return

        # Select experiment for detailed analysis
        exp_options = {
            f"{exp.config.name} ({exp.experiment_id[:8]})": exp for exp in completed_experiments
        }

        selected_exp_name = st.selectbox(
            "Select Experiment for Detailed Analysis", list(exp_options.keys())
        )

        if not selected_exp_name:
            return

        selected_exp = exp_options[selected_exp_name]

        # Experiment details
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Experiment Configuration**")
            st.json(
                {
                    "name": selected_exp.config.name,
                    "model_type": selected_exp.config.model_type,
                    "features": [f.value for f in selected_exp.config.features],
                    "model_params": selected_exp.config.model_params,
                }
            )

        with col2:
            st.write("**Performance Metrics**")
            if selected_exp.test_metrics:
                for metric, value in selected_exp.test_metrics.items():
                    st.metric(metric.title(), f"{value:.4f}")

        # Confusion matrix
        if selected_exp.confusion_matrix:
            st.write("**Confusion Matrix**")
            cm = np.array(selected_exp.confusion_matrix)

            fig = px.imshow(cm, text_auto=True, aspect="auto", title="Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)

        # Feature importance
        if selected_exp.feature_importance:
            st.write("**Feature Importance**")

            importance_data = sorted(
                selected_exp.feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:20]

            features, importances = zip(*importance_data)

            fig = px.bar(
                x=list(importances),
                y=list(features),
                orientation="h",
                title="Top 20 Feature Importances",
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

        # Prediction examples
        if selected_exp.prediction_examples:
            st.write("**Prediction Examples**")

            examples_df = pd.DataFrame(selected_exp.prediction_examples)

            # Separate correct and incorrect predictions
            correct_examples = examples_df[examples_df["correct"] == True]
            incorrect_examples = examples_df[examples_df["correct"] == False]

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Correct Predictions**")
                if not correct_examples.empty:
                    st.dataframe(
                        correct_examples[["name", "true_label", "predicted_label"]],
                        use_container_width=True,
                    )

            with col2:
                st.write("**Incorrect Predictions**")
                if not incorrect_examples.empty:
                    st.dataframe(
                        incorrect_examples[["name", "true_label", "predicted_label"]],
                        use_container_width=True,
                    )

    def show_predictions(self):
        """Show prediction interface"""
        st.header("Make Predictions")

        # Load available models
        experiments = self.experiment_tracker.list_experiments()
        completed_experiments = [
            e for e in experiments if e.status.value == "completed" and e.model_path
        ]

        if not completed_experiments:
            st.warning("No trained models available. Please run some experiments first.")
            return

        # Model selection
        model_options = {
            f"{exp.config.name} (Acc: {exp.test_metrics.get('accuracy', 0):.3f})": exp
            for exp in completed_experiments
            if exp.test_metrics
        }

        selected_model_name = st.selectbox("Select Model", list(model_options.keys()))

        if not selected_model_name:
            return

        selected_experiment = model_options[selected_model_name]

        # Prediction modes
        prediction_mode = st.radio(
            "Prediction Mode", ["Single Name", "Batch Upload", "Dataset Prediction"]
        )

        if prediction_mode == "Single Name":
            self.show_single_prediction(selected_experiment)
        elif prediction_mode == "Batch Upload":
            self.show_batch_prediction(selected_experiment)
        elif prediction_mode == "Dataset Prediction":
            self.show_dataset_prediction(selected_experiment)

    def show_single_prediction(self, experiment):
        """Show single name prediction interface"""
        st.subheader("Single Name Prediction")

        name_input = st.text_input("Enter a name:", placeholder="e.g., Jean Baptiste Mukendi")

        if name_input and st.button("Predict Gender"):
            try:
                # Load the model
                model = self.experiment_runner.load_experiment_model(experiment.experiment_id)

                if model is None:
                    st.error("Failed to load model")
                    return

                # Create a DataFrame with the input
                input_df = pd.DataFrame(
                    {
                        "name": [name_input],
                        "words": [len(name_input.split())],
                        "length": [len(name_input.replace(" ", ""))],
                        "province": ["unknown"],  # Default values
                        "identified_name": [None],
                        "identified_surname": [None],
                        "probable_native": [None],
                        "probable_surname": [None],
                    }
                )

                # Make prediction
                prediction = model.predict(input_df)[0]

                # Get prediction probability if available
                try:
                    probabilities = model.predict_proba(input_df)[0]
                    confidence = max(probabilities)
                except:
                    confidence = None

                # Display results
                col1, col2 = st.columns(2)

                with col1:
                    gender_label = "Female" if prediction == "f" else "Male"
                    st.success(f"**Predicted Gender:** {gender_label}")

                with col2:
                    if confidence:
                        st.metric("Confidence", f"{confidence:.2%}")

                # Additional info
                st.info(f"Model used: {experiment.batch_config.name}")
                st.info(
                    f"Features used: {', '.join([f.value for f in experiment.batch_config.features])}"
                )

            except Exception as e:
                st.error(f"Error making prediction: {e}")

    def show_batch_prediction(self, experiment):
        """Show batch prediction interface"""
        st.subheader("Batch Prediction")

        uploaded_file = st.file_uploader("Upload CSV file with names", type="csv")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                st.write("**Uploaded Data Preview:**")
                st.dataframe(df.head(), use_container_width=True)

                # Column selection
                if "name" not in df.columns:
                    name_column = st.selectbox("Select the name column:", df.columns)
                    df = df.rename(columns={name_column: "name"})

                if st.button("Run Batch Prediction"):
                    with st.spinner("Making predictions..."):
                        # Load model
                        model = self.experiment_runner.load_experiment_model(
                            experiment.experiment_id
                        )

                        if model is None:
                            st.error("Failed to load model")
                            return

                        # Prepare data (add missing columns with defaults)
                        required_columns = [
                            "words",
                            "length",
                            "province",
                            "identified_name",
                            "identified_surname",
                            "probable_native",
                            "probable_surname",
                        ]

                        for col in required_columns:
                            if col not in df.columns:
                                if col == "words":
                                    df[col] = df["name"].str.split().str.len()
                                elif col == "length":
                                    df[col] = df["name"].str.replace(" ", "").str.len()
                                else:
                                    df[col] = None

                        # Make predictions
                        predictions = model.predict(df)
                        df["predicted_gender"] = predictions
                        df["gender_label"] = df["predicted_gender"].map(
                            {"f": "Female", "m": "Male"}
                        )

                        # Try to get probabilities
                        try:
                            probabilities = model.predict_proba(df)
                            df["confidence"] = np.max(probabilities, axis=1)
                        except:
                            df["confidence"] = None

                    st.success("Predictions completed!")

                    # Show results
                    result_columns = ["name", "gender_label", "predicted_gender"]
                    if "confidence" in df.columns:
                        result_columns.append("confidence")

                    st.dataframe(df[result_columns], use_container_width=True)

                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions",
                        data=csv,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )

                    # Summary statistics
                    st.subheader("Prediction Summary")
                    gender_counts = df["gender_label"].value_counts()

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Predictions", len(df))
                    with col2:
                        st.metric("Female", gender_counts.get("Female", 0))
                    with col3:
                        st.metric("Male", gender_counts.get("Male", 0))

                    # Gender distribution chart
                    fig = px.pie(
                        values=gender_counts.values,
                        names=gender_counts.index,
                        title="Predicted Gender Distribution",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error processing file: {e}")

    def show_dataset_prediction(self, experiment):
        """Show dataset prediction interface"""
        st.subheader("Dataset Prediction")
        st.write("Apply the model to existing datasets")

        # Dataset selection
        dataset_options = {
            "Featured Dataset": self.config.data.output_files["featured"],
            "Evaluation Dataset": self.config.data.output_files["evaluation"],
        }

        selected_dataset = st.selectbox("Select Dataset", list(dataset_options.keys()))
        file_path = get_data_file_path(dataset_options[selected_dataset], self.config)

        if not file_path.exists():
            st.warning(f"Dataset not found: {file_path}")
            return

        # Load and show dataset info
        df = load_dataset(str(file_path))
        st.write(f"Dataset contains {len(df):,} records")

        # Prediction options
        col1, col2 = st.columns(2)

        with col1:
            sample_size = st.number_input(
                "Sample size (0 = all data)", 0, len(df), min(1000, len(df))
            )

        with col2:
            if "sex" in df.columns:
                compare_with_actual = st.checkbox("Compare with actual labels", value=True)
            else:
                compare_with_actual = False

        if st.button("Run Dataset Prediction"):
            with st.spinner("Running predictions..."):
                # Sample data if requested
                if sample_size > 0:
                    df_sample = df.sample(n=sample_size, random_state=42)
                else:
                    df_sample = df

                # Load model and make predictions
                model = self.experiment_runner.load_experiment_model(experiment.experiment_id)

                if model is None:
                    st.error("Failed to load model")
                    return

                predictions = model.predict(df_sample)
                df_sample["predicted_gender"] = predictions

                # Show results
                if compare_with_actual and "sex" in df_sample.columns:
                    # Calculate accuracy
                    accuracy = (df_sample["sex"] == df_sample["predicted_gender"]).mean()
                    st.metric("Accuracy on Selected Data", f"{accuracy:.4f}")

                    # Confusion matrix
                    from sklearn.metrics import confusion_matrix

                    cm = confusion_matrix(df_sample["sex"], df_sample["predicted_gender"])

                    fig = px.imshow(cm, text_auto=True, aspect="auto", title="Confusion Matrix")
                    st.plotly_chart(fig, use_container_width=True)

                    # Sample of correct and incorrect predictions
                    correct_mask = df_sample["sex"] == df_sample["predicted_gender"]

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Sample Correct Predictions**")
                        correct_sample = df_sample[correct_mask][
                            ["name", "sex", "predicted_gender"]
                        ].head(10)
                        st.dataframe(correct_sample, use_container_width=True)

                    with col2:
                        st.write("**Sample Incorrect Predictions**")
                        incorrect_sample = df_sample[~correct_mask][
                            ["name", "sex", "predicted_gender"]
                        ].head(10)
                        st.dataframe(incorrect_sample, use_container_width=True)

                else:
                    # Just show predictions
                    st.write("**Sample Predictions**")
                    sample_results = df_sample[["name", "predicted_gender"]].head(20)
                    st.dataframe(sample_results, use_container_width=True)

                    # Gender distribution
                    gender_counts = df_sample["predicted_gender"].value_counts()
                    fig = px.pie(
                        values=gender_counts.values,
                        names=gender_counts.index,
                        title="Predicted Gender Distribution",
                    )
                    st.plotly_chart(fig, use_container_width=True)

    def show_configuration(self):
        st.header("Current Configuration")
        st.json(self.config.model_dump())

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

    def clean_checkpoints(self):
        """Clean pipeline checkpoints"""
        for step in ["data_cleaning", "feature_extraction", "llm_annotation", "data_splitting"]:
            self.pipeline_monitor.clean_step_checkpoints(step, keep_last=1)
        st.success("Checkpoints cleaned!")

    def run_batch_experiments(
            self, base_name, model_types, ngram_ranges, feature_combinations, test_sizes, tags
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


def main():
    """Main application entry point"""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
