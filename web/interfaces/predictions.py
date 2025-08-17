from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core.utils.data_loader import OPTIMIZED_DTYPES
from research.experiment.experiment_runner import ExperimentRunner
from research.experiment.experiment_tracker import ExperimentTracker


class Predictions:
    def __init__(
            self, config, experiment_tracker: ExperimentTracker, experiment_runner: ExperimentRunner
    ):
        self.config = config
        self.experiment_tracker = experiment_tracker
        self.experiment_runner = experiment_runner

    def index(self):
        st.title("Predictions")

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
        name_input = st.text_input("Enter a name:", placeholder="e.g., Jean Baptiste Mukendi")
        if name_input and st.button("Predict Gender"):
            try:
                # Load the model
                model = self.experiment_runner.load_experiment_model(experiment.experiment_id)

                if model is None:
                    st.error("Failed to load model")
                    return

                # Create a DataFrame with the input
                input_df = self._prepare_single_input(name_input)

                # Make prediction
                prediction = model.predict(input_df)[0]

                # Get prediction probability if available
                confidence = self._get_prediction_confidence(model, input_df)

                # Display results
                self._display_single_prediction_results(
                    prediction, confidence, experiment, name_input
                )

            except Exception as e:
                st.error(f"Error making prediction: {e}")

    def _prepare_single_input(self, name_input: str) -> pd.DataFrame:
        """Prepare single name input for prediction"""
        return pd.DataFrame(
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

    def _get_prediction_confidence(self, model, input_df: pd.DataFrame) -> Optional[float]:
        """Get prediction confidence if available"""
        try:
            probabilities = model.predict_proba(input_df)[0]
            return max(probabilities)
        except:
            return None

    def _display_single_prediction_results(
            self, prediction: str, confidence: Optional[float], experiment, name_input: str
    ):
        """Display single prediction results"""
        col1, col2 = st.columns(2)

        with col1:
            gender_label = "Female" if prediction == "f" else "Male"
            st.success(f"**Predicted Gender:** {gender_label}")

        with col2:
            if confidence:
                st.metric("Confidence", f"{confidence:.2%}")

        # Additional info
        st.info(f"Model used: {experiment.config.name}")
        st.info(f"Features used: {', '.join([f.value for f in experiment.config.features])}")

    def show_batch_prediction(self, experiment):
        uploaded_file = st.file_uploader("Upload CSV file with names", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, dtype=OPTIMIZED_DTYPES)

                st.write("**Uploaded Data Preview:**")
                st.dataframe(df.head(), use_container_width=True)

                # Column selection
                df = self._prepare_batch_data(df)

                if st.button("Run Batch Prediction"):
                    self._run_batch_prediction(df, experiment)

            except Exception as e:
                st.error(f"Error processing file: {e}")

    def _prepare_batch_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare batch data for prediction"""
        # Column selection
        if "name" not in df.columns:
            name_column = st.selectbox("Select the name column:", df.columns)
            df = df.rename(columns={name_column: "name"})

        # Add missing columns with defaults
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

        return df

    def _run_batch_prediction(self, df: pd.DataFrame, experiment):
        """Run batch prediction and display results"""
        with st.spinner("Making predictions..."):
            # Load model
            model = self.experiment_runner.load_experiment_model(experiment.experiment_id)

            if model is None:
                st.error("Failed to load model")
                return

            # Make predictions
            predictions = model.predict(df)
            df["predicted_gender"] = predictions
            df["gender_label"] = df["predicted_gender"].map({"f": "Female", "m": "Male"})

            # Try to get probabilities
            try:
                probabilities = model.predict_proba(df)
                df["confidence"] = np.max(probabilities, axis=1)
            except:
                df["confidence"] = None

        st.success("Predictions completed!")

        # Show results
        self._display_batch_results(df)

    def _display_batch_results(self, df: pd.DataFrame):
        """Display batch prediction results"""
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
        self._display_batch_summary(df)

    def _display_batch_summary(self, df: pd.DataFrame):
        """Display batch prediction summary"""
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

    def show_dataset_prediction(self, experiment):
        dataset_options = {
            "Featured Dataset": self.config.data.output_files["featured"],
            "Evaluation Dataset": self.config.data.output_files["evaluation"],
        }

        selected_dataset = st.selectbox("Select Dataset", list(dataset_options.keys()))
        file_path = self.config.paths.get_data_path(dataset_options[selected_dataset])

        if not file_path.exists():
            st.warning(f"Dataset not found: {file_path}")
            return

        # Load and show dataset info
        df = self._load_dataset(str(file_path))
        if df.empty:
            return

        st.write(f"Dataset contains {len(df):,} records")

        # Prediction options
        col1, col2 = st.columns(2)

        with col1:
            sample_size = st.number_input(
                "Sample size (0 = all data)", 0, len(df), min(1000, len(df))
            )

        with col2:
            compare_with_actual = False
            if "sex" in df.columns:
                compare_with_actual = st.checkbox("Compare with actual labels", value=True)

        if st.button("Run Dataset Prediction"):
            self._run_dataset_prediction(df, experiment, sample_size, compare_with_actual)

    def _load_dataset(self, file_path: str) -> pd.DataFrame:
        """Load dataset with error handling"""
        try:
            return pd.read_csv(file_path, dtype=OPTIMIZED_DTYPES)
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return pd.DataFrame()

    def _run_dataset_prediction(
            self, df: pd.DataFrame, experiment, sample_size: int, compare_with_actual: bool
    ):
        """Run dataset prediction and display results"""
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
                self._display_dataset_comparison(df_sample)
            else:
                self._display_dataset_predictions(df_sample)

    def _display_dataset_comparison(self, df_sample: pd.DataFrame):
        """Display dataset predictions with actual comparison"""
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
            correct_sample = df_sample[correct_mask][["name", "sex", "predicted_gender"]].head(10)
            st.dataframe(correct_sample, use_container_width=True)

        with col2:
            st.write("**Sample Incorrect Predictions**")
            incorrect_sample = df_sample[~correct_mask][["name", "sex", "predicted_gender"]].head(
                10
            )
            st.dataframe(incorrect_sample, use_container_width=True)

    def _display_dataset_predictions(self, df_sample: pd.DataFrame):
        """Display dataset predictions without comparison"""
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
