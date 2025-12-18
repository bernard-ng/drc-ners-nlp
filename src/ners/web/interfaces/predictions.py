from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

from ners.core.utils.data_loader import OPTIMIZED_DTYPES
from ners.research.experiment.experiment_runner import ExperimentRunner
from ners.research.experiment.experiment_tracker import ExperimentTracker


class Predictions:
    """
    Streamlit interface for running predictions using trained experiments.

    Supports single-name prediction, batch CSV uploads, and full dataset predictions
    using previously completed experiments.
    """

    def __init__(
        self,
        config: Any,
        experiment_tracker: ExperimentTracker,
        experiment_runner: ExperimentRunner,
    ):
        """
        Initialize the Predictions interface.

        Args:
            config: Application configuration object.
            experiment_tracker: Tracker providing access to past experiments.
            experiment_runner: Runner responsible for loading trained models.
        """
        self.config = config
        self.experiment_tracker = experiment_tracker
        self.experiment_runner = experiment_runner

    def index(self) -> None:
        """
        Render the main Predictions page.

        Allows users to select a trained model and choose between single,
        batch, or dataset-level prediction modes.
        """
        st.title("Predictions")

        experiments = self.experiment_tracker.list_experiments()
        completed_experiments = [
            e for e in experiments if e.status.value == "completed" and e.model_path
        ]

        if not completed_experiments:
            st.warning(
                "No trained models available. Please run some experiments first."
            )
            return

        model_options = {
            f"{exp.config.name} (Acc: {exp.test_metrics.get('accuracy', 0):.3f})": exp
            for exp in completed_experiments
            if exp.test_metrics
        }

        selected_model_name = st.selectbox("Select Model", list(model_options.keys()))
        if not selected_model_name:
            return

        selected_experiment = model_options[selected_model_name]

        prediction_mode = st.radio(
            "Prediction Mode",
            ["Single Name", "Batch Upload", "Dataset Prediction"],
        )

        if prediction_mode == "Single Name":
            self.show_single_prediction(selected_experiment)
        elif prediction_mode == "Batch Upload":
            self.show_batch_prediction(selected_experiment)
        elif prediction_mode == "Dataset Prediction":
            self.show_dataset_prediction(selected_experiment)

    def show_single_prediction(self, experiment: Any) -> None:
        """
        Render the UI for predicting gender from a single name.
        """
        name_input = st.text_input(
            "Enter a name:", placeholder="e.g., Jean Baptiste Mukendi"
        )

        if name_input and st.button("Predict Gender"):
            try:
                model = self.experiment_runner.load_experiment_model(
                    experiment.experiment_id
                )

                if model is None:
                    st.error("Failed to load model")
                    return

                input_df = self._prepare_single_input(name_input)
                prediction = model.predict(input_df)[0]
                confidence = self._get_prediction_confidence(model, input_df)

                self._display_single_prediction_results(
                    prediction, confidence, experiment
                )

            except Exception as e:
                st.error(f"Error making prediction: {e}")

    def _prepare_single_input(self, name_input: str) -> pd.DataFrame:
        """
        Prepare a single-name DataFrame compatible with the prediction model.
        """
        return pd.DataFrame(
            {
                "name": [name_input],
                "words": [len(name_input.split())],
                "length": [len(name_input.replace(" ", ""))],
                "province": ["unknown"],
                "identified_name": [None],
                "identified_surname": [None],
                "probable_native": [None],
                "probable_surname": [None],
            }
        )

    def _get_prediction_confidence(
        self, model: Any, input_df: pd.DataFrame
    ) -> Optional[float]:
        """
        Extract prediction confidence from the model if probabilities are supported.
        """
        try:
            probabilities = model.predict_proba(input_df)[0]
            return float(max(probabilities))
        except Exception:
            return None

    def _display_single_prediction_results(
        self,
        prediction: str,
        confidence: Optional[float],
        experiment: Any,
    ) -> None:
        """
        Display prediction results for a single name.
        """
        col1, col2 = st.columns(2)

        with col1:
            gender_label = "Female" if prediction == "f" else "Male"
            st.success(f"**Predicted Gender:** {gender_label}")

        with col2:
            if confidence is not None:
                st.metric("Confidence", f"{confidence:.2%}")

        st.info(f"Model used: {experiment.config.name}")
        st.info(
            f"Features used: {', '.join(f.value for f in experiment.config.features)}"
        )

    def show_batch_prediction(self, experiment: Any) -> None:
        """
        Render the UI for batch predictions using an uploaded CSV file.
        """
        uploaded_file = st.file_uploader("Upload CSV file with names", type="csv")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(
                    uploaded_file,
                    dtype=OPTIMIZED_DTYPES,  # type: ignore[arg-type]
                )

                st.write("**Uploaded Data Preview:**")
                st.dataframe(df.head(), use_container_width=True)

                df = self._prepare_batch_data(df)

                if st.button("Run Batch Prediction"):
                    self._run_batch_prediction(df, experiment)

            except Exception as e:
                st.error(f"Error processing file: {e}")

    def _prepare_batch_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare batch input data by ensuring required feature columns exist.
        """
        if "name" not in df.columns:
            name_column = st.selectbox("Select the name column:", df.columns)
            df = df.rename(columns={name_column: "name"})

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
                    df[col] = df["name"].str.replace(" ", "", regex=False).str.len()
                else:
                    df[col] = None

        return df

    def _run_batch_prediction(self, df: pd.DataFrame, experiment: Any) -> None:
        """
        Run predictions on a batch DataFrame and display the results.
        """
        with st.spinner("Making predictions..."):
            model = self.experiment_runner.load_experiment_model(
                experiment.experiment_id
            )

            if model is None:
                st.error("Failed to load model")
                return

            predictions = model.predict(df)
            df["predicted_gender"] = predictions
            df["gender_label"] = df["predicted_gender"].map(
                lambda v: "Female" if v == "f" else "Male"
            )

            try:
                probabilities = model.predict_proba(df)
                df["confidence"] = np.max(probabilities, axis=1)
            except Exception:
                df["confidence"] = None

        st.success("Predictions completed!")
        st.dataframe(df, use_container_width=True)

    def show_dataset_prediction(self, experiment: Any) -> None:
        """
        Run predictions on the full featured dataset and display a preview.
        """
        data_path = self.config.paths.get_data_path(
            self.config.data.output_files["featured"]
        )

        if not data_path.exists():
            st.warning("Featured dataset not found. Please generate it first.")
            return

        try:
            df = pd.read_csv(
                data_path,
                dtype=OPTIMIZED_DTYPES,  # type: ignore[arg-type]
            )

            df = self._prepare_batch_data(df)
            self._run_batch_prediction(df, experiment)

        except Exception as e:
            st.error(f"Error running dataset prediction: {e}")
