import streamlit as st
from spacy import displacy
from typing import Any, Optional

from ners.core.config import PipelineConfig
from ners.processing.ner.name_model import NameModel


class NERTesting:
    """
    Streamlit interface for testing a trained Named Entity Recognition (NER) model.

    Allows users to load a trained model, inspect training/evaluation metadata,
    and interactively test name entity extraction on single or multiple inputs.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the NER testing interface.

        Args:
            config: Pipeline configuration containing paths and model settings.
        """
        self.config = config
        self.model_path = config.paths.models_dir / "drc_ner_model"
        self.ner_model: Optional[NameModel] = None
        self.training_stats: Optional[dict[str, Any]] = None
        self.evaluation_stats: Optional[dict[str, Any]] = None

    def load_ner_model(self) -> bool:
        """
        Load the trained NER model from disk.

        Initializes the NameModel instance if not already loaded and
        populates training and evaluation statistics.

        Returns:
            True if the model was successfully loaded, False otherwise.
        """
        try:
            if self.ner_model is None:
                model = NameModel(self.config)
                model.load(str(self.model_path))
                self.ner_model = model
                self.training_stats = model.training_stats
                self.evaluation_stats = {}
            return True
        except Exception as e:
            st.error(f"Error loading NER model: {e}")
            return False

    def index(self) -> None:
        """
        Render the main NER testing page.

        Handles model loading, displays model metadata, and provides
        interactive controls for testing single or multiple names.
        """
        st.title("Named Entity Recognition")

        # Load model
        if not self.load_ner_model():
            st.warning(
                "NER model could not be loaded. Please ensure the model is trained and available."
            )
            return

        # Display model information
        self.show_model_training_info()
        self.show_model_evaluation_info()

        st.markdown("---")
        st.subheader("Test the NER Model")

        input_method = st.radio("Input Method", ["Single Name", "Multiple Names"])
        if input_method == "Single Name":
            self.test_single_name()
        elif input_method == "Multiple Names":
            self.test_multiple_names()

    def show_model_training_info(self) -> None:
        """
        Display training statistics of the loaded NER model.
        """
        if self.training_stats:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Training Examples",
                    f"{self.training_stats.get('training_examples', 0):,}",
                )
            with col2:
                st.metric("Epochs", self.training_stats.get("epochs", 0))
            with col3:
                st.metric(
                    "Final Loss", f"{self.training_stats.get('final_loss', 0):.2f}"
                )
            with col4:
                st.metric(
                    "Batch Size",
                    f"{self.training_stats.get('batch_size', 0):,}",
                )

    def show_model_evaluation_info(self) -> None:
        """
        Display evaluation metrics of the loaded NER model, if available.
        """
        if self.evaluation_stats:
            col1, col2, col3 = st.columns(3)
            overall = self.evaluation_stats.get("overall", {})

            with col1:
                st.metric(
                    "Overall Precision", f"{overall.get('precision', 0):.2f}"
                )
            with col2:
                st.metric("Overall Recall", f"{overall.get('recall', 0):.2f}")
            with col3:
                st.metric("Overall F1 Score", f"{overall.get('f1_score', 0):.2f}")

            st.json(self.evaluation_stats.get("by_label", {}))

    def test_single_name(self) -> None:
        """
        Render UI controls for testing the NER model on a single name input.
        """
        name_input = st.text_input(
            "Name:",
            placeholder="e.g., Jean Baptiste Mukendi, Marie Kabamba Tshiala, Joseph Kasongo",
            help="Enter a full name or multiple names separated by spaces",
        )

        if name_input.strip() and st.button("Analyze Name", type="primary"):
            self.analyze_and_display(name_input)

    def test_multiple_names(self) -> None:
        """
        Render UI controls for testing the NER model on multiple name inputs.
        """
        names_input = st.text_area(
            "Names:",
            placeholder="Jean Baptiste Mukendi\nMarie Kabamba Tshiala\nJoseph Kasongo\nGrace Mbuyi Kalala",
            height=150,
            help="Enter each name on a new line",
        )

        if names_input.strip() and st.button("Analyze All Names", type="primary"):
            names = [name.strip() for name in names_input.split("\n") if name.strip()]
            for i, name in enumerate(names):
                st.markdown(f"**Name {i + 1}: {name}**")
                self.analyze_and_display(name)
                if i < len(names) - 1:
                    st.markdown("---")

    def analyze_and_display(self, text: str) -> None:
        """
        Run NER prediction on input text and display results.

        Args:
            text: Input string containing one or more names.
        """
        if self.ner_model is None:
            st.error("NER model is not loaded.")
            return

        try:
            result = self.ner_model.predict(text)
            st.subheader("Analysis Results")
            entities = result.get("entities", [])

            if entities:
                self.show_visual_entities(text, entities)

                native_count = sum(1 for e in entities if e.get("label") == "NATIVE")
                surname_count = sum(1 for e in entities if e.get("label") == "SURNAME")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Entities", len(entities))
                with col2:
                    st.metric("Native Names", native_count)
                with col3:
                    st.metric("Surnames", surname_count)
            else:
                st.warning("No entities detected in the input text.")
                st.info(
                    "Try using traditional Congolese names or ensure the spelling is correct."
                )

        except Exception as e:
            st.error(f"Error analyzing text: {e}")

    @classmethod
    def show_visual_entities(cls, text: str, entities: list[dict[str, Any]]) -> None:
        """
        Render a visual NER representation using spaCy displacy.

        Args:
            text: Original input text.
            entities: List of extracted entity dictionaries.
        """
        try:
            ents = [
                {
                    "start": entity["start"],
                    "end": entity["end"],
                    "label": entity["label"],
                }
                for entity in entities
            ]

            doc_data = {"text": text, "ents": ents, "title": None}

            colors = {
                "NATIVE": "#74C0FC",
                "SURNAME": "#69DB7C",
            }

            options = {"colors": colors, "distance": 90}

            html = displacy.render(
                doc_data, style="ent", manual=True, options=options
            )
            st.markdown(html, unsafe_allow_html=True)

        except Exception as e:
            st.warning(f"Could not generate visual representation: {e}")
