import streamlit as st

from ners.web.interfaces.experiments import Experiments

st.set_page_config(page_title="Experiments", page_icon="🧪", layout="wide")

if "config" in st.session_state:
    experiments = Experiments(
        st.session_state.config,
        st.session_state.experiment_tracker,
        st.session_state.experiment_runner,
    )
    experiments.index()
else:
    st.error("Please run the main app first to initialize the configuration.")
    st.markdown("Go back to the [main page](/) to start the application.")
