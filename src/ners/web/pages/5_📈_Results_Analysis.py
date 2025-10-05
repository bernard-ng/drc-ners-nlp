import streamlit as st

from ners.web.interfaces.results_analysis import ResultsAnalysis

st.set_page_config(page_title="Results & Analysis", page_icon="📈", layout="wide")

if "config" in st.session_state:
    results_analysis = ResultsAnalysis(
        st.session_state.config,
        st.session_state.experiment_tracker,
        st.session_state.experiment_runner,
    )
    results_analysis.index()
else:
    st.error("Please run the main app first to initialize the configuration.")
    st.markdown("Go back to the [main page](/) to start the application.")
