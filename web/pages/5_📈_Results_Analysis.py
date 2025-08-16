import sys
from pathlib import Path
import streamlit as st

# Add parent directory to Python path to access core modules
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from web.interfaces.results_analysis import ResultsAnalysis

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
