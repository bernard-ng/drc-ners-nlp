import sys
from pathlib import Path

import streamlit as st

# Add parent directory to Python path to access core modules
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from web.interfaces.experiments import Experiments

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
