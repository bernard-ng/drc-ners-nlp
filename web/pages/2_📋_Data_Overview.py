import sys
from pathlib import Path
import streamlit as st

# Add parent directory to Python path to access core modules
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from web.interfaces.data_overview import DataOverview

st.set_page_config(page_title="Data Overview", page_icon="📋", layout="wide")

if "config" in st.session_state:
    data_overview = DataOverview(st.session_state.config)
    data_overview.index()
else:
    st.error("Please run the main app first to initialize the configuration.")
    st.markdown("Go back to the [main page](/) to start the application.")
