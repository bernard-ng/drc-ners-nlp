import streamlit as st

from ners.web.interfaces.data_processing import DataProcessing

st.set_page_config(page_title="Data Processing", page_icon="⚙️", layout="wide")

if "config" in st.session_state:
    data_processing = DataProcessing(
        st.session_state.config, st.session_state.pipeline_monitor
    )
    data_processing.index()
else:
    st.error("Please run the main app first to initialize the configuration.")
    st.markdown("Go back to the [main page](/) to start the application.")
