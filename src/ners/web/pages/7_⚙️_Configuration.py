import streamlit as st

from ners.web.interfaces.configuration import Configuration

st.set_page_config(page_title="Configuration", page_icon="⚙️", layout="wide")

if "config" in st.session_state:
    configuration = Configuration(st.session_state.config)
    configuration.index()
else:
    st.error("Please run the main app first to initialize the configuration.")
    st.markdown("Go back to the [main page](/) to start the application.")
