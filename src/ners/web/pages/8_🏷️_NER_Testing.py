import streamlit as st

from ners.web.interfaces.ner_testing import NERTesting

st.set_page_config(page_title="NER Testing", page_icon="🏷️", layout="wide")

if "config" in st.session_state:
    ner_testing = NERTesting(st.session_state.config)
    ner_testing.index()
else:
    st.error("Please run the main app first to initialize the configuration.")
    st.markdown("Go back to the [main page](/) to start the application.")
