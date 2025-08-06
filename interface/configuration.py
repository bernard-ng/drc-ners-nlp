import streamlit as st


class Configuration:
    """Handles configuration display and management"""

    def __init__(self, config):
        self.config = config

    def index(self):
        st.header("Current Configuration")
        st.json(self.config.model_dump())
