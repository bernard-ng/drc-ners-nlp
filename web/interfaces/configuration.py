import streamlit as st


class Configuration:
    def __init__(self, config):
        self.config = config

    def index(self):
        st.title("Configuration")
        st.json(self.config.model_dump())
