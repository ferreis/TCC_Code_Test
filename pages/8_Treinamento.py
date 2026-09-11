from __future__ import annotations

import streamlit as st

from contextual_help import help_for

from tcc_discovery.training_ui import render_training_page

st.set_page_config(page_title="TCC - Treinamento YOLO26", layout="wide")
st.title("TCC — Treinamento", help=help_for('TCC — Treinamento'))
render_training_page()
