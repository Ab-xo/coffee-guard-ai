"""CoffeeGuard AI - Streamlit UI (talks to the FastAPI service; never loads the model).

Run from the repo root (so .streamlit/config.toml is picked up):
    uv run streamlit run apps/web/streamlit_app.py
The API address comes from API_URL (default http://localhost:8000).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # the `ui` package lives next to this file
    sys.path.insert(0, str(HERE))

st.set_page_config(page_title="CoffeeGuard", page_icon="🍃", layout="centered")

nav = st.navigation(
    [
        st.Page(
            str(HERE / "views" / "diagnose.py"),
            title="Diagnose",
            icon=":material/eco:",
            default=True,
        ),
        st.Page(str(HERE / "views" / "model.py"), title="The model", icon=":material/insights:"),
    ]
)
nav.run()
