"""CoffeeGuard AI - Streamlit app (diagnosis via the FastAPI service; reports from artifacts/).

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

st.set_page_config(page_title="CoffeeGuard AI", page_icon="🍃", layout="wide")

VIEWS = HERE / "views"
nav = st.navigation(
    {
        "": [
            st.Page(str(VIEWS / "home.py"), title="Home", icon=":material/home:", default=True),
            st.Page(
                str(VIEWS / "diagnose.py"),
                title="Diagnose",
                icon=":material/eco:",
                url_path="diagnose",
            ),
        ],
        "Results": [
            st.Page(
                str(VIEWS / "comparison.py"),
                title="Model Comparison",
                icon=":material/leaderboard:",
                url_path="model_comparison",
            ),
            st.Page(str(VIEWS / "eda.py"), title="EDA", icon=":material/dataset:", url_path="eda"),
            st.Page(
                str(VIEWS / "analysis.py"),
                title="Model Analysis",
                icon=":material/insights:",
                url_path="model_analysis",
            ),
        ],
        "About": [
            st.Page(
                str(VIEWS / "team.py"),
                title="About Team",
                icon=":material/groups:",
                url_path="team",
            ),
        ],
    }
)
nav.run()
