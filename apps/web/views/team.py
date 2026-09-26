"""About the team (people come from apps/web/team.json — edit that file, not this one)."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from ui import theme

TEAM = Path(__file__).resolve().parents[1] / "team.json"
theme.inject()
team = json.loads(TEAM.read_text("utf-8"))

st.markdown(
    f'<div class="cg-kicker">About</div><h1 style="margin-top:0">{theme.esc(team["team_name"])}'
    f'</h1><p class="cg-lead">{theme.esc(team["intro"])}</p>',
    unsafe_allow_html=True,
)


def initials(name: str, role: str) -> str:
    words = (name or role).replace("&", " ").split()
    return "".join(w[0] for w in words[:2]).upper()


members = team["members"]
if any(not m["name"] for m in members):
    st.caption("Names are added in `apps/web/team.json`.")
for row in range(0, len(members), 3):
    for col, m in zip(st.columns(3, gap="medium"), members[row : row + 3], strict=False):
        name = m["name"] or "Team member"
        gh = (
            f'<p style="margin-top:8px"><a href="https://github.com/{theme.esc(m["github"])}" '
            f'target="_blank">@{theme.esc(m["github"])}</a></p>'
            if m.get("github")
            else ""
        )
        avatar = theme.esc(initials(m["name"], m["role"]))
        col.markdown(
            f'<div class="cg-card"><div class="cg-avatar">{avatar}'
            f'</div><h4>{theme.esc(name)}</h4><div class="cg-role">{theme.esc(m["role"])}</div>'
            f"<p>{theme.esc(m['focus'])}</p>{gh}</div>",
            unsafe_allow_html=True,
        )
        col.write("")

theme.section("How we worked", kicker="Process")
st.markdown(
    "- **Reproducible pipeline:** every step is one command of the `coffeeguard` CLI; each "
    "result traces to a run with its configuration, git commit and data fingerprint.\n"
    "- **Honest evaluation:** duplicate-aware splits, a test set opened once, confidence "
    "intervals, and limitations stated plainly.\n"
    "- **Engineering:** typed configuration, 90+ automated tests, lint and CI, a Docker-ready "
    "API and web app.\n"
    "- **Everything documented:** `docs/PROGRESS.md` logs each step, decision and number."
)

theme.section("Acknowledgements", kicker="Thanks")
st.markdown(
    "- **Data:** [Ethiopian Coffee Leaf Disease dataset](https://www.kaggle.com/datasets/"
    "biniyamyoseph/ethiopian-coffee-leaf-disease) by Biniyam Yoseph (CC0).\n"
    "- **Out-of-distribution test images:** tomato, banana and bean leaf datasets, animal and "
    "indoor-scene photos (sources and licences in `docs/PROGRESS.md`).\n"
    "- **Compute:** free GPU time on Kaggle.\n"
    "- **Open source:** PyTorch, timm, ONNX Runtime, scikit-learn, FastAPI, Streamlit, Altair."
)
