# Demo script (about 5 minutes)

**Start:** `docker compose up --build` (or, with uv: `uv run uvicorn app.main:app --app-dir apps/api` and `uv run streamlit run apps/web/streamlit_app.py`). Open http://localhost:8501.

| Time | Page | Show | Say |
|---|---|---|---|
| 0:00 | **Home** | Hero, the four numbers, the six "variation" photos | The problem: diseases are hard to recognise from photos that vary in light, blur, background, framing and severity. The system answers — and says when it can't. |
| 0:45 | **Diagnose** → sample *Leaf Rust* | Green card ">99% confident", heat map on the pustules, probability bars | A confident answer, calibrated; the heat map shows *why*. Slide the heat map to 0 to compare with the photo. |
| 1:30 | Sample *Hard case* | Amber card "Cercospora or Leaf Rust — 74% / 25%" + advice | When unsure it names the candidates instead of guessing. |
| 2:00 | Sample *Too dark* | Red card "too dark, washed out" + retake tips | Poor photos get a retake request — the model is not even run. |
| 2:20 | Sample *Not coffee* (bean leaf) | "This doesn't look like a coffee leaf", no chart | Out-of-distribution photos are turned away (AUROC 0.993 vs. other plant leaves). |
| 2:45 | **Model Comparison** | Interval chart and leaf-only accuracy panel | Five models, same recipe; three tie on accuracy; the deployed one relies most on the leaf. |
| 3:30 | **EDA** | Cleaning funnel, confound charts | 12,000 files → 2,520 real photos; each disease was photographed in its own setup — which is why we test for shortcuts. |
| 4:10 | **Model Analysis** | Feature importance → Cross-validation tab | Heat maps are faithful (deletion test); CV 0.985 ± 0.007; honest limits. |
| 4:45 | http://localhost:8000/docs | `/analyze` "Try it out" | The same decisions are available as a documented API. |

Shareable links: `http://localhost:8501/diagnose?sample=leaf_rust` (`uncertain`, `too_dark`, `not_coffee_bean_leaf`, `healthy`, `cercospora`, `phoma`).
