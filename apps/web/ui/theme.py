"""Design system for the CoffeeGuard app: one stylesheet + small HTML building blocks.

Colours: coffee-leaf green brand, warm neutrals; status colours always come with an icon
and words. Every block is plain HTML rendered through ``st.markdown`` so pages stay simple.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path

import streamlit as st

CSS = """
<style>
:root {
  --cg-green: #2f7d4f; --cg-green-deep: #1d4a30; --cg-green-soft: #e8f3ec;
  --cg-coffee: #3b2a20; --cg-ink: #0b0b0b; --cg-muted: #52514e; --cg-line: #e4e3df;
  --cg-card: #ffffff; --cg-soft: #f4f3ef;
  --cg-ok: #0b6b2b; --cg-ok-bg: #e6f4ea; --cg-warn: #7a5200; --cg-warn-bg: #fff4d6;
  --cg-bad: #a11d1d; --cg-bad-bg: #fde7e7;
}
.block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 4rem; }
h1, h2, h3 { letter-spacing: -0.02em; }
.cg-kicker { text-transform: uppercase; letter-spacing: .14em; font-size: .72rem;
  font-weight: 700; color: var(--cg-green); margin-bottom: .35rem; }
.cg-lead { font-size: 1.08rem; color: var(--cg-muted); line-height: 1.6; max-width: 62ch; }
.cg-section { margin: 2.6rem 0 1rem; }
.cg-section h2 { font-size: 1.6rem; margin: 0 0 .3rem; }
.cg-section p { color: var(--cg-muted); margin: 0; max-width: 70ch; }

/* hero */
.cg-hero { display: grid; grid-template-columns: 1.15fr 1fr; gap: 36px; align-items: center;
  padding: 44px 44px; border-radius: 28px; color: #fff;
  background: radial-gradient(1200px 400px at 0% 0%, rgba(255,255,255,.10), transparent 60%),
              linear-gradient(135deg, #16331f 0%, #1d4a30 45%, #2f7d4f 100%); }
.cg-hero .cg-kicker { color: #bfe3cc; }
.cg-hero h1 { color: #fff; font-size: 2.9rem; line-height: 1.05; margin: 0 0 16px; }
.cg-hero p { color: #dcefe3; font-size: 1.08rem; line-height: 1.6; margin: 0 0 24px; }
.cg-cta { display: inline-block; background: #fff; color: var(--cg-green-deep) !important;
  font-weight: 700; padding: 12px 22px; border-radius: 999px; text-decoration: none !important; }
.cg-cta:hover { background: #e8f3ec; }
.cg-cta-ghost { display: inline-block; margin-left: 10px; color: #fff !important;
  padding: 12px 18px; border-radius: 999px; border: 1px solid rgba(255,255,255,.45);
  text-decoration: none !important; }
.cg-collage { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.cg-collage figure { margin: 0; position: relative; border-radius: 16px; overflow: hidden;
  aspect-ratio: 1 / 1; box-shadow: 0 8px 24px rgba(0,0,0,.25); }
.cg-collage img { width: 100%; height: 100%; object-fit: cover; display: block; }
.cg-collage figcaption { position: absolute; left: 10px; bottom: 10px; font-size: .74rem;
  font-weight: 600; background: rgba(12,20,14,.72); color: #fff; padding: 4px 10px;
  border-radius: 999px; }

/* cards and stats */
.cg-card { background: var(--cg-card); border: 1px solid var(--cg-line); border-radius: 18px;
  padding: 20px 22px; height: 100%; }
.cg-card h4 { margin: 0 0 6px; font-size: 1.02rem; }
.cg-card p { margin: 0; color: var(--cg-muted); font-size: .93rem; line-height: 1.55; }
.cg-card img { width: 100%; aspect-ratio: 4 / 3; object-fit: cover; border-radius: 12px;
  margin-bottom: 12px; display: block; }
.cg-stat { background: var(--cg-card); border: 1px solid var(--cg-line); border-radius: 18px;
  padding: 18px 20px; height: 100%; }
.cg-stat .v { font-size: 2rem; font-weight: 750; color: var(--cg-ink); line-height: 1.1; }
.cg-stat .l { color: var(--cg-muted); font-size: .88rem; margin-top: 6px; line-height: 1.4; }
.cg-step { display: flex; gap: 14px; align-items: flex-start; }
.cg-step .n { flex: none; width: 34px; height: 34px; border-radius: 50%; display: grid;
  place-items: center; background: var(--cg-green-soft); color: var(--cg-green-deep);
  font-weight: 750; }
.cg-note { border-left: 4px solid var(--cg-green); background: var(--cg-soft);
  padding: 14px 18px; border-radius: 0 12px 12px 0; color: var(--cg-ink); line-height: 1.55; }
.cg-note.warn { border-left-color: #c98500; }

.cg-num { width: 34px; height: 34px; border-radius: 50%; display: grid; place-items: center;
  background: var(--cg-green-soft); color: var(--cg-green-deep); font-weight: 750;
  margin-bottom: 12px; }
.cg-grid3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
.cg-grid3 figure { margin: 0; position: relative; border-radius: 14px; overflow: hidden;
  aspect-ratio: 1 / 1; }
.cg-grid3 img { width: 100%; height: 100%; object-fit: cover; display: block; }
.cg-grid3 figcaption { position: absolute; left: 8px; bottom: 8px; font-size: .72rem;
  font-weight: 600; background: rgba(12,20,14,.72); color: #fff; padding: 3px 9px;
  border-radius: 999px; }
/* sample gallery buttons: compact labels */
.st-key-samples button p { font-size: .8rem; }
.st-key-samples [data-testid="stImage"] img { border-radius: 10px; }

/* diagnosis result */
.cg-result { border-radius: 20px; padding: 22px 24px; border: 1px solid var(--cg-line);
  background: var(--cg-card); }
.cg-pill { display: inline-flex; gap: 6px; align-items: center; border-radius: 999px;
  padding: 4px 12px; font-weight: 700; font-size: .78rem; letter-spacing: .02em; }
.cg-pill.ok { background: var(--cg-ok-bg); color: var(--cg-ok); }
.cg-pill.warn { background: var(--cg-warn-bg); color: var(--cg-warn); }
.cg-pill.bad { background: var(--cg-bad-bg); color: var(--cg-bad); }
.cg-result .title { font-size: 2rem; font-weight: 750; margin: 12px 0 2px; line-height: 1.15; }
.cg-result .sub { color: var(--cg-muted); margin-bottom: 14px; }
.cg-meter { height: 10px; border-radius: 999px; background: var(--cg-soft); overflow: hidden; }
.cg-meter > div { height: 100%; border-radius: 999px; background: var(--cg-green); }
.cg-meter.warn > div { background: #c98500; }
.cg-advice { margin: 14px 0 0; padding-left: 1.1rem; color: var(--cg-ink); }
.cg-advice li { margin: 4px 0; }

/* team */
.cg-avatar { width: 64px; height: 64px; border-radius: 50%; display: grid; place-items: center;
  font-weight: 750; font-size: 1.3rem; color: #fff; margin-bottom: 12px;
  background: linear-gradient(135deg, #1d4a30, #2f7d4f); }
.cg-role { color: var(--cg-green); font-weight: 700; font-size: .85rem; margin-bottom: 8px; }

@media (max-width: 860px) {
  .cg-hero { grid-template-columns: 1fr; padding: 28px; }
  .cg-hero h1 { font-size: 2.1rem; }
}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(text: object) -> str:
    return html.escape(str(text))


def img_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def section(title: str, text: str = "", kicker: str = "") -> None:
    k = f'<div class="cg-kicker">{esc(kicker)}</div>' if kicker else ""
    p = f"<p>{text}</p>" if text else ""
    st.markdown(
        f'<div class="cg-section">{k}<h2>{esc(title)}</h2>{p}</div>', unsafe_allow_html=True
    )


def stat(value: str, label: str) -> str:
    return (
        f'<div class="cg-stat"><div class="v">{esc(value)}</div><div class="l">{label}</div></div>'
    )


def stats_row(items: list[tuple[str, str]]) -> None:
    for col, (v, label) in zip(st.columns(len(items)), items, strict=True):
        col.markdown(stat(v, label), unsafe_allow_html=True)


def card(title: str, body: str, image: Path | None = None) -> str:
    img = f'<img src="{img_uri(image)}" alt="{esc(title)}">' if image else ""
    return f'<div class="cg-card">{img}<h4>{esc(title)}</h4><p>{body}</p></div>'


def note(text: str, warn: bool = False) -> None:
    st.markdown(
        f'<div class="cg-note{" warn" if warn else ""}">{text}</div>', unsafe_allow_html=True
    )
