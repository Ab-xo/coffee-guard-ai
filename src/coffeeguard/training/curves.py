"""Training-curve figure for one run, from its ``history.jsonl``."""

from __future__ import annotations

import json
from pathlib import Path

from coffeeguard.utils.plotting import SERIES, TEXT_MUTED, apply_style, save


def read_history(run_dir: Path) -> list[dict]:
    lines = (run_dir / "history.jsonl").read_text("utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def plot_curves(run_dir: Path, out: Path | None = None) -> Path:
    """Loss (train vs. val) and val macro-F1 (raw vs. EMA weights) per epoch.

    Epochs are numbered continuously across stages; a dotted line marks each stage start.
    """
    import matplotlib.pyplot as plt

    apply_style()
    rows = read_history(run_dir)
    x = list(range(1, len(rows) + 1))
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(10, 3.6))

    ax_loss.plot(x, [r["train_loss"] for r in rows], color=SERIES[0], label="train")
    ax_loss.plot(x, [r["val_loss"] for r in rows], color=SERIES[1], label="val (raw)")
    ax_loss.set_title("Loss")
    ax_f1.plot(x, [r["val_macro_f1"] for r in rows], color=SERIES[0], label="raw weights")
    if "ema_val_macro_f1" in rows[0]:
        ema = [r["ema_val_macro_f1"] for r in rows]
        ax_f1.plot(x, ema, color=SERIES[1], label="EMA weights")
    ax_f1.set_title("Validation macro-F1")

    # stage boundaries + labels
    starts = [i + 1 for i, r in enumerate(rows) if i == 0 or r["stage"] != rows[i - 1]["stage"]]
    for ax in (ax_loss, ax_f1):
        for s in starts[1:]:
            ax.axvline(s - 0.5, color=TEXT_MUTED, linestyle=":", linewidth=1)
        ax.set_xlabel("epoch (all stages)")
        ax.legend()
    for s in starts:
        ax_f1.annotate(
            rows[s - 1]["stage"],
            (s, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(2, -12),
            textcoords="offset points",
            fontsize=8,
            color=TEXT_MUTED,
        )
    best = max(
        range(len(rows)),
        key=lambda i: max(rows[i]["val_macro_f1"], rows[i].get("ema_val_macro_f1", 0)),
    )
    best_f1 = max(rows[best]["val_macro_f1"], rows[best].get("ema_val_macro_f1", 0))
    ax_f1.scatter([best + 1], [best_f1], color=SERIES[2], zorder=3, s=25)
    ax_f1.annotate(
        f"best {best_f1:.3f}", (best + 1, best_f1), xytext=(4, -12), textcoords="offset points"
    )
    fig.suptitle(run_dir.name, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return save(fig, out or run_dir / "figures" / "training_curves.png")
