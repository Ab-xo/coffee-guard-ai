"""Decision matrix for the deployment model (Phase 7): gathers every earlier result.

Sources per bundle: ``artifacts/eval/<b>/metrics.json`` (Phase 4),
``artifacts/robustness/<b>/robustness.json`` (Phase 5), ``artifacts/ood/<b>/ood.json``
(Phase 6), ``artifacts/benchmark/<b>.json`` (latency/size) and the paired bootstrap
in ``artifacts/metrics/test_metrics.json``.
"""

from __future__ import annotations

from pathlib import Path

from coffeeguard.utils.io import read_json


def _get(path: Path) -> dict | None:
    return read_json(path) if path.exists() else None


def gather(bundles: list[str], root: Path) -> list[dict]:
    paired = (_get(root / "metrics" / "test_metrics.json") or {}).get(
        "paired_bootstrap_vs_main", {}
    )
    rows = []
    for b in bundles:
        ev = _get(root / "eval" / b / "metrics.json")
        rb = _get(root / "robustness" / b / "robustness.json")
        od = _get(root / "ood" / b / "ood.json")
        bm = _get(root / "benchmark" / f"{b}.json")
        row: dict = {"bundle": b}
        if ev:
            t = ev["test"]
            row |= {
                "model": ev["model"],
                "val_macro_f1": ev["val"]["macro_f1"],
                "test_macro_f1": t["macro_f1"],
                "test_ci": [
                    t["bootstrap_95ci"]["macro_f1"]["lo"],
                    t["bootstrap_95ci"]["macro_f1"]["hi"],
                ],
                "test_errors": t["errors"],
                "ece_after": t["calibration"]["ece_after"],
            }
        if b == paired.get("main"):
            row["paired_vs_main"] = "main"
        elif b in paired.get("comparisons", {}):
            c = paired["comparisons"][b]
            row["paired_vs_main"] = {"diff_main_minus_this": c["diff"], "ci": [c["lo"], c["hi"]]}
        if rb:
            row |= {
                "relative_robustness_sev1_3": rb["relative_robustness"]["severity_1_3"],
                "relative_robustness_sev1_5": rb["relative_robustness"]["severity_1_5"],
                "background_only_acc": rb["shortcut"]["background_only"]["accuracy"],
                "leaf_only_acc": rb["shortcut"]["leaf_only"]["accuracy"],
            }
        if od:
            s = od["test"][od["scorer"]]
            row |= {
                "ood_scorer": od["scorer"],
                "ood_auroc_near": s["near"]["auroc"],
                "ood_auroc_far": s["far"]["auroc"],
                "accepted_share": od["decisions"]["test"]["accepted_share"],
                "accepted_accuracy": od["decisions"]["test"]["accepted_accuracy"],
                "ood_test_accepted": od["decisions"]["ood_test"]["counts"].get("accepted:None", 0),
            }
        if bm:
            row |= {
                "onnx_mb": bm["onnx_mb"],
                "params_m": bm["params_m"],
                "latency_model_p50_ms": bm["model_only"]["p50_ms"],
                "latency_model_p95_ms": bm["model_only"]["p95_ms"],
                "latency_end_to_end_p50_ms": bm["end_to_end"]["p50_ms"],
            }
        rows.append(row)
    return rows


def markdown(rows: list[dict]) -> str:
    def f(v, fmt="{:.3f}"):
        return "—" if v is None else fmt.format(v)

    head = (
        "| Model | Test macro-F1 [95% CI] | Errors | ECE | Rel. robustness (sev 1–3) | "
        "Leaf-only acc | "
        "OOD AUROC near / far | Accepted (acc.) | Params (M) | ONNX MB | CPU p50 / p95 ms |\n"
        "|---|---|---:|---:|---:|---:|---|---|---:|---:|---|\n"
    )
    lines = []
    for r in rows:
        ci = r.get("test_ci") or [None, None]
        lines.append(
            f"| {r.get('model', r['bundle'])} (`{r['bundle']}`) | {f(r.get('test_macro_f1'))} "
            f"[{f(ci[0])}, {f(ci[1])}] | {r.get('test_errors', '—')} | {f(r.get('ece_after'))} | "
            f"{f(r.get('relative_robustness_sev1_3'))} | {f(r.get('leaf_only_acc'))} | "
            f"{f(r.get('ood_auroc_near'))} / {f(r.get('ood_auroc_far'))} | "
            f"{f(r.get('accepted_share'), '{:.1%}')} ({f(r.get('accepted_accuracy'), '{:.1%}')}) | "
            f"{f(r.get('params_m'), '{:.2f}')} | {f(r.get('onnx_mb'), '{:.1f}')} | "
            f"{f(r.get('latency_model_p50_ms'), '{:.1f}')} / "
            f"{f(r.get('latency_model_p95_ms'), '{:.1f}')} |"
        )
    return head + "\n".join(lines) + "\n"
