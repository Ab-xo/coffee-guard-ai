# Training recipes

All recipes except `effnetv2_b0.yaml` share one LP-FT schedule, chosen by the Phase 3
ablation: a 5-epoch linear probe (`head`), then up to 15 epochs fine-tuning the last 3
stages at LR 1e-4 (early stopping on val macro-F1, patience 4). Model comparisons
therefore differ only in the backbone. Run with:

    uv run coffeeguard train -c configs/train/<recipe>.yaml [--seed N] [--set key=value]

| Recipe | Role |
|---|---|
| `mobilenetv3_small.yaml` | Baseline (Phase 2 run used the earlier all-layer schedule) |
| `effnetv2_b0.yaml` | Ablation variant B: full fine-tune + layer-wise LR decay (not chosen) |
| `effnetv2_b0_partial.yaml` | **Main model**, variant A (spec): unfreeze last 3 stages only |
| `mobilenetv3_large.yaml` | Comparison candidate |
| `efficientnet_b0.yaml` | Comparison candidate |
| `smoke.yaml` | CPU smoke test on tiny data (tests / CI) |
