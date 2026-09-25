# Training recipes

All recipes share the LP-FT schedule (linear probe `head` stage, then a fine-tuning
stage) so model comparisons differ only in the backbone. Run with:

    uv run coffeeguard train -c configs/train/<recipe>.yaml [--seed N] [--set key=value]

| Recipe | Role |
|---|---|
| `mobilenetv3_small.yaml` | Baseline (Phase 2) |
| `effnetv2_b0.yaml` | Main model, variant B: full fine-tune + layer-wise LR decay |
| `effnetv2_b0_partial.yaml` | Main model, variant A (spec): unfreeze last 3 stages only |
| `mobilenetv3_large.yaml` | Comparison candidate |
| `efficientnet_b0.yaml` | Comparison candidate |
| `smoke.yaml` | CPU smoke test on tiny data (tests / CI) |
