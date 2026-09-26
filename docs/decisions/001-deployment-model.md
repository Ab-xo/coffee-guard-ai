# ADR 001 — Deployment model

**Status:** accepted (Phase 7) · **Decision:** deploy **EfficientNetV2-B0 trained with background swap** as bundle `artifacts/models/coffeeguard-effv2b0-v1` (from `cand-effv2b0-bgswap`).

## Context

The specification names EfficientNetV2-B0 as the primary model and asks for a comparison with lighter alternatives. All candidates were trained with the same recipe (5-epoch linear probe → ≤ 15 epochs fine-tuning the last 3 stages, one seed, Kaggle T4) and evaluated identically through their exported ONNX bundles — the exact artifact the API serves. The service runs on CPU (target: ≤ 100 ms per image on the development laptop, an Intel i5 8th gen with 4 ONNX Runtime threads).

## Options (test split, 379 photos; `artifacts/metrics/model_comparison.{json,md}`)

| Model | Test macro-F1 [95% CI] | Errors | ECE | Rel. robustness (sev 1–3) | Leaf-only acc | OOD AUROC near / far | Answered (accuracy) | Params (M) | ONNX MB | CPU p50 / p95 ms |
|---|---|---:|---:|---:|---:|---|---|---:|---:|---|
| **EfficientNetV2-B0 + background swap** | 0.974 [0.956, 0.990] | 10 | 0.014 | 0.979 | **0.963** | 0.993 / 1.000 | 90.8% (99.4%) | 5.83 | 23.5 | 15.5 / 23.0 |
| EfficientNetV2-B0 | 0.979 [0.963, 0.993] | 8 | 0.012 | 0.975 | 0.926 | 0.990 / 1.000 | 94.5% (99.4%) | 5.83 | 23.5 | 22.3 / 39.3 |
| EfficientNet-B0 | 0.974 [0.957, 0.988] | 10 | 0.016 | 0.980 | 0.903 | 0.990 / 1.000 | 92.3% (98.6%) | 3.99 | 16.0 | 17.0 / 29.6 |
| MobileNetV3-Small | 0.960 [0.938, 0.979] | 15 | 0.023 | 0.991 | 0.956 | 0.997 / 0.992 | 87.9% (99.1%) | 1.52 | 6.1 | 2.7 / 3.6 |
| MobileNetV3-Large | 0.958 [0.935, 0.977] | 16 | 0.018 | 0.983 | 0.919 | 0.995 / 1.000 | 89.4% (99.4%) | 4.19 | 16.8 | 6.5 / 7.6 |

"Answered" = share of genuine test photos the full decision (quality gate → OOD gate → confidence) accepts; "accuracy" = accuracy of those answers. Latency = ONNX forward pass, batch 1, 200 runs after warm-up; the two EfficientNetV2-B0 rows are the *same* architecture, so their 15.5 vs 22.3 ms shows the laptop's measurement noise (~±30%). A full request (decode a 2048 px phone JPEG + preprocess + model) takes **63 ms p50 / 72 ms p95** for the chosen model.

**Paired bootstrap on test** (chosen model minus each alternative, 2,000 resamples): EfficientNetV2-B0 −0.005 [−0.017, +0.006] · EfficientNet-B0 −0.000 [−0.017, +0.016] · MobileNetV3-Small +0.014 [−0.009, +0.036] · MobileNetV3-Large +0.016 [−0.002, +0.036] (P(not better) = 0.04).

## Decision and reasons

1. **Accuracy is a tie among the three EfficientNet variants** (all paired CIs include 0); the MobileNets are 1.4–1.6 points lower, MobileNetV3-Large probably really so.
2. **Leaf reliance breaks the tie.** With the background removed, the background-swap model keeps 0.963 accuracy vs. 0.926 / 0.903 for the others, has the highest CAM leaf focus (0.66) and is least confident on leaf-less images (0.54 vs. 0.83). In the field no photo will come on the blue or white paper of the training set, so the model that depends least on the photo setup is the safer choice — its small test deficit is expected, because the test split shares the setup bias.
3. **Speed and size are not limiting:** 15–23 ms model latency and 63 ms end to end are well inside the 100 ms target; 23.5 MB is fine for a server image.
4. **Robustness and OOD detection** are equivalent across the EfficientNets (0.975–0.980; AUROC ≥ 0.990 / 1.000).
5. It is the specification's primary architecture.

## Consequences

- The deployed model is slightly more cautious: it answers 90.8% of genuine test photos directly (vs. 94.5% without background swap) at the same 99.4% accuracy; the rest get *uncertain* with the top candidates or a retake request.
- The bundle (28 MB incl. the KNN bank) is committed so a fresh clone and the Docker image can serve it; candidate bundles keep only `bundle.json` in git.
- **MobileNetV3-Small** (6 MB, 2.7 ms, most robust in relative terms) is the candidate for a future on-device/offline app; it should be retrained with background swap first.
- INT8 quantisation was not needed for the latency target and was not done (stretch item).
