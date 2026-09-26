# Model card — CoffeeGuard EfficientNetV2-B0 v1

Bundle: `artifacts/models/coffeeguard-effv2b0-v1/` (ONNX + `bundle.json` with every threshold; SHA-256 of each file in `bundle.json`). Decision record: [`decisions/001-deployment-model.md`](decisions/001-deployment-model.md). All numbers trace to the files named in [`PROGRESS.md`](PROGRESS.md).

## Intended use

- **Task:** classify a photo of a single Arabica coffee leaf as **Healthy, Cercospora (brown eye spot), Leaf Rust or Phoma**, with a calibrated confidence and a heat map of the evidence.
- **Users:** extension workers and farmers in Ethiopia as a *first opinion* that says when it is unsure; not a replacement for an agronomist.
- **Out of scope:** other crops, other coffee diseases (e.g. berry disease, wilt), several leaves or whole plants per photo, severity grading, treatment advice.

## How a photo is handled

1. **Quality gate** — rejects photos that are too dark (mean brightness < 41), over-exposed (> 203), flat (RMS contrast < 12.5), blurred (Laplacian variance < 1.6 at 384 px) or have almost no leaf colour (< 2%). The dark/flat/blur limits sit where the model's accuracy on degraded validation photos falls below 90%. Answer: *retake*, with the reason.
2. **OOD gate** — KNN distance (k = 10, cosine) from the photo's embedding to 1,764 training embeddings; rejects above τ = 0.554 (99.5th percentile of validation photos). Answer: *this doesn't look like a coffee leaf*.
3. **Prediction** — EfficientNetV2-B0 (`tf_efficientnetv2_b0.in1k`, 224 px), temperature-scaled (T = 0.52). **Accepted** if the 98% conformal set has one class and confidence ≥ 0.947; otherwise **uncertain** with the top candidates.
4. **Explanation** — CAM of the predicted class from the same forward pass (identical to Grad-CAM for this head; verified by test).

## Training data

- [Ethiopian Coffee Leaf Disease dataset](https://www.kaggle.com/datasets/biniyamyoseph/ethiopian-coffee-leaf-disease) (CC0). Of 12,000 files only 3,491 are original photos (the rest are the author's generated copies, excluded); after removing byte-identical copies and 69 cross-label conflicts: **2,520 photos** (Healthy 715, Cercospora 505, Leaf Rust 837, Phoma 463).
- **Group-aware split** 70/15/15 (1,764 / 377 / 379): copies of the same leaf — exact, near-duplicate, rotated/mirrored or re-shot (DINOv2 similarity ≥ 0.97 within a class) — never cross splits. Data fingerprint `53320bd30430396d`.
- **Training:** linear probe (5 epochs) then fine-tuning of the last 3 stages (≤ 15 epochs, early stopping on val macro-F1), AdamW, label smoothing 0.1, EMA, quality-equalising augmentation (random down-scaling and JPEG), rotation with border fill, **background swap** (p = 0.5). One seed, Kaggle T4, 6 min. Run `20260926-042311-effnetv2_b0_bgswap-s0` (git `2bdbee7` + background-swap code committed in `f300ab4`).

## Performance (test split, 379 photos, opened once)

| Metric | Value |
|---|---|
| Macro-F1 [95% bootstrap CI] | **0.974** [0.956, 0.990] |
| Accuracy | 0.974 (10 errors) |
| Macro ROC-AUC | 0.996 |
| Per class F1 (precision / recall) | Healthy 0.986 (0.98 / 0.99) · Cercospora 0.967 (0.96 / 0.97) · Leaf Rust 0.964 (0.96 / 0.97) · Phoma 0.978 (1.00 / 0.96) |
| Main confusion | Leaf Rust → Cercospora (3); half of the errors of the pre-swap model were images the label audit flags as possibly mislabelled |
| Calibration (ECE, 15 bins) | 0.097 → **0.014** after temperature scaling |
| Conformal coverage (98% target) | 0.974 |
| Through the full decision | **90.8% answered, 99.4% of answers correct**; 7.4% *uncertain*, 1.8% rejected |
| Confidence ≥ 0.8 | 367 photos, 98.4% correct |
| CPU latency (Intel i5 laptop, 4 threads, batch 1) | model 15.5 ms p50 / 23 ms p95; full request incl. decoding a 2048 px phone photo 63 ms p50 / 72 ms p95 |

Validation macro-F1 0.987. Single training seed: differences between models of ± 0.005 on 379 photos are within noise (paired bootstrap in the decision record).

## Robustness and shortcut checks

- **Corruptions** (9 types × 5 severities, test): keeps 97.9% of its macro-F1 at severity ≤ 3 and 93.8% over all severities. Weakest at severity 5: Gaussian noise 0.68, occlusion 0.67, JPEG quality 5 0.72.
- **Heavy degradation pushes answers to *Healthy***, not to the classes that happened to be photographed at lower quality — evidence against a photo-quality shortcut, but it means **a diseased leaf in a very poor photo can be called healthy**. The quality gate catches strong darkening, low contrast and blur; it does **not** detect heavy noise or JPEG artefacts.
- **Heat maps are faithful** (hiding the hottest 5% of patches drops confidence 0.97 → 0.72; random 5%: 0.88) and focus on the leaf (66% of CAM mass on a leaf covering 25% of the image).
- **Photo setup:** each class was photographed in its own setup (blue paper for Cercospora/Leaf Rust, white paper for Healthy/Phoma, field photos for some Healthy/Leaf Rust). Background-swap training made the model rely on the leaf (leaf-only accuracy 0.963; confidence on leaf-less images 0.54), but **blue-paper backgrounds alone still lean towards Cercospora/Leaf Rust**.

## Out-of-distribution detection (sources never used for tuning)

AUROC 0.993 against other plant leaves (bean), 1.000 against non-leaf images (indoor scenes, text/screenshots, blank/noise frames). End to end, 175 of 182 such test images are rejected; 3 bean leaves were accepted.

## Limitations and risks

- **Small, setup-confounded dataset:** 2,520 photos from few sources; classes differ in camera, resolution and background. Real field photos of every class are the most valuable next data.
- **Label noise:** 19 images (0.75%) flagged by an automatic audit, mostly Cercospora ↔ Leaf Rust; not removed without expert review.
- **Only four classes:** any other disease, pest damage, nutrient deficiency or mechanical damage will be forced into one of them unless the OOD gate rejects it.
- **Single seed, single dataset:** reported uncertainty covers test-set sampling, not training randomness or a new region/season/camera.
- **Not a diagnosis:** treat *Healthy* answers on poor photos with care (see robustness), and confirm any disease finding before treatment.
