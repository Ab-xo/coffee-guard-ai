# CoffeeGuard AI — Technical report

*A coffee-leaf disease classifier that exposes its confidence, explanation, robustness and rejection behaviour.* Version 1.0. Every number below traces to a committed artifact (paths in brackets) and to [`PROGRESS.md`](PROGRESS.md), which logs each step.

## 1. Summary

Coffee leaf diseases can be difficult to identify quickly from photographs, especially when images vary in lighting, blur, background, framing and disease severity. CoffeeGuard classifies a photo of one Arabica leaf as **Healthy, Cercospora, Leaf Rust or Phoma** and, beyond a label, reports a calibrated confidence, a heat map of the evidence, and refuses to answer poor or unrelated photos.

| Headline (held-out test set, 379 photos, opened once) | Value |
|---|---|
| Macro-F1 [95% bootstrap CI] | **0.974** [0.956, 0.990] |
| 5-fold group cross-validation (validation macro-F1) | 0.985 ± 0.007 |
| Calibration error (ECE) after temperature scaling | 0.097 → **0.014** |
| Answered directly / accuracy of those answers | **90.8% / 99.4%** (7.4% *uncertain*, 1.8% rejected) |
| Out-of-distribution detection AUROC, other plant leaves / non-leaves | 0.993 / 1.000 (sources never used for tuning) |
| Score kept under moderate photo damage (9 corruption types) | 97.9% |
| Latency on a CPU server | 15 ms model; 63 ms for a full 2048 px phone photo |

The deployed model is EfficientNetV2-B0 fine-tuned on 1,764 photos with background-swap augmentation, served as an ONNX bundle by a FastAPI service with a Streamlit front end (`docker compose up`).

## 2. Data

Details: [`DATA_CARD.md`](DATA_CARD.md). The public dataset (12,000 files) turned out to contain 8,509 author-generated augmentations and a test set that overlapped its training set. After setting those aside and removing byte-identical duplicates (902) and cross-label conflicts (69), **2,520 unique photos** remain. Copies of the same leaf — exact, near-duplicate, rotated/mirrored or re-shot — were grouped (SHA-256, rotation-invariant pHash, DINOv2 similarity ≥ 0.97 within class) and split **group-aware and stratified** into 1,764 / 377 / 379 photos; no group crosses splits.

**Confound.** Classes were photographed in different setups: resolution (Healthy/Phoma 2048 px vs. Cercospora/Leaf Rust 1024 px), compression, sharpness and background (blue vs. white paper, field photos) all correlate with the label. This shaped the training recipe (§3) and the checks (§6).

**Label audit.** Cleanlab on out-of-fold DINOv2 probabilities flags 19 photos (0.75%), mostly Cercospora ↔ Leaf Rust; none were removed without expert review. A frozen-DINOv2 linear probe gives a strong reference: validation macro-F1 0.961.

## 3. Method

- **Backbone:** EfficientNetV2-B0 (`tf_efficientnetv2_b0.in1k`, 5.8 M parameters), 224 px input from a 384 px cache.
- **LP-FT** (linear probe then fine-tune; Kumar et al., 2022): 5 epochs training only a zero-initialised head (LR 1e-3), then ≤ 15 epochs fine-tuning the last 3 stages (LR 1e-4, AdamW, weight decay 0.05, cosine schedule with warm-up), label smoothing 0.1, EMA weights, gradient clipping, early stopping on validation macro-F1 (patience 4); the better of raw/EMA weights per epoch is kept.
- **Augmentation:** rotation with border-colour fill (black corners would be a train-only cue), random resized crop, flips, mild brightness/contrast/saturation jitter with **no hue shift** (lesion colour is diagnostic), light blur, **quality equalisation** (random down-scaling and JPEG re-compression, so photo quality stops predicting the class) and **background swap** (p = 0.5: the leaf is cut out with a colour mask and pasted onto another photo's background or a plain one).
- **Compute:** training on a Kaggle Tesla T4 launched from the CLI (`coffeeguard remote train`); ~4 minutes per run. A CPU laptop does everything else.
- **Bug found by the first real run:** timm initialises a new EfficientNet/MobileNet head with U(±1/√num_classes) — ±0.5 for 4 classes — which gave initial logits with std 6 and a linear probe stuck at chance. The head is now zero-initialised.

## 4. Experiments

| Experiment | Result (validation unless noted) | Decision |
|---|---|---|
| Baseline MobileNetV3-Small (CPU) | 0.991 (test 0.966) | reference |
| Fine-tune last 3 stages (A) vs. all layers with layer-wise LR decay (B), same seed | A 0.985, B 0.984 — same accuracy (6/377 errors each) | A: flatter validation curve; the last 3 stages hold 97% of the weights anyway |
| Shorter schedule (5 + ≤ 15 epochs) | identical best epoch and score | adopted for all runs |
| Comparison models, same recipe (test macro-F1) | EffNetV2-B0 0.979 · EfficientNet-B0 0.974 · MobileNetV3-S 0.960 · MobileNetV3-L 0.958 | three EfficientNets statistically tied (paired bootstrap) |
| Background swap on EffNetV2-B0 | leaf-only accuracy 0.926 → **0.963**, confidence on leaf-less images 0.83 → 0.54, test 0.979 → 0.974 (paired Δ not significant) | **deployed** — relies least on the photo setup |
| 5-fold group CV of the deployed recipe | 0.991 · 0.973 · 0.989 · 0.986 · 0.985 → **0.985 ± 0.007** | scores vary ±0.7 points with the data split; test 0.974 at the low end |

Model choice (decision record [`decisions/001-deployment-model.md`](decisions/001-deployment-model.md)): among statistically tied accuracies, the background-swap model wins on leaf reliance, robustness and OOD detection are equivalent, and speed/size are not limiting.

## 5. Evaluation of the deployed model

Evaluated through the exported ONNX bundle (exactly what is served) [`artifacts/eval/cand-effv2b0-bgswap/metrics.json`].

- **Test:** macro-F1 0.974 [0.956, 0.990], accuracy 0.974 (10 errors), macro ROC-AUC 0.996. Per-class F1: Healthy 0.986, Cercospora 0.967, Leaf Rust 0.964, Phoma 0.978. Main confusion: Leaf Rust → Cercospora (3).
- **Label noise:** for the pre-swap model, 4 of its 8 test errors — including the two most confident — were photos the label audit had flagged independently.
- **Calibration:** the fine-tuned models were *under*-confident (a side effect of label smoothing). Temperature scaling fitted on validation (T = 0.52) reduces ECE from 0.097 to 0.014.
- **Uncertainty:** split conformal prediction (LAC, Sadinle et al., 2019) at α = 0.02 (at the planned α = 0.10 every set had one class because the model is ~98% accurate); test coverage 0.974. The serving decision answers *uncertain* when the 98% set has more than one class or the calibrated confidence is below τ_conf = 0.947 (chosen on validation for 99% accuracy of direct answers).

## 6. Explanation, robustness and shortcuts

- **Explanation:** class activation maps from the ONNX feature map and classifier weights — equal to Grad-CAM for a pool→linear head, verified by a unit test (r > 0.99) — so serving needs no torch or backward pass. **Faithful:** hiding the hottest 5% of patches drops the predicted-class probability far faster than hiding random patches (deletion AUC 0.445 vs. 0.630; CAM order wins on 91% of test photos). **Focus:** 66% of the heat lands on the leaf, which covers 25% of the image.
- **Robustness** (9 deterministic corruptions × 5 severities, ImageNet-C style): relative robustness 0.979 at severities 1–3 and 0.938 over all; weakest at severity 5 are Gaussian noise (0.68), occlusion (0.67) and heavy JPEG (0.72). Heavy corruption pushes errors towards **Healthy** — not towards the classes that were photographed at low quality — evidence against a photo-quality shortcut, and a safety concern handled by the quality gate.
- **Shortcut test** (photos on paper, background or leaf blanked out): leaf-only accuracy 0.963; background-only accuracy 0.386 (majority class 0.32) at mean confidence 0.54 — but blue-paper backgrounds still lean towards Cercospora/Leaf Rust, a residual effect of the dataset's setup confound.
- **Severity (proxy — no labels):** colour-based lesion coverage per leaf; the mildest third of each disease is not handled worse (wrong top class 0–4%, one mild Phoma called Healthy) and is answered *uncertain* slightly more often.

## 7. Rejection gates

- **Quality gate:** thresholds for darkness, low contrast and blur sit where the model's validation accuracy under corruption drops below 90% (the training-photo percentiles first tried rejected 94–100% of mildly dark or blurred photos the model handles well); over-exposure and leaf-colour thresholds use training percentiles. 0.5% of genuine photos are rejected.
- **OOD gate:** of MSP, energy, Mahalanobis and KNN scores on the model's embedding, **KNN** (cosine distance to the 10th-nearest training embedding; Sun et al., 2022) was best on the calibration sources; τ is the highest validation percentile (99.5th) at which ≤ 1% of OOD calibration images pass. On unseen test sources: AUROC 0.993 (bean leaves) and 1.000 (indoor scenes, text, synthetic frames). Mahalanobis looked perfect on calibration data but collapsed on unseen far-OOD sources (0.47) — a reason to split OOD data by source.
- **End to end (test):** 344 genuine photos accepted (99.4% correct), 28 uncertain, 5 rejected as OOD, 2 as low quality; 175 of 182 OOD images rejected (3 bean leaves accepted).

## 8. System

- **Bundle** `artifacts/models/coffeeguard-effv2b0-v1`: `model.onnx` (outputs logits, embedding, feature map; torch↔ONNX max |Δlogit| < 1e-6), KNN bank, `bundle.json` with every threshold, headline metrics and SHA-256 per file. It reproduces the evaluation's test predictions exactly.
- **API** (FastAPI): `/health`, `/model-info`, `/predict` (decision + advice), `/analyze` (+ probabilities, quality report, OOD score, CAM overlay); magic-byte type checks, size and pixel limits, EXIF rotation, one error format, request IDs, JSON logs. 37 ms per request.
- **Web app** (Streamlit): Home, Diagnose, Model Comparison, EDA, Model Analysis, About Team; the report pages read committed results only.
- **Packaging:** `Dockerfile.api` (locked `api` extra only, no torch, non-root, health check, < 400 MB), `Dockerfile.web`, `docker-compose.yml`; CI builds both images and sends real photos through the running containers.
- **Engineering:** typed configs, one CLI for every step, 100+ tests (unit, API against a generated ONNX model, headless UI), ruff, CI; every run records config, git SHA and data fingerprint.

## 9. Limitations and future work

1. **No field validation.** All results come from one dataset with class-specific photo setups; robustness was tested with synthetic corruptions. The most valuable next step is a field test set: 30–50 farm phone photos per class labelled by an agronomist.
2. **Residual background cue:** blue-paper backgrounds still lean towards Cercospora/Leaf Rust.
3. **Early infection** is not represented; performance on barely visible symptoms is unknown.
4. **Photo conditions not covered:** colour casts, several leaves per photo, heavy noise or compression (not detected by the quality gate).
5. **Four classes only;** other diseases, pests or deficiencies are forced into one class unless the OOD gate rejects them.
6. **Single training seed** for the deployed model (cross-validation quantifies the data-split variance: ±0.7 points).
7. **Uploads:** the web UI shrinks photos after they reach the server; shrinking on the phone needs a custom front end. An offline on-phone app would favour MobileNetV3-Small (6 MB, 2.7 ms).

## 10. Reproducing

```bash
uv sync --all-extras
uv run coffeeguard data download && uv run coffeeguard data prepare && uv run coffeeguard data report
uv run coffeeguard embed audit
uv run coffeeguard remote upload-data
uv run coffeeguard remote train -c configs/train/effnetv2_b0_bgswap.yaml --seeds 0
uv run coffeeguard export --run runs/<run> --name <bundle>
uv run coffeeguard evaluate -b artifacts/models/<bundle> ...
uv run coffeeguard explain -b ... && uv run coffeeguard robustness -b ...
uv run coffeeguard ood collect && uv run coffeeguard ood fit -b artifacts/models/<bundle>
uv run coffeeguard data cv-folds && uv run coffeeguard remote train -c configs/train/effnetv2_b0_bgswap.yaml --cv-folds 5
docker compose up --build
```

## References

Kumar et al. 2022 (fine-tuning can distort pretrained features — LP-FT) · Tan & Le 2021 (EfficientNetV2) · Oquab et al. 2023 (DINOv2) · Northcutt et al. 2021 (confident learning / cleanlab) · Guo et al. 2017 (temperature scaling) · Sadinle et al. 2019 (least ambiguous set-valued classifiers) · Zhou et al. 2016 (CAM) · Selvaraju et al. 2017 (Grad-CAM) · Hendrycks & Dietterich 2019 (corruption robustness) · Liu et al. 2020 (energy OOD score) · Lee et al. 2018 (Mahalanobis OOD) · Sun et al. 2022 (deep nearest-neighbour OOD).
