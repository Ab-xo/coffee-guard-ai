# 🍃 CoffeeGuard AI

> **Intelligent Coffee Leaf Disease Detection System for Ethiopian Coffee**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Project Status](https://img.shields.io/badge/status-Phase%209%20next-orange.svg)](docs/PROGRESS.md)

An AI-powered computer vision system for detecting and classifying coffee leaf diseases in Ethiopian coffee plants using deep learning, explainable AI, and production-ready deployment.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Development Status](#-development-status)
- [Documentation](#-documentation)
- [License](#-license)

---

## 🎯 Overview

**CoffeeGuard AI** addresses the critical challenge of rapidly identifying coffee leaf diseases in Ethiopian coffee farms. The system leverages state-of-the-art deep learning to classify leaf images into four categories:

- ✅ **Healthy**
- 🔴 **Cercospora**
- 🟠 **Leaf Rust**
- 🟡 **Phoma**

### Project Scope

- **Type**: Computer Vision + Deep Learning + AI Engineering
- **Timeline**: 4-7 days intensive development
- **Model**: EfficientNetV2-B0 with transfer learning (LP-FT)
- **Deployment**: FastAPI + Streamlit, ONNX Runtime serving
- **Dataset**: [Ethiopian Coffee Leaf Disease Dataset](https://www.kaggle.com/datasets/biniyamyoseph/ethiopian-coffee-leaf-disease/data) (CC0)

---

## ✨ Key Features

### 🔬 Core Capabilities

- **Multi-class Classification**: Accurate disease identification across 4 classes
- **Transfer Learning**: EfficientNetV2-B0 pre-trained on ImageNet
- **Explainable AI**: Grad-CAM visualizations showing model attention
- **Robustness Testing**: Evaluation against realistic image perturbations
- **OOD Detection**: Rejects irrelevant/out-of-distribution inputs

### 🛠 Engineering Excellence

- **Data Quality Pipeline**: Validation, exact / near / rotated-copy duplicate detection, label-issue audit
- **Leakage-safe Splitting**: Group-aware stratified 70/15/15 split — copies of one leaf never cross splits
- **Confidence Analysis**: Calibration, conformal prediction sets, high/low confidence error analysis
- **Model Comparison**: Benchmark multiple architectures on accuracy, latency, size and robustness
- **Production-Ready API**: FastAPI inference service on a slim ONNX Runtime bundle
- **Interactive UI**: Streamlit web application

---

## 📁 Project Structure

```
coffee-guard-ai/
├── src/coffeeguard/          # Python package (installed; CLI: `coffeeguard`)
│   ├── cli.py                # every pipeline step is one command
│   ├── config.py             # typed (pydantic) configs
│   ├── data/                 # download, scan/validate, dedup, split, report, dataset
│   ├── embeddings/           # DINOv2 embeddings, label audit, probe baseline
│   ├── models/ training/     # timm factory, multi-stage trainer, Kaggle GPU runner
│   ├── evaluation/ export/   # metrics, ONNX export + parity check
│   ├── explainability/ robustness/ ood/
│   ├── inference/            # slim runtime: preprocessing, quality, ONNX predictor
│   └── utils/
├── apps/api/                 # FastAPI service
├── apps/web/                 # Streamlit UI
├── configs/                  # data.yaml, train/*.yaml recipes
├── kaggle/                   # GPU kernel template
├── data/                     # raw/ processed/ manifests/ (git-ignored), splits/ (committed)
├── artifacts/                # figures, reports, metrics, model bundles
├── notebooks/                # narrative only
├── tests/                    # unit + integration (synthetic fixtures)
└── docs/                     # plan, progress log, specification, decisions
```

---

## 🚀 Getting Started

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (manages Python 3.12 and all dependencies)
- A Kaggle account and API token (for the dataset and free GPU training)
- ~8 GB free disk space

### Installation

```bash
git clone https://github.com/Ab-xo/coffee-guard-ai.git
cd coffee-guard-ai
uv sync --all-extras          # creates .venv with Python 3.12
uv run coffeeguard --help
```

Kaggle authentication: run `kaggle auth login`, or save your token from
<https://www.kaggle.com/settings/api> as `~/.kaggle/access_token`.

### Pipeline

```bash
uv run coffeeguard data download        # Kaggle -> data/raw
uv run coffeeguard data prepare         # scan, embed, dedup/group, split
uv run coffeeguard data report          # EDA figures + eda_summary.json
uv run coffeeguard embed audit          # label-issue audit + DINOv2 probe baseline
uv run coffeeguard train -c configs/train/effnetv2_b0.yaml            # local (CPU/GPU)
uv run coffeeguard remote train -c configs/train/effnetv2_b0.yaml --seeds 0,1,2  # Kaggle GPU
uv run coffeeguard export --run runs/<run-dir>                        # ONNX bundle
```

### Serve and try it

```bash
# API (http://localhost:8000/docs); serves artifacts/models/coffeeguard-effv2b0-v1 by default
uv run uvicorn app.main:app --app-dir apps/api
# UI (http://localhost:8501), in a second terminal
uv run streamlit run apps/web/streamlit_app.py
```

Endpoints: `GET /health`, `GET /model-info`, `POST /predict` (decision + advice), `POST /analyze` (+ probabilities, quality report, OOD score, heat map). Another bundle: set `MODEL_BUNDLE` (PowerShell: `$env:MODEL_BUNDLE = "artifacts/models/<bundle>"`).

### Tests and lint

```bash
uv run pytest -m "not slow"   # fast unit tests
uv run pytest                 # + train/export smoke test
uv run ruff check . && uv run ruff format --check .
```

---

## 🔄 Development Status

### Current Progress: Phase 9 next (Streamlit UI)

| Phase  | Description           |       Status       | Key Deliverables                                                                                                                                                                |
| :----: | --------------------- | :----------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **0**  | **Foundation**        |  ✅ **Complete**   | • Modern tooling (uv, ruff, pre-commit)<br>• CI/CD pipeline<br>• Type-safe configuration<br>• CLI interface                                                                     |
| **1**  | **Data Engineering**  |  ✅ **Complete**   | • Dataset validation & cleaning<br>• Duplicate detection (exact/near/rotated)<br>• Leakage-safe splitting (70/15/15)<br>• EDA reports & visualizations<br>• Label quality audit |
| **2** | **Training Pipeline** | ✅ **Complete** | • Multi-stage trainer (LP → FT)<br>• Quality-equalising augmentation<br>• MobileNetV3-Small baseline: val macro-F1 **0.991**<br>• Walking skeleton: ONNX → FastAPI → Streamlit |
| **3** | **Main Model Training** | ✅ **Complete** | • EfficientNetV2-B0, LP-FT (last 3 stages): val macro-F1 **0.985**<br>• Variant ablation A vs. B (tie → A)<br>• Comparison runs: EfficientNet-B0, MobileNetV3-L/S<br>• Kaggle T4 GPU, 1 seed, ~3–4 min per run |
| **4** | **Evaluation** | ✅ **Complete** | • Test macro-F1 **0.979** [0.963, 0.993] (EffV2-B0)<br>• Calibration: ECE 0.098 → 0.012<br>• Conformal 98% sets: coverage 0.987<br>• Error analysis (4/8 errors on audit-flagged labels) |
| **5** | **Explainability & Robustness** | ✅ **Complete** | • CAM = Grad-CAM (verified), faithful (deletion test)<br>• Relative robustness **0.975** (9 corruptions, sev ≤ 3)<br>• Shortcut test: background-only acc 0.40<br>• Found: blue-paper backgrounds carry class information |
| **6** | **Quality & OOD gates** | ✅ **Complete** | • KNN OOD detector: AUROC near 0.993 / far 1.000 (unseen sources)<br>• Quality gate fitted to where accuracy drops<br>• Accepted answers 99.4% correct; 96% of non-coffee images rejected |
| **7** | **Model Comparison & Export** | ✅ **Complete** | • 5 models compared (accuracy, leaf reliance, OOD, latency, size)<br>• Chosen: EfficientNetV2-B0 + background swap ([ADR](docs/decisions/001-deployment-model.md))<br>• Release bundle `coffeeguard-effv2b0-v1`, 63 ms per photo on CPU<br>• [Model card](docs/MODEL_CARD.md) |
| **8** | **FastAPI Service** | ✅ **Complete** | • `/health`, `/model-info`, `/predict`, `/analyze` (heat map)<br>• Quality + OOD gates, accepted / uncertain / rejected with advice<br>• Request IDs, JSON logs, upload hardening; 23 API tests<br>• Serves the offline results exactly (37 ms per photo) |
| **9**  | **Streamlit UI**      |     ⏳ **Next**     | • Interactive web interface<br>• Explainability dashboard                                                                                                                       |
| **10** | **Deployment**        |     ⏳ Planned     | • Docker containers<br>• Final documentation                                                                                                                                    |

**Legend:** ✅ Complete • 🔄 In Progress • ⏳ Planned

### What's Been Accomplished

#### ✅ Phase 0 & 1 Complete (Foundation + Data)

- **107 files changed:** 67 added, 11 modified, 29 removed
- **Production-ready data pipeline** with validation, deduplication, and group-aware splitting
- **Comprehensive test suite** (5 unit tests + 1 integration test)
- **Complete ML infrastructure** ready for training (models, trainers, exporters)
- **Baseline established:** DINOv2 probe classifier + label quality audit

#### ✅ Phase 2 Complete (Training pipeline + walking skeleton)

- **Baseline trained:** MobileNetV3-Small reaches val macro-F1 **0.991** / accuracy 0.992 (DINOv2 probe reference: 0.961)
- **End to end:** ONNX bundle (torch parity 100%) served by FastAPI (`/health`, `/predict`), used by a Streamlit page
- **Fixed:** classifier initialisation that kept the linear-probe stage at chance

#### ✅ Phase 3 Complete (Main model)

- **EfficientNetV2-B0** (linear probe → fine-tune last 3 stages) reaches val macro-F1 **0.985** in under 4 min on a Kaggle T4
- Comparison models trained with the same recipe: EfficientNet-B0 0.984, MobileNetV3-Small 0.984, MobileNetV3-Large 0.979 — within 1–3 validation images of each other
- Model choice is decided in Phases 4–7 on test-set confidence intervals, calibration, robustness, speed and size

#### ✅ Phase 4 Complete (Evaluation)

- **Test macro-F1 0.979** [95% CI 0.963–0.993], ROC-AUC 0.997, every class F1 ≥ 0.97 (EfficientNetV2-B0; test set opened once)
- **Calibrated:** temperature scaling cuts ECE from 0.098 to 0.012; **98% conformal prediction sets** reach 0.987 coverage and flag 38% of errors as uncertain
- Beats MobileNetV3-Large (paired bootstrap), ties EfficientNet-B0; half of its 8 test errors are images the label audit had already flagged

#### ✅ Phase 5 Complete (Explainability & robustness)

- **Explanations are faithful:** the torch-free CAM matches Grad-CAM (r > 0.99); hiding its hottest 5% of patches drops confidence 0.97 → 0.72 (random: 0.88); 61% of CAM mass sits on the leaf (24% of the image)
- **Robust:** keeps 97.5% of its macro-F1 across 9 corruptions at severity ≤ 3; heavy blur/noise/JPEG make it answer *Healthy*, not the low-quality classes — evidence against a photo-quality shortcut
- **Limitation found:** with the leaf removed, blue-paper backgrounds are still called Cercospora/Leaf Rust — the model partly learned each class's photo setup
- **Fix applied — background-swap augmentation** (leaves pasted onto other photos' backgrounds during training): leaf-only accuracy 0.926 → 0.963, confidence on leaf-less images 0.83 → 0.54, robustness 0.979; test macro-F1 0.974 (difference not significant). This model is now the main model; blue-paper backgrounds still lean to Cercospora/Leaf Rust (needs field photos)

#### ✅ Phase 6 Complete (Quality & OOD gates)

- **Rejects what it shouldn't answer:** a KNN detector on the model's embeddings separates coffee leaves from other plant leaves (AUROC 0.993) and non-leaf images (1.000), tested on image sources never used for tuning
- **Quality gate** asks for a retake only where the model's accuracy really drops (0.5% of genuine photos rejected, not 94% of slightly dark ones as a naive rule did)
- **End to end on test:** 90.8% of genuine photos answered, 99.4% of those correct; the rest get *uncertain + top candidates* or a retake request; 175 of 182 non-coffee images rejected

#### ✅ Phase 7 Complete (Model choice & release)

- **Chosen: EfficientNetV2-B0 + background swap** — tied on accuracy with EfficientNet-B0 and plain EfficientNetV2-B0, but relies most on the leaf rather than the photo setup ([decision record](docs/decisions/001-deployment-model.md))
- **Fast on a laptop CPU:** 15 ms per forward pass, 63 ms for a full 2048 px phone photo
- **Release bundle** `artifacts/models/coffeeguard-effv2b0-v1` (committed) reproduces the test results exactly; see the [model card](docs/MODEL_CARD.md)

#### ✅ Phase 8 Complete (API)

- **FastAPI service** with `/predict` (accepted / uncertain / rejected + advice) and `/analyze` (+ probabilities, quality report, OOD score, heat map)
- **Same answers as the offline evaluation** on all 379 test photos and 182 non-coffee images; 37 ms per request
- Request IDs, JSON logs, upload checks (type by content, size, pixels, corrupt files, EXIF rotation); 23 tests against a tiny generated model

#### How far does it meet the project goal?

| Goal | Status |
|---|---|
| Predict the disease from a leaf photo | ✅ 4 classes, test macro-F1 0.974 |
| Confidence, explanation, robustness, rejection | ✅ calibrated confidence + *uncertain* answers, faithful heat maps, 9×5 corruption sweep, quality + OOD gates |
| Lighting and blur | ✅ mostly (darkening and blur tolerated; colour casts untested) |
| Background and framing | ⚠️ partly (background swap helps; blue-paper cue remains; one leaf per photo) |
| Disease severity | ⚠️ proxy only — mild cases not worse, but very early infection isn't in the data |
| Real field photos | ❌ not yet tested — **key next step:** 30–50 farm photos per class labelled by an agronomist |

### Quick Status Check

```bash
# See detailed progress with decisions and results
cat docs/PROGRESS.md

# See all changes in this release
cat CHANGELOG.md

# Verify everything works
uv run pytest -m "not slow"  # Fast unit tests only
```

For detailed technical decisions, experiment results, and step-by-step progress: [docs/PROGRESS.md](docs/PROGRESS.md).

---

## 📚 Documentation

- **[📝 CHANGELOG](CHANGELOG.md)** — what changed in each release, organized by phase
- **[🗺️ Implementation Plan](docs/IMPLEMENTATION_PLAN.md)** — phases, design decisions, success targets
- **[📊 Progress Log](docs/PROGRESS.md)** — what was built, results and decisions, step by step
- **[📋 Technical Specification](docs/PROJECT_SPECIFICATION.md)** — original requirements baseline
- **[🤝 Contributing](CONTRIBUTING.md)** — development workflow and conventions

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: [Ethiopian Coffee Leaf Disease Dataset](https://www.kaggle.com/datasets/biniyamyoseph/ethiopian-coffee-leaf-disease/data) by Biniyam Yoseph
- **EfficientNet**: Google Research · **DINOv2**: Meta AI
- **Community**: PyTorch, timm, FastAPI, and Streamlit communities

---

<div align="center">
  <strong>Built with ❤️ for Ethiopian Coffee Farmers</strong>
</div>
