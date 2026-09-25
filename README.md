# 🍃 CoffeeGuard AI

> **Intelligent Coffee Leaf Disease Detection System for Ethiopian Coffee**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Project Status](https://img.shields.io/badge/status-Phase%206%20next-orange.svg)](docs/PROGRESS.md)

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
# API (http://localhost:8000/docs)
MODEL_BUNDLE=artifacts/models/<bundle> uv run uvicorn app.main:app --app-dir apps/api
# UI (http://localhost:8501), in a second terminal
uv run streamlit run apps/web/streamlit_app.py
```

PowerShell: `$env:MODEL_BUNDLE = "artifacts/models/<bundle>"` before the `uvicorn` line.

### Tests and lint

```bash
uv run pytest -m "not slow"   # fast unit tests
uv run pytest                 # + train/export smoke test
uv run ruff check . && uv run ruff format --check .
```

---

## 🔄 Development Status

### Current Progress: Phase 6 next (OOD gate)

| Phase  | Description           |       Status       | Key Deliverables                                                                                                                                                                |
| :----: | --------------------- | :----------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **0**  | **Foundation**        |  ✅ **Complete**   | • Modern tooling (uv, ruff, pre-commit)<br>• CI/CD pipeline<br>• Type-safe configuration<br>• CLI interface                                                                     |
| **1**  | **Data Engineering**  |  ✅ **Complete**   | • Dataset validation & cleaning<br>• Duplicate detection (exact/near/rotated)<br>• Leakage-safe splitting (70/15/15)<br>• EDA reports & visualizations<br>• Label quality audit |
| **2** | **Training Pipeline** | ✅ **Complete** | • Multi-stage trainer (LP → FT)<br>• Quality-equalising augmentation<br>• MobileNetV3-Small baseline: val macro-F1 **0.991**<br>• Walking skeleton: ONNX → FastAPI → Streamlit |
| **3** | **Main Model Training** | ✅ **Complete** | • EfficientNetV2-B0, LP-FT (last 3 stages): val macro-F1 **0.985**<br>• Variant ablation A vs. B (tie → A)<br>• Comparison runs: EfficientNet-B0, MobileNetV3-L/S<br>• Kaggle T4 GPU, 1 seed, ~3–4 min per run |
| **4** | **Evaluation** | ✅ **Complete** | • Test macro-F1 **0.979** [0.963, 0.993] (EffV2-B0)<br>• Calibration: ECE 0.098 → 0.012<br>• Conformal 98% sets: coverage 0.987<br>• Error analysis (4/8 errors on audit-flagged labels) |
| **5** | **Explainability & Robustness** | ✅ **Complete** | • CAM = Grad-CAM (verified), faithful (deletion test)<br>• Relative robustness **0.975** (9 corruptions, sev ≤ 3)<br>• Shortcut test: background-only acc 0.40<br>• Found: blue-paper backgrounds carry class information |
| **6**  | **Robustness & OOD**  |     ⏳ **Next**     | • Perturbation testing<br>• Out-of-distribution detection                                                                                                                       |
| **7**  | **Model Comparison**  |     ⏳ Planned     | • Multi-architecture benchmark<br>• ONNX export                                                                                                                                 |
| **8**  | **FastAPI Service**   |     ⏳ Planned     | • REST API<br>• ONNX Runtime serving                                                                                                                                            |
| **9**  | **Streamlit UI**      |     ⏳ Planned     | • Interactive web interface<br>• Explainability dashboard                                                                                                                       |
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
