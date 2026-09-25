# 🍃 CoffeeGuard AI

> **Intelligent Coffee Leaf Disease Detection System for Ethiopian Coffee**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Project Status](https://img.shields.io/badge/status-Phase%202%20in%20progress-orange.svg)](docs/PROGRESS.md)

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

### Tests and lint

```bash
uv run pytest -m "not slow"   # fast unit tests
uv run pytest                 # + train/export smoke test
uv run ruff check . && uv run ruff format --check .
```

---

## 🔄 Development Status

| Phase | Status |
|---|---|
| 0 — Foundation (uv, ruff, CI, typed config, CLI) | ✅ |
| 1 — Data engineering (validation, dedup, split, EDA, label audit) | ✅ |
| 2 — Training pipeline, baseline, walking skeleton | 🔄 code written, smoke-tested |
| 3 — EfficientNetV2-B0 (LP-FT, 3 seeds) | ⏳ |
| 4 — Evaluation, calibration, conformal prediction | ⏳ |
| 5 — Explainability and robustness | ⏳ |
| 6 — OOD gate | ⏳ |
| 7 — Model comparison and export | ⏳ |
| 8–10 — API, UI, Docker, documentation | ⏳ |

Details, numbers and decisions for every step: [docs/PROGRESS.md](docs/PROGRESS.md).

---

## 📚 Documentation

- **[Implementation Plan](docs/IMPLEMENTATION_PLAN.md)** — phases, design decisions, success targets
- **[Progress Log](docs/PROGRESS.md)** — what was built, results and decisions, step by step
- **[Technical Specification](docs/PROJECT_SPECIFICATION.md)** — original requirements baseline
- **[Project Brief](CoffeeGuard_AI_Project_Structure.md)** — original project outline
- **[Contributing](CONTRIBUTING.md)** — development workflow and conventions

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
