# 🍃 CoffeeGuard AI

> **Intelligent Coffee Leaf Disease Detection System for Ethiopian Coffee**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Project Status](https://img.shields.io/badge/status-Phase%201-orange.svg)]()

An AI-powered computer vision system for detecting and classifying coffee leaf diseases in Ethiopian coffee plants using deep learning, explainable AI, and production-ready deployment.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Team & Contributions](#team--contributions)
- [Development Status](#development-status)
- [Documentation](#documentation)
- [License](#license)

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
- **Model**: EfficientNetV2-B0 with transfer learning
- **Deployment**: FastAPI + Streamlit
- **Dataset**: [Ethiopian Coffee Leaf Disease Dataset](https://www.kaggle.com/datasets/biniyamyoseph/ethiopian-coffee-leaf-disease/data)

---

## ✨ Key Features

### 🔬 Core Capabilities

- **Multi-class Classification**: Accurate disease identification across 4 classes
- **Transfer Learning**: EfficientNetV2-B0 pre-trained on ImageNet
- **Explainable AI**: Grad-CAM visualizations showing model attention
- **Robustness Testing**: Evaluation against realistic image perturbations
- **OOD Detection**: Rejects irrelevant/out-of-distribution inputs

### 🛠 Engineering Excellence

- **Data Quality Pipeline**: Automated validation, duplicate detection, corruption checks
- **Stratified Splitting**: Leakage-safe 70/15/15 train/val/test splits
- **Confidence Analysis**: High/low confidence error analysis
- **Model Comparison**: Benchmark multiple architectures
- **Production-Ready API**: FastAPI inference service
- **Interactive UI**: Streamlit web application

---

## 📁 Project Structure

```
coffee-guard-ai/
├── 📂 apps/                    # Application layer
│   ├── api/                    # FastAPI service
│   └── web/                    # Streamlit UI
│
├── 📂 ml/                      # Machine learning core
│   ├── data/                   # Data ingestion, validation, splitting
│   ├── preprocessing/          # Image transforms and augmentation
│   ├── models/                 # Model architectures
│   ├── training/               # Training loops and fine-tuning
│   ├── evaluation/             # Metrics and evaluation
│   ├── explainability/         # Grad-CAM and interpretability
│   ├── robustness/             # Perturbation testing
│   └── ood/                    # Out-of-distribution detection
│
├── 📂 data/                    # Data storage
│   ├── raw/                    # Original Kaggle dataset
│   ├── processed/              # Cleaned and validated data
│   ├── splits/                 # Train/val/test splits
│   ├── manifests/              # Data tracking manifests
│   └── interim/                # Intermediate processing outputs
│
├── 📂 notebooks/               # Jupyter notebooks for experiments
│
├── 📂 tests/                   # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── fixtures/               # Test fixtures
│
├── 📂 configs/                 # Configuration files
│
├── 📂 artifacts/               # Training outputs
│   ├── checkpoints/            # Model checkpoints
│   ├── metrics/                # Evaluation metrics
│   ├── figures/                # Visualizations
│   └── reports/                # Analysis reports
│
├── 📂 scripts/                 # Utility scripts
│
├── 📂 docs/                    # Documentation
│   ├── architecture/           # System design docs
│   ├── decisions/              # Architectural Decision Records
│   ├── experiments/            # Experiment logs
│   └── PROJECT_SPECIFICATION.md
│
└── 📂 .github/                 # GitHub workflows and templates
    ├── workflows/              # CI/CD pipelines
    └── ISSUE_TEMPLATE/         # Issue templates
```

For detailed structure explanation, see [CoffeeGuard_AI_Project_Structure.md](CoffeeGuard_AI_Project_Structure.md)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- 4GB+ available disk space for dataset
- (Optional) GPU for faster training

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/coffee-guard-ai.git
cd coffee-guard-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Quick Start

```bash
# 1. Download dataset from Kaggle
# Place it in data/raw/

# 2. Run data validation
python -m ml.data.validation

# 3. Train baseline model
python -m ml.training.baseline

# 4. Start API server
python -m apps.api.app

# 5. Launch Streamlit UI
streamlit run apps/web/app.py
```

---

## 👥 Team & Contributions

This is a **collaborative team project** with **6 members**. We follow a structured contribution workflow to ensure code quality and smooth collaboration.

### How to Contribute

1. Read [CONTRIBUTING.md](CONTRIBUTING.md) for detailed workflow
2. Pick an issue or create one
3. Create a feature branch
4. Submit a pull request
5. Get code review approval
6. Merge to main

### Team Workflow

- **Branching Strategy**: GitFlow (main, develop, feature/_, bugfix/_)
- **Code Review**: Required before merge
- **Commits**: Follow conventional commits format
- **Communication**: Use GitHub issues and PR discussions

---

## 🔄 Development Status

### Phase 1: Project Setup ✅ (Current)

- [x] Repository structure
- [x] Documentation foundation
- [x] Team workflow guidelines
- [ ] Development environment setup
- [ ] CI/CD pipeline configuration

### Phase 2: Data Engineering (Upcoming)

- [ ] Dataset download and validation
- [ ] EDA and quality analysis
- [ ] Duplicate detection
- [ ] Stratified splitting
- [ ] Data manifests

### Phase 3: Model Development

- [ ] Baseline model
- [ ] Transfer learning pipeline
- [ ] Fine-tuning strategy
- [ ] Hyperparameter optimization

### Phase 4: Evaluation & Analysis

- [ ] Metrics computation
- [ ] Error analysis
- [ ] Confidence analysis
- [ ] Grad-CAM implementation

### Phase 5: Robustness & OOD

- [ ] Perturbation testing
- [ ] OOD detection
- [ ] Model comparison

### Phase 6: Deployment

- [ ] FastAPI service
- [ ] Streamlit UI
- [ ] Docker containerization
- [ ] Documentation finalization

---

## 📚 Documentation

- **[Project Specification](CoffeeGuard_AI_Project_Structure.md)**: Complete technical specification
- **[Contributing Guide](CONTRIBUTING.md)**: Team workflow and guidelines
- **[Architecture Docs](docs/architecture/)**: System design documents
- **[Experiment Logs](docs/experiments/)**: Training experiments and results

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: [Ethiopian Coffee Leaf Disease Dataset](https://www.kaggle.com/datasets/biniyamyoseph/ethiopian-coffee-leaf-disease/data) by Biniyam Yoseph
- **EfficientNet**: Google Research
- **Community**: PyTorch, FastAPI, and Streamlit communities

---

## 📧 Contact

For questions or suggestions, please open an issue or contact the project maintainers.

---

<div align="center">
  <strong>Built with ❤️ for Ethiopian Coffee Farmers</strong>
</div>
