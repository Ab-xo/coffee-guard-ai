# Changelog

All notable changes to the CoffeeGuard AI project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Phase 1 Completed - Data Pipeline ✅

#### Added - Data Engineering
- **Complete data pipeline implementation**
  - `src/coffeeguard/data/download.py` - Kaggle dataset downloader with integrity checks
  - `src/coffeeguard/data/scan.py` - Image validation (format, size, corruption detection)
  - `src/coffeeguard/data/dedup.py` - Exact, near-duplicate, and rotated-copy detection using perceptual hashing
  - `src/coffeeguard/data/split.py` - Group-aware stratified splitting (70/15/15) preventing data leakage
  - `src/coffeeguard/data/report.py` - Automated EDA with visualizations and JSON summaries
  - `src/coffeeguard/data/dataset.py` - PyTorch Dataset implementation with augmentation
  - `src/coffeeguard/data/transforms.py` - Training and inference transforms

#### Added - ML Infrastructure
- **Model and training framework**
  - `src/coffeeguard/models/factory.py` - timm model factory with EfficientNet support
  - `src/coffeeguard/training/trainer.py` - Multi-stage trainer (LP → FT)
  - `src/coffeeguard/training/remote.py` - Kaggle GPU kernel integration
  - `src/coffeeguard/evaluation/metrics.py` - Classification metrics calculation
  - `src/coffeeguard/export/onnx_export.py` - ONNX export with parity checking

#### Added - Embeddings & Analysis
- **DINOv2 embeddings pipeline**
  - `src/coffeeguard/embeddings/extract.py` - Feature extraction for label auditing
  - `src/coffeeguard/embeddings/audit.py` - Cleanlab-based label issue detection
  - Baseline probe classifier using DINOv2 features

#### Added - Inference Runtime
- **Production-ready inference module**
  - `src/coffeeguard/inference/predictor.py` - ONNX Runtime predictor
  - `src/coffeeguard/inference/preprocess.py` - Inference preprocessing pipeline
  - `src/coffeeguard/inference/quality.py` - Input quality checks

#### Added - Testing
- **Comprehensive test suite**
  - `tests/fixtures/synthetic.py` - Synthetic dataset generator for testing
  - `tests/unit/test_cli.py` - CLI command tests
  - `tests/unit/test_config.py` - Configuration validation tests
  - `tests/unit/test_data_pipeline.py` - Data pipeline tests
  - `tests/unit/test_inference_core.py` - Inference tests
  - `tests/unit/test_utils.py` - Utility function tests
  - `tests/integration/test_training_smoke.py` - End-to-end training smoke test

#### Added - Configuration
- **Configuration files**
  - `configs/data.yaml` - Data pipeline configuration
  - `configs/train/*.yaml` - Training recipes (EfficientNet, MobileNet variants, smoke test)
  - `.pre-commit-config.yaml` - Pre-commit hooks (ruff, nbstripout, file checks)
  - `.github/workflows/ci.yml` - CI/CD pipeline
  - `.python-version` - Python 3.12 pinned
  - `.gitattributes` - Git LFS configuration

#### Added - Documentation
- `docs/IMPLEMENTATION_PLAN.md` - Detailed phase-by-phase implementation roadmap
- `docs/PROGRESS.md` - Running progress log with decisions and results
- `kaggle/run_template.py` - Template for remote GPU training

#### Added - Artifacts
- **Data quality reports**
  - `artifacts/reports/eda_summary.json` - Dataset statistics
  - `artifacts/reports/label_audit.json` - Label quality analysis
  - `artifacts/reports/label_issues.csv` - Detected label problems
  - `artifacts/reports/data_quality_removed.csv` - Removed problematic samples
  
- **EDA Visualizations**
  - `artifacts/figures/eda/class_distribution.png` - Class balance visualization
  - `artifacts/figures/eda/image_sizes.png` - Image dimension distribution
  - `artifacts/figures/eda/resolution_by_class.png` - Resolution analysis per class
  - `artifacts/figures/eda/quality_by_class.png` - Quality metrics by class
  - `artifacts/figures/eda/duplicate_groups.png` - Duplicate detection results
  - `artifacts/figures/eda/label_issues.png` - Label audit visualization
  - `artifacts/figures/eda/samples_*.png` - Sample images from each class

- **Baseline metrics**
  - `artifacts/metrics/baseline_dinov2_probe_val.json` - DINOv2 probe baseline results

#### Added - Data Splits
- `data/splits/train.csv` - Training set (70%)
- `data/splits/val.csv` - Validation set (15%)
- `data/splits/test.csv` - Test set (15%)
- `data/splits/split_info.json` - Split metadata and statistics
- `data/splits/data_config.yaml` - Data configuration snapshot

### Phase 0 Completed - Foundation ✅

#### Changed - Project Structure
- **Package reorganization**
  - Moved `ml/` → `src/coffeeguard/` (modern src layout)
  - Removed empty placeholder packages
  - Added new modules: `utils/`, `embeddings/`, `export/`, `inference/`

#### Changed - Build System
- **Modern Python tooling**
  - Switched to `uv` package manager for fast, reliable dependency management
  - Updated `pyproject.toml` with `uv_build` backend
  - Added optional dependency groups:
    - `[inference]` - ONNX Runtime (slim deployment)
    - `[ml]` - Full ML stack (torch, timm, scikit-learn, etc.)
    - `[api]` - FastAPI service dependencies
    - `[web]` - Streamlit UI dependencies
  - Dev dependencies: ruff, pytest, pytest-cov, mypy, pre-commit

#### Changed - Core Utilities
- **Configuration system** (`src/coffeeguard/config.py`)
  - Pydantic-based type-safe configuration
  - CLI override support (`--set key=value`)
  - Automatic path resolution and validation

- **Utility modules** (`src/coffeeguard/utils/`)
  - `paths.py` - Project root detection and path resolution
  - `hashing.py` - SHA256 hashing and fingerprinting
  - `seed.py` - Reproducible random seed management
  - `log.py` - Rich logging setup
  - `io.py` - JSON I/O with NumPy/Path support
  - `runs.py` - Experiment run directory management
  - `plotting.py` - Visualization utilities

#### Changed - CLI
- **Command-line interface** (`src/coffeeguard/cli.py`)
  - `coffeeguard info` - System and config information
  - `coffeeguard data download` - Download Kaggle dataset
  - `coffeeguard data prepare` - Run full data pipeline
  - `coffeeguard data report` - Generate EDA
  - `coffeeguard embed audit` - Label quality audit
  - `coffeeguard train` - Local training
  - `coffeeguard remote train` - Kaggle GPU training
  - `coffeeguard export` - ONNX export

#### Changed - Documentation
- Updated `README.md` with modern structure and getting started guide
- Updated `CONTRIBUTING.md` with development workflow
- Updated `.env.example` with comprehensive environment variables

#### Changed - Git Configuration
- Enhanced `.gitignore` for ML project patterns
  - Virtual environments, cache directories
  - Large model files and artifacts
  - Data directories (raw/processed)
  - Build artifacts and logs

#### Removed
- `PHASE_1_CHECKLIST.md` - Consolidated into PROGRESS.md
- `docs/GIT_WORKFLOW.md` - Merged into CONTRIBUTING.md
- `docs/QUICK_START.md` - Merged into README.md
- `docs/TEAM_ROLES.md` - No longer needed
- `docs/architecture/.gitkeep` - Empty placeholder
- `docs/decisions/.gitkeep` - Empty placeholder
- `docs/experiments/.gitkeep` - Empty placeholder
- `scripts/.gitkeep` - Empty placeholder
- `configs/.gitkeep` - Replaced with actual configs
- Various `.gitkeep` files in `artifacts/` - Replaced with actual artifacts

---

## [0.1.0] - Initial Scaffold

### Added
- Initial project structure with empty package directories
- Basic documentation (PROJECT_SPECIFICATION.md, initial README)
- MIT License
- GitHub issue and PR templates
- Initial pyproject.toml with setuptools

---

## Summary of Changes by Phase

| Phase | Description | Status | Files Changed |
|-------|-------------|--------|---------------|
| **Phase 0** | Foundation - tooling, structure, CI | ✅ Complete | ~25 files |
| **Phase 1** | Data engineering - validation, dedup, split, EDA | ✅ Complete | ~82 files |
| **Phase 2** | Training pipeline, baseline, walking skeleton | 🔄 Code complete | - |
| **Phase 3** | EfficientNetV2-B0 training (LP-FT, 3 seeds) | ⏳ Planned | - |
| **Phase 4** | Evaluation, calibration, conformal prediction | ⏳ Planned | - |
| **Phase 5** | Explainability and robustness | ⏳ Planned | - |
| **Phase 6** | OOD detection gate | ⏳ Planned | - |
| **Phase 7** | Model comparison and export | ⏳ Planned | - |
| **Phase 8** | FastAPI service | ⏳ Planned | - |
| **Phase 9** | Streamlit UI | ⏳ Planned | - |
| **Phase 10** | Docker, documentation, final polish | ⏳ Planned | - |

---

## How to Read This Changelog

- **Added** - New features, files, or capabilities
- **Changed** - Changes to existing functionality
- **Deprecated** - Features that will be removed in future versions
- **Removed** - Deleted features or files
- **Fixed** - Bug fixes
- **Security** - Security-related changes

For detailed technical decisions and step-by-step progress, see [docs/PROGRESS.md](docs/PROGRESS.md).
