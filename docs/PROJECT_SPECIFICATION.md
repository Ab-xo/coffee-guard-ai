# CoffeeGuard AI - Technical Project Specification

**Version**: 1.0  
**Last Updated**: Phase 1  
**Status**: Active Development

---

## Document Overview

This document provides the complete technical specification for the CoffeeGuard AI project. It serves as the single source of truth for architecture decisions, technical requirements, and implementation details.

---

## Table of Contents

1. [Project Summary](#1-project-summary)
2. [Technical Requirements](#2-technical-requirements)
3. [System Architecture](#3-system-architecture)
4. [Data Pipeline Specification](#4-data-pipeline-specification)
5. [Model Specification](#5-model-specification)
6. [Evaluation Framework](#6-evaluation-framework)
7. [API Specification](#7-api-specification)
8. [UI Specification](#8-ui-specification)
9. [Infrastructure & Deployment](#9-infrastructure--deployment)
10. [Quality Assurance](#10-quality-assurance)
11. [Timeline & Milestones](#11-timeline--milestones)

---

## 1. Project Summary

### 1.1 Problem Statement

Ethiopian coffee farmers face significant challenges in rapidly identifying coffee leaf diseases. Manual identification requires expertise and is time-consuming, leading to delayed treatment and crop losses. CoffeeGuard AI provides an automated, accurate, and explainable solution for real-time disease detection.

### 1.2 Solution Overview

An end-to-end deep learning system that:

- Classifies coffee leaf images into 4 disease categories
- Provides confidence scores and visual explanations
- Handles realistic image variations and quality issues
- Rejects out-of-distribution inputs
- Exposes predictions via REST API
- Provides an intuitive web interface

### 1.3 Target Classes

| Class ID | Class Name | Description                                      |
| -------- | ---------- | ------------------------------------------------ |
| 0        | Healthy    | No visible disease symptoms                      |
| 1        | Cercospora | Brown leaf spots caused by Cercospora coffeicola |
| 2        | Leaf Rust  | Orange/yellow pustules (Hemileia vastatrix)      |
| 3        | Phoma      | Dark brown lesions with lighter centers          |

### 1.4 Success Criteria

- **Macro F1 Score**: ≥ 0.85 on test set
- **Per-Class F1**: ≥ 0.80 for each disease class
- **Inference Latency**: < 200ms per image
- **OOD Rejection Rate**: ≥ 90% for clearly irrelevant inputs
- **Grad-CAM Relevance**: Focus on leaf regions (manual validation)
- **API Uptime**: 99%+ availability

---

## 2. Technical Requirements

### 2.1 Technology Stack

#### Core ML Framework

- **Python**: 3.9+
- **PyTorch**: 2.0+
- **torchvision**: Latest stable
- **timm** (PyTorch Image Models): For EfficientNetV2

#### Data Processing

- **PIL/Pillow**: Image loading and manipulation
- **OpenCV**: Advanced image processing
- **NumPy**: Numerical operations
- **Pandas**: Data manifests and metadata

#### Visualization & Analysis

- **Matplotlib**: Plotting and visualization
- **Seaborn**: Statistical visualizations
- **scikit-learn**: Metrics and utilities
- **pytorch-grad-cam**: Grad-CAM implementation

#### API & Deployment

- **FastAPI**: REST API framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation
- **Streamlit**: Web UI

#### Development Tools

- **pytest**: Testing framework
- **black**: Code formatting
- **flake8**: Linting
- **mypy**: Type checking
- **pre-commit**: Git hooks

### 2.2 Hardware Requirements

#### Development

- **Minimum**: 8GB RAM, CPU-only training possible but slow
- **Recommended**: 16GB RAM, NVIDIA GPU with 6GB+ VRAM
- **Optimal**: 32GB RAM, NVIDIA GPU with 8GB+ VRAM

#### Production

- **API Server**: 4GB RAM, 2 CPU cores
- **Model Serving**: 8GB RAM, GPU optional (CPU acceptable for < 10 req/s)

### 2.3 Dataset Requirements

- **Source**: [Kaggle - Ethiopian Coffee Leaf Disease Dataset](https://www.kaggle.com/datasets/biniyamyoseph/ethiopian-coffee-leaf-disease/data)
- **Format**: JPEG/PNG images
- **Expected Size**: 1000-2000 images (verify actual)
- **License**: Check Kaggle dataset page
- **Storage**: ~2-5GB raw data

---

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface Layer                    │
│  ┌──────────────────────┐      ┌──────────────────────┐    │
│  │  Streamlit Web UI    │      │   Mobile/Web Client  │    │
│  └──────────┬───────────┘      └──────────┬───────────┘    │
└─────────────┼──────────────────────────────┼────────────────┘
              │                              │
              └──────────────┬───────────────┘
                             │ HTTPS
┌─────────────────────────────────────────────────────────────┐
│                      API Layer (FastAPI)                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  /predict  │  /analyze  │  /health  │  /model-info  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────┐
│                  Inference Pipeline Layer                    │
│  ┌───────────┐  ┌──────────────┐  ┌──────────────────┐    │
│  │ Image     │→ │ Validation & │→ │  Preprocessing   │    │
│  │ Upload    │  │ OOD Check    │  │  & Transform     │    │
│  └───────────┘  └──────────────┘  └────────┬─────────┘    │
│                                              │               │
│  ┌──────────────────────────────────────────┘               │
│  │                                                           │
│  ├→ ┌─────────────────┐ → ┌──────────────────────────┐    │
│     │  Model Inference │   │  Post-processing         │    │
│     │  EfficientNetV2  │   │  - Softmax               │    │
│     └─────────────────┘   │  - Confidence threshold   │    │
│                            │  - Grad-CAM (optional)    │    │
│                            └──────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────┐
│                     ML Training Layer                        │
│  ┌────────────┐  ┌──────────┐  ┌───────────┐              │
│  │ Data       │→ │ Training │→ │ Evaluation│→ Artifacts   │
│  │ Pipeline   │  │ Pipeline │  │ & Analysis│              │
│  └────────────┘  └──────────┘  └───────────┘              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Module Dependencies

```
ml/
├── data/              (no dependencies)
├── preprocessing/     (depends: data)
├── models/            (depends: preprocessing)
├── training/          (depends: models, preprocessing, data)
├── evaluation/        (depends: training, models)
├── explainability/    (depends: models, evaluation)
├── robustness/        (depends: models, evaluation)
└── ood/               (depends: models, preprocessing)

apps/
├── api/               (depends: ml/*)
└── web/               (depends: api)
```

---

## 4. Data Pipeline Specification

### 4.1 Data Ingestion

**Module**: `ml/data/ingestion/`

**Responsibilities**:

- Download dataset from Kaggle (manual or API)
- Extract and organize raw files
- Generate initial data manifest

**Output**:

```python
DataManifest = {
    "image_path": str,
    "label": str,
    "class_id": int,
    "file_size_bytes": int,
    "image_width": int,
    "image_height": int,
    "image_format": str,
    "is_valid": bool,
    "error_message": Optional[str]
}
```

### 4.2 Data Validation

**Module**: `ml/data/validation/`

**Validation Checks**:

| Check            | Description                    | Action on Failure    |
| ---------------- | ------------------------------ | -------------------- |
| File Readability | Can PIL/cv2 open the file?     | Mark invalid         |
| Format Support   | Is it JPEG/PNG?                | Mark invalid         |
| Corruption       | Is file corrupted?             | Mark invalid         |
| Size Threshold   | Is dimension ≥ 100×100?        | Mark invalid or warn |
| Aspect Ratio     | Is ratio reasonable (0.5-2.0)? | Warn only            |
| Label Validity   | Is label in expected classes?  | Mark invalid         |

**Output**: Updated manifest with `is_valid` flags and `validation_notes`

### 4.3 Duplicate Detection

**Module**: `ml/data/validation/duplicate_detector.py`

**Strategies**:

1. **Exact Duplicates**: MD5/SHA256 file hash
2. **Near Duplicates**: Perceptual hash (pHash or dHash)
   - Hamming distance threshold: < 5 bits difference

**Output**: List of duplicate groups, recommend which to keep (first occurrence)

### 4.4 Exploratory Data Analysis

**Module**: `notebooks/01_eda.ipynb` + `ml/data/analysis.py`

**Analyses**:

1. Class distribution (counts, percentages)
2. Image dimension distribution
3. File size distribution
4. Image quality metrics (blur, brightness, contrast)
5. Sample visualization grid (5×5 per class)

**Outputs**:

- `artifacts/figures/class_distribution.png`
- `artifacts/figures/sample_grid_{class}.png`
- `artifacts/reports/eda_summary.json`

### 4.5 Data Splitting

**Module**: `ml/data/splitting/`

**Strategy**: Stratified random split

- **Train**: 70%
- **Validation**: 15%
- **Test**: 15%

**Implementation**:

```python
from sklearn.model_selection import train_test_split

# Stratify by class to maintain proportions
train, temp = train_test_split(data, test_size=0.3, stratify=labels, random_state=42)
val, test = train_test_split(temp, test_size=0.5, stratify=temp_labels, random_state=42)
```

**Constraints**:

- Split **before** duplicate removal OR remove duplicates **before** split
- Use fixed random seed for reproducibility
- Verify class balance in each split

**Output**:

- `data/splits/train_manifest.csv`
- `data/splits/val_manifest.csv`
- `data/splits/test_manifest.csv`

### 4.6 Data Preprocessing

**Module**: `ml/preprocessing/`

#### 4.6.1 Base Transform (All Splits)

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet statistics
        std=[0.229, 0.224, 0.225]
    )
])
```

#### 4.6.2 Training Augmentation

```python
train_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

**Augmentation Guidelines**:

- Preserve disease characteristics
- No extreme distortions that would confuse diagnosis
- Test augmented samples visually

---

## 5. Model Specification

### 5.1 Baseline Model

**Architecture**: Simple CNN or MobileNetV3-Small

**Purpose**:

- Establish reproducible baseline
- Verify data pipeline
- Quick iteration benchmark

**Expected Performance**: 60-75% accuracy

### 5.2 Main Model: EfficientNetV2-B0

**Architecture Selection Rationale**:

- **Accuracy**: SOTA for image classification
- **Efficiency**: Optimized for speed and size
- **Transfer Learning**: Strong ImageNet pre-training
- **Inference Speed**: Suitable for edge deployment

**Model Configuration**:

```python
import timm

model = timm.create_model(
    'tf_efficientnetv2_b0',
    pretrained=True,
    num_classes=4
)
```

**Architecture**:

```
Input (224×224×3)
    ↓
EfficientNetV2-B0 Backbone (pretrained)
    ↓
Global Average Pooling
    ↓
Dropout (0.2)
    ↓
Linear (num_features → 4)
    ↓
Output (4 class logits)
```

### 5.3 Training Strategy

#### Phase 1: Transfer Learning

- **Freeze**: All backbone layers
- **Train**: Only classification head
- **Epochs**: 10-15
- **Learning Rate**: 1e-3
- **Optimizer**: AdamW
- **Loss**: CrossEntropyLoss

#### Phase 2: Fine-Tuning

- **Unfreeze**: Last 2-3 blocks of backbone
- **Train**: Backbone (selected layers) + head
- **Epochs**: 10-20
- **Learning Rate**: 1e-4 to 1e-5
- **Optimizer**: AdamW
- **Loss**: CrossEntropyLoss

**Hyperparameters**:

```python
config = {
    "batch_size": 32,
    "initial_lr": 1e-3,
    "fine_tune_lr": 1e-4,
    "weight_decay": 1e-4,
    "momentum": 0.9,
    "dropout": 0.2,
    "epochs_phase1": 15,
    "epochs_phase2": 20,
}
```

**Callbacks**:

- Early stopping (patience=5, monitor='val_loss')
- Learning rate scheduler (ReduceLROnPlateau)
- Model checkpoint (save best validation F1)
- TensorBoard logging

### 5.4 Loss Function

**Primary**: CrossEntropyLoss

**Alternative (if class imbalance severe)**: Weighted CrossEntropyLoss

```python
class_weights = compute_class_weight('balanced', classes=classes, y=train_labels)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
```

### 5.5 Model Comparison

Compare these architectures:

| Model             | Parameters | Size (MB) | Expected Accuracy | Notes         |
| ----------------- | ---------- | --------- | ----------------- | ------------- |
| MobileNetV3-Small | 2.5M       | 10        | 75-82%            | Baseline      |
| MobileNetV3-Large | 5.4M       | 21        | 80-85%            | Alternative   |
| EfficientNet-B0   | 5.3M       | 20        | 82-88%            | Comparison    |
| EfficientNetV2-B0 | 7.1M       | 27        | 85-90%            | Primary model |

---

## 6. Evaluation Framework

### 6.1 Metrics

**Primary Metric**: **Macro F1 Score**

- Treats all classes equally
- Sensitive to class imbalance

**Secondary Metrics**:

- Accuracy
- Per-class Precision, Recall, F1
- Confusion Matrix
- ROC-AUC (one-vs-rest)

**Metric Computation**:

```python
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

metrics = {
    "accuracy": accuracy_score(y_true, y_pred),
    "macro_f1": f1_score(y_true, y_pred, average='macro'),
    "per_class": classification_report(y_true, y_pred, output_dict=True),
    "confusion_matrix": confusion_matrix(y_true, y_pred)
}
```

### 6.2 Error Analysis

**Confidence Buckets**:

- 0.0 - 0.2: Very low confidence
- 0.2 - 0.4: Low confidence
- 0.4 - 0.6: Medium confidence
- 0.6 - 0.8: High confidence
- 0.8 - 1.0: Very high confidence

**Analysis Questions**:

1. What percentage of high-confidence predictions are wrong?
2. Are low-confidence predictions uniformly distributed across classes?
3. Which class pairs are most confused?
4. Do errors correlate with image quality issues?

**Visualizations**:

- Confidence histogram
- Confidence vs. correctness scatter
- Per-class confusion heatmap
- Error sample gallery

### 6.3 Explainability: Grad-CAM

**Module**: `ml/explainability/gradcam.py`

**Implementation**: Use `pytorch-grad-cam` library

**Target Layer**: Last convolutional layer of EfficientNetV2

**Process**:

1. Forward pass through model
2. Compute gradients of predicted class w.r.t. target layer
3. Weight activations by gradients
4. Apply ReLU and upsample to input size
5. Overlay heatmap on original image

**Validation**:

- Manually inspect 20-30 samples per class
- Verify heatmap focuses on leaf tissue, not background
- Document failure modes

### 6.4 Robustness Testing

**Module**: `ml/robustness/perturbations.py`

**Perturbations**:

| Perturbation     | Parameters      | Purpose                  |
| ---------------- | --------------- | ------------------------ |
| Brightness       | ±30%            | Lighting variation       |
| Contrast         | ±30%            | Camera/display variation |
| Gaussian Blur    | σ=2.0           | Out-of-focus             |
| Gaussian Noise   | σ=0.05          | Sensor noise             |
| JPEG Compression | quality=30      | Messaging/social media   |
| Random Crop      | 20% border crop | Framing variation        |
| Rotation         | ±45°            | Camera angle             |

**Evaluation**:

- Apply each perturbation to test set
- Measure accuracy and macro F1
- Compare to clean test performance
- Identify most vulnerable perturbations

### 6.5 OOD Detection

**Module**: `ml/ood/detector.py`

**Test Inputs**:

- Maize leaves
- Banana leaves
- Random objects (cars, buildings)
- Soil/ground images
- Blank/uniform images

**Detection Strategy**:

- Maximum softmax probability < threshold (e.g., 0.5)
- Entropy > threshold
- Distance from training distribution (optional)

**Metrics**:

- OOD detection rate (true positive rate)
- In-distribution acceptance rate (true negative rate)
- AUROC for OOD detection

---

## 7. API Specification

### 7.1 Technology

- **Framework**: FastAPI 0.100+
- **Server**: Uvicorn
- **Validation**: Pydantic
- **Documentation**: Auto-generated OpenAPI (Swagger)

### 7.2 Endpoints

#### 7.2.1 Health Check

```
GET /health
```

**Response**:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

#### 7.2.2 Simple Prediction

```
POST /predict
```

**Request**:

- Content-Type: multipart/form-data
- Body: `file` (image file)

**Response**:

```json
{
  "prediction": "Leaf Rust",
  "class_id": 2,
  "confidence": 0.94,
  "accepted": true,
  "processing_time_ms": 145
}
```

#### 7.2.3 Detailed Analysis

```
POST /analyze
```

**Request**:

- Content-Type: multipart/form-data
- Body: `file` (image file), `include_gradcam` (optional bool)

**Response**:

```json
{
  "prediction": "Leaf Rust",
  "class_id": 2,
  "confidence": 0.94,
  "accepted": true,
  "class_probabilities": {
    "Healthy": 0.02,
    "Cercospora": 0.03,
    "Leaf Rust": 0.94,
    "Phoma": 0.01
  },
  "gradcam_image": "base64_encoded_image",
  "processing_time_ms": 187,
  "model_info": {
    "name": "efficientnetv2_b0",
    "version": "1.0"
  }
}
```

#### 7.2.4 Model Information

```
GET /model-info
```

**Response**:

```json
{
  "model_name": "EfficientNetV2-B0",
  "version": "1.0.0",
  "classes": ["Healthy", "Cercospora", "Leaf Rust", "Phoma"],
  "input_size": [224, 224],
  "training_date": "2024-XX-XX",
  "metrics": {
    "accuracy": 0.89,
    "macro_f1": 0.87
  }
}
```

### 7.3 Error Handling

**Status Codes**:

- 200: Success
- 400: Invalid input (e.g., not an image)
- 413: File too large
- 500: Internal server error

**Error Response**:

```json
{
  "error": "Invalid file format",
  "detail": "Only JPEG and PNG images are supported",
  "status_code": 400
}
```

---

## 8. UI Specification

### 8.1 Technology

- **Framework**: Streamlit
- **Layout**: Single-page app
- **Styling**: Streamlit native + custom CSS

### 8.2 User Flow

```
┌─────────────────────┐
│   Landing Page      │
│  - Title            │
│  - Description      │
│  - File uploader    │
└──────────┬──────────┘
           │
           ↓ Upload image
┌─────────────────────┐
│   Image Preview     │
│  - Show uploaded    │
│  - Display size     │
└──────────┬──────────┘
           │
           ↓ Click "Analyze"
┌─────────────────────┐
│   Processing        │
│  - Spinner          │
└──────────┬──────────┘
           │
           ↓ Results ready
┌─────────────────────┐
│   Results Display   │
│  - Prediction       │
│  - Confidence bar   │
│  - Class probs      │
│  - Grad-CAM         │
│  - Recommendations  │
└─────────────────────┘
```

### 8.3 Components

**1. Header**

- App title and logo
- Brief description

**2. File Upload**

- Drag-and-drop or browse
- Accept: JPEG, PNG
- Max size: 10MB

**3. Preview**

- Display uploaded image
- Show dimensions and size

**4. Results Panel**

- Predicted class (large text)
- Confidence score (progress bar)
- Class probability chart (bar chart)
- Grad-CAM overlay (side-by-side)

**5. Recommendations** (Optional)

- Disease-specific treatment suggestions
- Confidence-based messaging

**6. Footer**

- About the project
- Dataset attribution
- Version info

---

## 9. Infrastructure & Deployment

### 9.1 Development Environment

**Setup**:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

**Dependencies Management**:

- Use `pyproject.toml` for dependency specification
- Pin major versions in production
- Use `requirements.txt` for exact reproducibility

### 9.2 Docker Containerization

**Dockerfile** (example):

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run API
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 9.3 CI/CD Pipeline

**GitHub Actions Workflow** (`.github/workflows/ci.yml`):

- Trigger: Push to main, PR to main
- Jobs:
  - Lint (black, flake8)
  - Type check (mypy)
  - Test (pytest)
  - Build Docker image
  - (Optional) Deploy to staging

### 9.4 Model Versioning

- Store model checkpoints in `artifacts/checkpoints/`
- Naming: `model_v{version}_{date}_{metric}.pth`
- Track model metadata in `artifacts/model_registry.json`

---

## 10. Quality Assurance

### 10.1 Testing Strategy

**Test Coverage Goals**:

- Unit tests: 80%+
- Integration tests: Critical paths covered
- API tests: All endpoints

**Test Types**:

1. **Unit Tests**: Individual functions and classes
2. **Integration Tests**: Data pipeline, training loop
3. **API Tests**: Endpoint responses and error handling
4. **Model Tests**: Inference correctness, output shapes

### 10.2 Code Quality

**Tools**:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pylint**: Additional linting

**Pre-commit Hooks**:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
  - repo: https://github.com/PyCQA/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

### 10.3 Documentation Standards

- All public functions: Docstrings
- Complex algorithms: Inline comments
- API endpoints: OpenAPI docs
- Architecture decisions: ADR documents

---

## 11. Timeline & Milestones

### Phase 1: Project Setup (Days 0-1) ✅

- [x] Repository structure
- [x] Documentation framework
- [x] Team workflow setup
- [ ] Development environment
- [ ] CI/CD pipeline

### Phase 2: Data Engineering (Days 1-2)

- [ ] Dataset download
- [ ] Data validation pipeline
- [ ] Duplicate detection
- [ ] EDA notebooks
- [ ] Data splitting
- [ ] Manifests generation

### Phase 3: Baseline & Preprocessing (Day 2)

- [ ] Preprocessing pipeline
- [ ] Data loaders
- [ ] Baseline model training
- [ ] Initial evaluation

### Phase 4: Main Model Training (Days 3-4)

- [ ] EfficientNetV2 implementation
- [ ] Transfer learning phase
- [ ] Fine-tuning phase
- [ ] Hyperparameter tuning
- [ ] Training curves

### Phase 5: Evaluation & Analysis (Day 4-5)

- [ ] Comprehensive metrics
- [ ] Confusion matrix analysis
- [ ] Error analysis
- [ ] Confidence analysis
- [ ] Grad-CAM implementation
- [ ] Robustness testing

### Phase 6: OOD & Model Selection (Day 5-6)

- [ ] OOD detection implementation
- [ ] Model comparison experiments
- [ ] Final model selection
- [ ] Model optimization

### Phase 7: Deployment (Day 6-7)

- [ ] FastAPI implementation
- [ ] API testing
- [ ] Streamlit UI
- [ ] Docker containerization
- [ ] Documentation finalization

### Phase 8: Polish & Presentation (Day 7)

- [ ] Final testing
- [ ] Demo preparation
- [ ] Technical report
- [ ] README polish
- [ ] Presentation materials

---

## Appendices

### A. Kaggle Dataset Structure

Expected structure (verify):

```
raw/
├── Healthy/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── Cercospora/
│   └── ...
├── Leaf_Rust/
│   └── ...
└── Phoma/
    └── ...
```

### B. Configuration File Example

`configs/training_config.yaml`:

```yaml
model:
  name: "efficientnetv2_b0"
  pretrained: true
  num_classes: 4
  dropout: 0.2

training:
  batch_size: 32
  epochs_phase1: 15
  epochs_phase2: 20
  learning_rate: 0.001
  weight_decay: 0.0001

data:
  image_size: 224
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  random_seed: 42
```

### C. Useful References

- [EfficientNetV2 Paper](https://arxiv.org/abs/2104.00298)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)
- [PyTorch Image Models (timm) Docs](https://timm.fast.ai/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**Document Maintainer**: Project Lead  
**Review Schedule**: Updated after each major phase  
**Feedback**: Open GitHub issue with label `documentation`
