# Ethiopian Coffee Leaf Disease Intelligence — CoffeeGuard AI

## 1. Project Overview

**Project type:** Computer Vision + Deep Learning + AI Engineering  
**Target implementation:** 4–7 days  
**Primary task:** Multi-class coffee leaf disease classification  
**Core model:** EfficientNetV2-B0 with transfer learning and fine-tuning  
**Deployment:** FastAPI + Streamlit  
**Advanced components:** Grad-CAM, confidence analysis, robustness testing, OOD rejection, model comparison

### Problem

Coffee leaf diseases can be difficult to identify quickly from photographs, especially when images vary in lighting, blur, background, framing, and disease severity. The project develops an AI system that analyzes a coffee-leaf photograph and predicts its disease condition while also exposing confidence, explanation, robustness, and rejection behavior.

### Dataset

**Ethiopian Coffee Leaf Disease Dataset — Kaggle**  
https://www.kaggle.com/datasets/biniyamyoseph/ethiopian-coffee-leaf-disease/data

The supplied project documentation identifies four classes:

- Healthy
- Cercospora
- Leaf Rust
- Phoma

> **Verification note:** Before final submission, verify the live Kaggle dataset card, current image counts, file contents, and licensing terms.

---

# 2. Objectives

## General Objective

Develop an explainable and robust deep-learning system capable of classifying Ethiopian coffee leaf images into disease categories and exposing the trained model through a practical API and web interface.

## Specific Objectives

1. Build a reproducible image ingestion and validation pipeline.
2. Explore class distribution and image quality.
3. Detect corrupted, invalid, duplicate, and unusable images.
4. Create a leakage-safe stratified train/validation/test split.
5. Establish a baseline model.
6. Fine-tune EfficientNetV2-B0 using transfer learning.
7. Evaluate accuracy, precision, recall, macro F1, and confusion matrix.
8. Analyze confidence and high-confidence errors.
9. Use Grad-CAM for visual explanations.
10. Test robustness against realistic image perturbations.
11. Add out-of-distribution rejection for irrelevant inputs.
12. Compare lightweight candidate models.
13. Serve the final model with FastAPI.
14. Build a Streamlit demonstration interface.

---

# 3. End-to-End Pipeline

```text
COFFEE LEAF IMAGE
       ↓
Data Ingestion
       ↓
Data Validation
(corrupt / invalid / duplicates)
       ↓
EDA & Quality Analysis
       ↓
Stratified 70/15/15 Split
       ↓
RGB → Resize 224×224 → Normalize
       ↓
Training Augmentation
       ↓
Baseline Model
       ↓
EfficientNetV2-B0 Transfer Learning
       ↓
Fine-Tuning
       ↓
Evaluation
       ├── Accuracy
       ├── Macro Precision/Recall/F1
       ├── Confusion Matrix
       └── Per-Class Metrics
       ↓
Error & Confidence Analysis
       ↓
Grad-CAM Explainability
       ↓
Robustness Testing
       ├── Brightness
       ├── Contrast
       ├── Blur
       ├── Noise
       ├── Compression
       └── Occlusion/Crop
       ↓
OOD / Input Gatekeeper
       ↓
FastAPI Inference
       ↓
Streamlit Application
```

---

# 4. Data Preprocessing

## 4.1 Ingestion

Create a deterministic loader that records image path, label, dimensions, format, file size, and validity.

## 4.2 Validation

Check for corrupted/unreadable files, unsupported formats, unusual dimensions, very small images, empty files, and invalid labels.

## 4.3 Duplicate Detection

Use file hashes for exact duplicates and perceptual hashes for near-duplicates. Perform this before splitting the dataset to reduce train/test leakage.

## 4.4 EDA

Analyze:

- class counts and imbalance;
- image dimensions and aspect ratios;
- brightness and contrast;
- blur/sharpness;
- background complexity;
- disease visibility.

Outputs should include a class-distribution chart, sample grid, dimension statistics, quality report, and duplicate report.

## 4.5 Split

Use a stratified:

- **70% training**
- **15% validation**
- **15% test**

Split before augmentation. Validation and test images must not receive training augmentation.

## 4.6 Image Processing

```text
Image
 ↓
RGB conversion
 ↓
Resize 224×224
 ↓
Tensor conversion
 ↓
Model normalization
```

## 4.7 Training Augmentation

Apply only to training data:

- horizontal flip;
- small rotation;
- random crop;
- zoom;
- brightness/contrast variation;
- mild blur.

Avoid transformations that destroy disease morphology.

---

# 5. Modeling

## Baseline

Start with a small CNN or MobileNetV3 to establish a reproducible reference.

## Main Model

Use ImageNet-pretrained **EfficientNetV2-B0** and replace its classification head with a four-class output:

```text
Input Image
   ↓
EfficientNetV2-B0
   ↓
Feature Representation
   ↓
Classification Head
   ↓
Healthy / Cercospora / Leaf Rust / Phoma
```

## Transfer Learning

Initially freeze most of the backbone and train the classification head.

## Fine-Tuning

Unfreeze selected upper layers, lower the learning rate, and continue training with early stopping based on validation performance.

---

# 6. Evaluation

Measure:

- Accuracy
- Precision
- Recall
- Macro F1
- Per-class F1
- Confusion matrix
- Training/validation curves

Macro F1 should be emphasized because it prevents performance on a dominant class from hiding poor performance on another class.

---

# 7. Error and Confidence Analysis

Create separate analyses for:

- high-confidence incorrect predictions;
- low-confidence predictions;
- blurred images;
- dark/overexposed images;
- background-heavy images;
- visually similar disease classes.

Analyze confidence buckets such as 0.0–0.2, 0.2–0.4, 0.4–0.6, 0.6–0.8, and 0.8–1.0, comparing confidence with actual correctness.

---

# 8. Explainable AI — Grad-CAM

Generate Grad-CAM visualizations for correct and incorrect predictions.

The objective is to determine whether the model focuses on disease-relevant leaf regions rather than backgrounds or unrelated image features.

```text
Original Leaf
     +
Grad-CAM Heatmap
     ↓
Disease-relevant visual region
```

---

# 9. Robustness Evaluation

Test the trained model under controlled perturbations:

| Perturbation | Purpose |
|---|---|
| Brightness | Field lighting variation |
| Contrast | Camera/environment variation |
| Blur | Focus/motion degradation |
| Noise | Sensor/environment noise |
| JPEG compression | Messaging/social-media compression |
| Occlusion | Partial leaf visibility |
| Crop/scale | Different camera framing |

Compare accuracy, macro F1, and confidence against the clean test set.

---

# 10. Out-of-Distribution Detection

Test unrelated images such as maize leaves, banana leaves, grass, soil, and ordinary objects. The system should reject clearly irrelevant inputs rather than forcing them into one of the four disease classes.

Example:

```json
{
  "predicted_class": null,
  "confidence": 0.41,
  "status": "rejected",
  "reason": "low_confidence_or_ood"
}
```

---

# 11. Model Comparison

Compare:

| Model | Accuracy | Macro F1 | Parameters | Size | Inference Time |
|---|---:|---:|---:|---:|---:|
| MobileNetV3 | — | — | — | — | — |
| EfficientNet-B0 | — | — | — | — | — |
| EfficientNetV2-B0 | — | — | — | — | — |

Choose the deployment model using predictive performance **and** model size, latency, and robustness.

---

# 12. Production Inference

```text
User Image
 ↓
File Validation
 ↓
Quality/OOD Gate
 ↓
Preprocessing
 ↓
Model Inference
 ↓
Class Probabilities
 ↓
Confidence Threshold
 ↓
Prediction + Explanation
 ↓
JSON Response
```

Example:

```json
{
  "prediction": "Leaf Rust",
  "confidence": 0.94,
  "accepted": true,
  "model": "efficientnetv2-b0"
}
```

---

# 13. API

Use **FastAPI** with:

```text
GET  /health
POST /predict
POST /analyze
GET  /model-info
```

`/analyze` should return prediction, confidence, class probabilities, explanation/Grad-CAM, and rejection information when appropriate.

---

# 14. User Interface

Use **Streamlit**:

```text
Upload Leaf Image
       ↓
Preview
       ↓
Analyze
       ↓
Prediction + Confidence
       ↓
Class Probabilities
       ↓
Grad-CAM
```

---

# 15. Suggested Repository Structure

```text
coffee-guard-ai/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_quality.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_training.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_gradcam.ipynb
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── models/
│   ├── evaluation/
│   └── explainability/
├── api/
│   └── main.py
├── app/
│   └── streamlit_app.py
├── models/
├── tests/
├── requirements.txt
├── Dockerfile
└── README.md
```

---

# 16. 7-Day Implementation Plan

### Day 1 — Dataset & EDA

Download, validate, inspect class balance, detect duplicates, analyze quality, and create stratified splits.

### Day 2 — Preprocessing & Baseline

Implement transforms, augmentation, baseline training, and initial metrics.

### Day 3 — Transfer Learning

Train EfficientNetV2-B0 with a frozen backbone.

### Day 4 — Fine-Tuning & Evaluation

Fine-tune, generate metrics, confusion matrix, and training curves.

### Day 5 — Explainability & Robustness

Implement Grad-CAM, confidence analysis, error analysis, and perturbation testing.

### Day 6 — OOD & API

Implement the input gatekeeper and FastAPI inference service.

### Day 7 — UI & Documentation

Build Streamlit, finalize experiments, architecture diagrams, README, and demonstration.

---

# 17. Final Deliverables

- Reproducible data pipeline
- Clean dataset manifest
- EDA report
- Baseline model
- Fine-tuned EfficientNetV2-B0
- Training curves
- Confusion matrix
- Per-class metrics
- Error analysis
- Confidence analysis
- Grad-CAM visualizations
- Robustness evaluation
- OOD experiment
- Model comparison
- FastAPI service
- Streamlit UI
- Docker configuration
- README and technical report

---

# 18. Why This Is an AI Engineering Project

The project deliberately goes beyond a basic classifier:

```text
DATA
 ↓
DATA QUALITY
 ↓
PREPROCESSING
 ↓
BASELINE
 ↓
TRANSFER LEARNING
 ↓
FINE-TUNING
 ↓
EVALUATION
 ↓
ERROR ANALYSIS
 ↓
EXPLAINABILITY
 ↓
ROBUSTNESS
 ↓
OOD DETECTION
 ↓
MODEL SELECTION
 ↓
API
 ↓
APPLICATION
```

The result is a complete ML/AI engineering MVP combining model development, experimental evaluation, reliability analysis, and deployment.

---

# 19. Success Criteria

A successful implementation should demonstrate:

1. Reliable data preprocessing.
2. No obvious train/test leakage.
3. Strong macro-F1 across the four classes.
4. Transparent per-class performance.
5. Meaningful error analysis.
6. Disease-focused Grad-CAM explanations.
7. Measurable robustness under image perturbations.
8. Rejection of clearly irrelevant inputs.
9. Reproducible inference.
10. Working API and web demonstration.

---

# 20. Final Project Statement

**CoffeeGuard AI** is an Ethiopian-context computer-vision system that demonstrates the complete lifecycle of a modern deep-learning application:

> **Dataset → Data Engineering → Deep Learning → Evaluation → Explainability → Robustness → OOD Detection → API → Application**

The scope is intentionally designed so that a functional MVP can be developed in approximately **4–7 intensive days**, while retaining enough technical depth for an AI/Data Science final project, portfolio, or engineering demonstration.
