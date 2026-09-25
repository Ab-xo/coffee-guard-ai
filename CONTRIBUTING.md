# 🤝 Contributing to CoffeeGuard AI

Thank you for contributing to **CoffeeGuard AI**! This guide will help you understand our team workflow, coding standards, and collaboration practices.

---

## 📋 Table of Contents

- [Team Structure](#team-structure)
- [Development Workflow](#development-workflow)
- [Git Branching Strategy](#git-branching-strategy)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Code Standards](#code-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)
- [Communication](#communication)

---

## 👥 Team Structure

Our team consists of **6 members** working collaboratively. Here's how we organize:

### Roles & Responsibilities

1. **Data Engineering Lead**
   - Dataset acquisition and validation
   - Data quality pipeline
   - Split strategy implementation

2. **ML Model Lead**
   - Model architecture selection
   - Training pipeline development
   - Hyperparameter tuning

3. **Evaluation & Analysis Lead**
   - Metrics implementation
   - Error analysis
   - Confidence analysis

4. **Explainability & Robustness Lead**
   - Grad-CAM implementation
   - Perturbation testing
   - OOD detection

5. **Backend/API Lead**
   - FastAPI service development
   - Model serving infrastructure
   - API documentation

6. **Frontend/UI Lead**
   - Streamlit application
   - User experience design
   - Visualization components

> **Note**: Roles are flexible. Team members can contribute to multiple areas.

---

## 🔄 Development Workflow

### 1. Pick or Create an Issue

Before starting work:

1. Check existing [GitHub Issues](../../issues)
2. Pick an unassigned issue or create a new one
3. Assign yourself to the issue
4. Add appropriate labels (e.g., `enhancement`, `bug`, `documentation`)

### 2. Create a Feature Branch

```bash
# Update your local main branch
git checkout main
git pull origin main

# Create a feature branch
git checkout -b feature/your-feature-name

# OR for bug fixes
git checkout -b bugfix/issue-description

# OR for experiments
git checkout -b experiment/model-comparison
```

### 3. Make Your Changes

- Write clean, readable code
- Follow our [Code Standards](#code-standards)
- Add tests for new functionality
- Update documentation as needed

### 4. Commit Your Changes

```bash
# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "feat: add data validation pipeline"
```

See [Commit Guidelines](#commit-guidelines) for commit message format.

### 5. Push to Remote

```bash
git push origin feature/your-feature-name
```

### 6. Open a Pull Request

1. Go to GitHub repository
2. Click "New Pull Request"
3. Select your branch
4. Fill out the PR template
5. Request reviews from team members
6. Link related issues

### 7. Code Review

- Address reviewer feedback
- Make requested changes
- Push updates to your branch
- Request re-review if needed

### 8. Merge

Once approved:

- Squash and merge (preferred) or merge commit
- Delete the feature branch
- Close related issues

---

## 🌳 Git Branching Strategy

We follow a **simplified GitFlow** strategy:

### Branch Types

```
main
  ├── develop
  │     ├── feature/data-ingestion
  │     ├── feature/baseline-model
  │     ├── bugfix/image-loader
  │     └── experiment/efficientnet-comparison
```

### Branch Naming Conventions

| Type          | Pattern                     | Example                                |
| ------------- | --------------------------- | -------------------------------------- |
| Feature       | `feature/short-description` | `feature/gradcam-visualization`        |
| Bug Fix       | `bugfix/issue-description`  | `bugfix/duplicate-detection`           |
| Experiment    | `experiment/description`    | `experiment/mobilenet-vs-efficientnet` |
| Documentation | `docs/description`          | `docs/api-documentation`               |
| Refactor      | `refactor/description`      | `refactor/data-pipeline`               |

### Branch Lifecycle

1. **main**: Production-ready code
   - Protected branch
   - All tests must pass
   - Requires PR approval

2. **develop**: Integration branch (optional for our project size)
   - Latest development code
   - Features merge here first

3. **feature/\***: New features
   - Branch from `main` or `develop`
   - Merge back via PR
   - Delete after merge

4. **bugfix/\***: Bug fixes
   - Branch from `main`
   - Merge back via PR
   - Delete after merge

---

## 💬 Commit Guidelines

We follow **Conventional Commits** format for clear commit history.

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

| Type       | Description             | Example                                          |
| ---------- | ----------------------- | ------------------------------------------------ |
| `feat`     | New feature             | `feat(data): add duplicate detection`            |
| `fix`      | Bug fix                 | `fix(model): correct image normalization`        |
| `docs`     | Documentation           | `docs(readme): update installation steps`        |
| `style`    | Code style (formatting) | `style(preprocessing): format with black`        |
| `refactor` | Code refactoring        | `refactor(training): simplify training loop`     |
| `test`     | Adding tests            | `test(validation): add unit tests for validator` |
| `chore`    | Maintenance             | `chore(deps): update requirements`               |
| `perf`     | Performance improvement | `perf(inference): optimize image preprocessing`  |
| `exp`      | Experiment              | `exp(models): test efficientnet-b3`              |

### Examples

```bash
# Good commits
git commit -m "feat(data): implement stratified splitting"
git commit -m "fix(api): handle corrupted image files"
git commit -m "docs(contributing): add commit guidelines"
git commit -m "test(ood): add OOD detection tests"

# Avoid
git commit -m "fixed stuff"
git commit -m "WIP"
git commit -m "update"
```

### Multi-line Commits

```bash
git commit -m "feat(training): add early stopping callback

- Monitor validation loss
- Save best checkpoint
- Configurable patience parameter

Closes #42"
```

---

## 🔍 Pull Request Process

### PR Title Format

Follow the same format as commits:

```
<type>(<scope>): <description>
```

Example: `feat(gradcam): implement grad-cam visualization`

### PR Template

When creating a PR, include:

```markdown
## Description

Brief description of changes

## Type of Change

- [ ] New feature
- [ ] Bug fix
- [ ] Documentation
- [ ] Refactoring
- [ ] Experiment

## Changes Made

- Item 1
- Item 2
- Item 3

## Testing

- [ ] Unit tests added/updated
- [ ] All tests passing
- [ ] Manual testing completed

## Checklist

- [ ] Code follows project style guidelines
- [ ] Documentation updated
- [ ] No breaking changes
- [ ] Related issues linked

## Screenshots (if applicable)

## Related Issues

Closes #XX
```

### Review Guidelines

**For Authors:**

- Keep PRs focused and small when possible
- Respond to feedback promptly
- Update documentation
- Ensure tests pass

**For Reviewers:**

- Review within 24 hours when possible
- Be constructive and specific
- Test the changes locally if needed
- Approve when satisfied

### Review Checklist

- [ ] Code is clean and readable
- [ ] Follows project conventions
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No obvious bugs or issues
- [ ] Commit messages are clear

---

## 📏 Code Standards

### Python Style

We follow **PEP 8** with some modifications:

```python
# Good
def preprocess_image(image_path: str, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Preprocess a single image for model inference.

    Args:
        image_path: Path to the input image
        target_size: Target dimensions (height, width)

    Returns:
        Preprocessed image array
    """
    image = Image.open(image_path)
    image = image.resize(target_size)
    return np.array(image)
```

### Formatting Tools

```bash
# Use Black for formatting
black ml/ apps/ tests/

# Use isort for import sorting
isort ml/ apps/ tests/

# Use flake8 for linting
flake8 ml/ apps/ tests/
```

### Code Organization

```python
# Standard library imports
import os
from pathlib import Path

# Third-party imports
import numpy as np
import torch
from PIL import Image

# Local imports
from ml.preprocessing.transforms import normalize_image
from ml.models.efficientnet import EfficientNetClassifier
```

### Type Hints

Always use type hints:

```python
from typing import List, Tuple, Optional

def split_dataset(
    data: List[str],
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)
) -> Tuple[List[str], List[str], List[str]]:
    """Split dataset into train, val, test sets."""
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate classification metrics.

    Args:
        y_true: Ground truth labels (n_samples,)
        y_pred: Predicted labels (n_samples,)

    Returns:
        Dictionary containing accuracy, precision, recall, and F1 scores

    Raises:
        ValueError: If arrays have different lengths

    Example:
        >>> metrics = calculate_metrics([0, 1, 1], [0, 0, 1])
        >>> print(metrics['accuracy'])
        0.667
    """
    ...
```

---

## 🧪 Testing Requirements

### Test Structure

```
tests/
├── unit/                    # Fast, isolated tests
│   ├── test_data_validation.py
│   ├── test_preprocessing.py
│   └── test_models.py
├── integration/             # Component integration tests
│   ├── test_training_pipeline.py
│   └── test_api_endpoints.py
└── fixtures/               # Shared test data
    ├── sample_images/
    └── test_configs/
```

### Writing Tests

```python
import pytest
from ml.data.validation import validate_image

def test_validate_valid_image():
    """Test validation passes for valid images."""
    result = validate_image("tests/fixtures/sample_images/healthy_001.jpg")
    assert result.is_valid is True
    assert result.error_message is None

def test_validate_corrupted_image():
    """Test validation fails for corrupted images."""
    result = validate_image("tests/fixtures/sample_images/corrupted.jpg")
    assert result.is_valid is False
    assert "corrupted" in result.error_message.lower()
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_data_validation.py

# Run with coverage
pytest --cov=ml --cov-report=html

# Run only fast tests
pytest -m "not slow"
```

### Test Requirements

- **Unit tests**: Required for new functions and classes
- **Integration tests**: Required for pipelines and workflows
- **Coverage**: Aim for 80%+ coverage
- **Speed**: Unit tests should run in milliseconds

---

## 📖 Documentation

### What to Document

1. **Code Documentation**
   - All public functions and classes
   - Complex logic and algorithms
   - Type hints for all parameters

2. **API Documentation**
   - All endpoints
   - Request/response schemas
   - Example usage

3. **Architecture Docs**
   - System design decisions
   - Data flow diagrams
   - Component interactions

4. **Experiment Logs**
   - Model configurations
   - Training results
   - Lessons learned

### Documentation Locations

```
docs/
├── architecture/
│   ├── data_pipeline.md
│   ├── model_architecture.md
│   └── deployment_strategy.md
├── decisions/
│   ├── 001-efficientnet-selection.md
│   ├── 002-data-augmentation.md
│   └── 003-api-framework.md
└── experiments/
    ├── baseline_results.md
    ├── transfer_learning_comparison.md
    └── robustness_analysis.md
```

### Updating Documentation

- Update relevant docs in the same PR as code changes
- Keep documentation close to the code when possible
- Use diagrams for complex concepts
- Include examples and usage patterns

---

## 💬 Communication

### Channels

1. **GitHub Issues**: Feature requests, bug reports, tasks
2. **Pull Requests**: Code discussions, reviews
3. **GitHub Discussions**: General questions, ideas
4. **[Your Team Channel]**: Real-time communication

### Best Practices

- **Be Respectful**: Professional and constructive communication
- **Be Clear**: Provide context and details
- **Be Responsive**: Reply to mentions within 24 hours
- **Ask Questions**: No question is too small
- **Share Knowledge**: Document learnings and solutions

### Issue Creation

```markdown
## Issue Template: Feature Request

**Title**: [Feature] Add confidence threshold configuration

**Description**
Clear description of the feature

**Why is this needed?**
Business or technical justification

**Proposed Solution**
How you think it should work

**Acceptance Criteria**

- [ ] Criterion 1
- [ ] Criterion 2

**Additional Context**
Screenshots, examples, references
```

---

## 🚀 Quick Reference

### Daily Workflow

```bash
# Start of day
git checkout main
git pull origin main
git checkout -b feature/my-feature

# During development
git add .
git commit -m "feat(scope): description"
git push origin feature/my-feature

# After PR approval
# Delete local branch
git checkout main
git branch -d feature/my-feature
```

### Common Commands

```bash
# Check status
git status

# View changes
git diff

# Update from main
git checkout main
git pull origin main
git checkout feature/my-feature
git merge main

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Stash changes
git stash
git stash pop
```

---

## 📝 Checklist for Contributors

Before submitting a PR:

- [ ] Code follows style guidelines
- [ ] Type hints added
- [ ] Docstrings added
- [ ] Unit tests added
- [ ] All tests passing locally
- [ ] Documentation updated
- [ ] Commit messages follow convention
- [ ] PR template filled out
- [ ] Related issues linked
- [ ] No sensitive data committed

---

## ❓ Need Help?

- Check existing documentation
- Search closed issues
- Ask in team channel
- Create a new issue with the `question` label
- Reach out to project maintainers

---

## 🎉 Thank You!

Your contributions make CoffeeGuard AI better! We appreciate your time and effort in building this project together.

---

<div align="center">
  <strong>Happy Contributing! 🚀</strong>
</div>
