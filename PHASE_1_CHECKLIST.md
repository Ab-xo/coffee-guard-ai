# Phase 1 Setup Checklist

**CoffeeGuard AI - Project Initialization**

This checklist ensures all team members have a properly configured development environment and understand the project workflow.

---

## 📋 Repository Setup (Project Lead)

### Initial Configuration

- [x] Create repository structure
- [x] Add README.md with project overview
- [x] Add CONTRIBUTING.md with team workflow
- [x] Add LICENSE file
- [x] Create .gitignore for Python projects
- [ ] Create initial GitHub release (v0.1.0-alpha)

### Documentation

- [x] Create PROJECT_SPECIFICATION.md
- [x] Create CoffeeGuard_AI_Project_Structure.md
- [x] Create TEAM_ROLES.md
- [x] Create GIT_WORKFLOW.md
- [ ] Add CHANGELOG.md
- [ ] Create docs/architecture/ folder structure
- [ ] Add architecture diagrams (create later)

### GitHub Configuration

- [x] Add issue templates (feature, bug, task)
- [x] Add pull request template
- [ ] Set up branch protection rules for main
- [ ] Configure required PR reviews (minimum 1 approver)
- [ ] Create project board with columns: Backlog, To Do, In Progress, Review, Done
- [ ] Create milestone for each phase
- [ ] Add labels: enhancement, bug, documentation, good first issue, help wanted

### CI/CD Setup (Optional for Phase 1)

- [ ] Create .github/workflows/ci.yml for testing
- [ ] Create .github/workflows/lint.yml for code quality
- [ ] Configure automatic testing on PR

---

## 👥 Team Member Onboarding

### For Each Team Member

#### 1. Environment Setup

- [ ] Install Python 3.9+ ([Download](https://www.python.org/downloads/))
- [ ] Install Git ([Download](https://git-scm.com/downloads))
- [ ] Install VS Code or PyCharm (recommended)
- [ ] Create GitHub account (if not already)
- [ ] Get added as collaborator to repository

#### 2. Clone Repository

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/coffee-guard-ai.git
cd coffee-guard-ai

# Verify structure
ls -la  # or dir on Windows
```

#### 3. Git Configuration

```bash
# Set your name and email
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Verify configuration
git config --list

# Set default branch name
git config --global init.defaultBranch main
```

#### 4. Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Verify Python version
python --version  # Should be 3.9+
```

#### 5. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install project in development mode
pip install -e .

# If dependencies aren't set up yet, install common packages:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install timm pillow opencv-python numpy pandas matplotlib seaborn scikit-learn
pip install fastapi uvicorn pydantic streamlit
pip install pytest black flake8 mypy

# Save installed packages
pip freeze > requirements.txt
```

#### 6. IDE Configuration

**VS Code:**
- [ ] Install Python extension
- [ ] Install Pylance extension
- [ ] Install GitLens extension (optional)
- [ ] Configure Python interpreter (select venv)
- [ ] Set up Black formatter
- [ ] Enable format on save

**PyCharm:**
- [ ] Configure Python interpreter (select venv)
- [ ] Enable PEP 8 warnings
- [ ] Configure Black as formatter
- [ ] Enable Git integration

#### 7. Verify Setup

```bash
# Check Python works
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Check Git works
git status

# Check you can run tests (when available)
pytest tests/
```

#### 8. Read Documentation

- [ ] Read README.md
- [ ] Read CONTRIBUTING.md
- [ ] Read your role in TEAM_ROLES.md
- [ ] Read GIT_WORKFLOW.md
- [ ] Skim PROJECT_SPECIFICATION.md

#### 9. Communication Setup

- [ ] Join team communication channel (Slack/Discord/etc.)
- [ ] Enable GitHub notifications
- [ ] Add team meetings to calendar

---

## 🗂 Project Structure Verification

### Verify Folders Exist

```bash
# Run this to check structure (Linux/Mac)
tree -L 2

# Or manually verify key folders:
ls -la apps/
ls -la ml/
ls -la data/
ls -la docs/
ls -la tests/
ls -la artifacts/
```

### Expected Structure

- [x] `apps/` - Application layer (API, Web)
- [x] `ml/` - Machine learning modules
- [x] `data/` - Data storage (raw, processed, splits)
- [x] `docs/` - Documentation
- [x] `tests/` - Test suite
- [x] `artifacts/` - Training outputs
- [x] `notebooks/` - Jupyter notebooks
- [x] `configs/` - Configuration files
- [x] `scripts/` - Utility scripts
- [x] `.github/` - GitHub workflows and templates

---

## 📦 Dependency Management

### Create pyproject.toml (Project Lead)

- [ ] Define project metadata
- [ ] List core dependencies
- [ ] List development dependencies
- [ ] Define optional dependencies for different use cases

### Create requirements.txt

- [ ] Generate from environment: `pip freeze > requirements.txt`
- [ ] Or manually specify pinned versions
- [ ] Separate dev requirements: `requirements-dev.txt`

---

## 🧪 Testing Infrastructure

### Setup pytest

- [ ] Create `pytest.ini` configuration
- [ ] Add test fixtures in `tests/fixtures/`
- [ ] Create sample test files to verify setup
- [ ] Document testing guidelines in CONTRIBUTING.md

### Example Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── unit/
│   ├── __init__.py
│   ├── test_validation.py   # Example test
│   └── test_transforms.py   # Example test
├── integration/
│   ├── __init__.py
│   └── test_pipeline.py     # Example test
└── fixtures/
    ├── sample_images/
    └── test_configs/
```

---

## 📊 Project Management

### GitHub Project Board

Create columns:
- [ ] **Backlog** - Future tasks
- [ ] **To Do** - Prioritized for current sprint
- [ ] **In Progress** - Currently being worked on
- [ ] **In Review** - PR submitted, awaiting review
- [ ] **Done** - Completed and merged

### Initial Issues

Create issues for Phase 2 tasks:
- [ ] "Setup Kaggle dataset download"
- [ ] "Implement data validation pipeline"
- [ ] "Create EDA notebook"
- [ ] "Implement data splitting"
- [ ] "Implement baseline model"
- [ ] "Implement EfficientNetV2 model"
- [ ] "Create training pipeline"
- [ ] "Implement evaluation metrics"
- [ ] "Implement Grad-CAM"
- [ ] "Create FastAPI endpoints"
- [ ] "Build Streamlit UI"

### Milestones

- [ ] Phase 1: Project Setup (Current)
- [ ] Phase 2: Data Engineering
- [ ] Phase 3: Model Development
- [ ] Phase 4: Evaluation & Analysis
- [ ] Phase 5: Robustness & OOD
- [ ] Phase 6: Deployment
- [ ] Phase 7: Final Polish

---

## 🔒 Security & Best Practices

### .gitignore Configuration

Verify these are in `.gitignore`:
- [x] `venv/` or `env/`
- [x] `__pycache__/`
- [x] `*.pyc`
- [x] `.env` (for secrets)
- [x] `data/raw/` (large dataset files)
- [x] `artifacts/checkpoints/*.pth` (large model files)
- [ ] `.DS_Store` (macOS)
- [ ] `*.ipynb_checkpoints`

### Environment Variables

- [ ] Create `.env.example` template
- [ ] Document required environment variables
- [ ] Never commit `.env` with actual credentials

### Pre-commit Hooks (Optional)

- [ ] Install pre-commit: `pip install pre-commit`
- [ ] Create `.pre-commit-config.yaml`
- [ ] Run `pre-commit install`
- [ ] Test: `pre-commit run --all-files`

---

## 🚀 First Team Exercise

### Practice Workflow

Each team member should:

1. **Create a test branch**
   ```bash
   git checkout -b test/your-name
   ```

2. **Make a small change**
   - Add your name to a new file `docs/TEAM_MEMBERS.md`
   
3. **Commit and push**
   ```bash
   git add docs/TEAM_MEMBERS.md
   git commit -m "docs: add team member name"
   git push -u origin test/your-name
   ```

4. **Create a Pull Request**
   - Go to GitHub
   - Create PR from your branch to main
   - Fill out PR template
   
5. **Code Review**
   - Review another team member's PR
   - Approve when satisfied
   
6. **Merge**
   - Project lead merges all PRs
   
7. **Update local main**
   ```bash
   git checkout main
   git pull origin main
   git branch -d test/your-name
   ```

**Success Criteria:** All team members complete this exercise successfully.

---

## 📅 Team Kickoff Meeting Agenda

### Meeting Goals
- Introduce all team members
- Review project scope and timeline
- Assign roles and responsibilities
- Discuss communication protocols
- Address questions

### Agenda (60 minutes)

1. **Introductions** (10 min)
   - Each member shares background and interests
   
2. **Project Overview** (15 min)
   - Review README and PROJECT_SPECIFICATION
   - Discuss expected outcomes
   
3. **Team Structure** (10 min)
   - Review TEAM_ROLES.md
   - Confirm role assignments
   
4. **Workflow & Tools** (15 min)
   - Git workflow walkthrough
   - GitHub project board
   - Communication channels
   
5. **Q&A** (10 min)
   - Address concerns
   - Clarify expectations

---

## ✅ Phase 1 Completion Criteria

Phase 1 is complete when:

- [x] Repository structure is finalized
- [x] All documentation is in place
- [ ] All team members have working development environments
- [ ] Each team member has successfully created a test PR
- [ ] Communication channels are established
- [ ] GitHub project board is set up with initial tasks
- [ ] Team kickoff meeting is completed
- [ ] Everyone understands their role and responsibilities

---

## 📝 Next Steps (Phase 2)

After Phase 1 completion:

1. **Data Engineering Lead** starts dataset acquisition
2. **ML Model Lead** begins researching model implementations
3. **Others** can start prototyping their components or helping with data
4. **Project Lead** tracks progress and removes blockers

---

## 🆘 Troubleshooting Common Issues

### Python Installation Issues

**Windows:**
- Ensure "Add Python to PATH" was checked during installation
- Restart terminal after installation

**macOS:**
- Use Homebrew: `brew install python@3.9`

**Linux:**
- Use package manager: `sudo apt install python3.9`

### Virtual Environment Activation Issues

**Windows PowerShell:** If execution policy prevents activation:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Git Authentication Issues

**HTTPS:** Use Personal Access Token, not password
- Go to GitHub Settings → Developer settings → Personal access tokens
- Generate token with `repo` scope
- Use token as password

**SSH:** Set up SSH keys (recommended)
```bash
ssh-keygen -t ed25519 -C "your.email@example.com"
# Add public key to GitHub Settings → SSH keys
```

### PyTorch Installation Issues

If CUDA version doesn't match:
```bash
# CPU only (works everywhere)
pip install torch torchvision torchaudio

# Or use specific CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 📞 Getting Help

- **Technical Issues:** Create GitHub issue with `help-wanted` label
- **Workflow Questions:** Ask in team channel
- **Urgent Blockers:** Contact Project Lead directly

---

<div align="center">
  <strong>Ready to build something amazing! 🚀</strong>
</div>
