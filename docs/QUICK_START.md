# Quick Start Guide

**Get up and running with CoffeeGuard AI in 10 minutes**

---

## 🎯 For New Team Members

### 1️⃣ Clone Repository (1 minute)

```bash
git clone https://github.com/YOUR_USERNAME/coffee-guard-ai.git
cd coffee-guard-ai
```

### 2️⃣ Set Up Environment (3 minutes)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies (when available)
pip install -e .
```

### 3️⃣ Configure Git (1 minute)

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### 4️⃣ Read Key Docs (5 minutes)

Must read:
1. [README.md](../README.md) - Project overview
2. [CONTRIBUTING.md](../CONTRIBUTING.md) - How to contribute
3. [TEAM_ROLES.md](TEAM_ROLES.md) - Your role and responsibilities

Optional:
- [GIT_WORKFLOW.md](GIT_WORKFLOW.md) - Detailed Git guide
- [PROJECT_SPECIFICATION.md](PROJECT_SPECIFICATION.md) - Technical details

---

## 🚀 Your First Contribution

### Step-by-Step

```bash
# 1. Create a branch
git checkout -b feature/your-feature-name

# 2. Make changes in your code editor
# ... edit files ...

# 3. Stage and commit
git add .
git commit -m "feat: description of your changes"

# 4. Push to GitHub
git push -u origin feature/your-feature-name

# 5. Create Pull Request on GitHub website

# 6. After merge, update your local main
git checkout main
git pull origin main
git branch -d feature/your-feature-name
```

---

## 📚 Essential Commands

### Git Basics

```bash
git status              # See current state
git branch              # List branches
git checkout main       # Switch to main
git pull origin main    # Update from remote
git log --oneline -5    # See recent commits
```

### Development

```bash
# Run tests (when available)
pytest

# Format code
black ml/ apps/ tests/

# Check code style
flake8 ml/ apps/ tests/

# Run notebooks
jupyter notebook
```

---

## 🗺️ Project Navigation

### Key Directories

```
coffee-guard-ai/
├── 📂 ml/              ← Machine learning code
├── 📂 apps/            ← API and web app
├── 📂 data/            ← Dataset storage
├── 📂 notebooks/       ← Jupyter notebooks
├── 📂 tests/           ← Test files
├── 📂 docs/            ← Documentation (you are here!)
└── 📂 artifacts/       ← Model outputs
```

### Key Files

- `README.md` - Project overview
- `CONTRIBUTING.md` - Contribution guidelines
- `PHASE_1_CHECKLIST.md` - Setup checklist
- `CoffeeGuard_AI_Project_Structure.md` - Detailed project description

---

## 🎯 Phase 1 Quick Tasks

Everyone should:
- [ ] Set up development environment
- [ ] Read core documentation
- [ ] Make a test pull request
- [ ] Understand Git workflow
- [ ] Know your role and responsibilities

---

## 📞 Need Help?

**Quick Questions:** Ask in team chat  
**Technical Issues:** Create GitHub issue  
**Git Problems:** See [GIT_WORKFLOW.md](GIT_WORKFLOW.md) troubleshooting section  
**Urgent:** Contact Project Lead

---

## 🎓 Learning Resources

### Python & Deep Learning
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Python Documentation](https://docs.python.org/3/)

### Tools
- [Git Handbook](https://guides.github.com/introduction/git-handbook/)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Streamlit Docs](https://docs.streamlit.io/)

### Project-Specific
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Transfer Learning Guide](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

---

## ✅ Checklist for Today

**Your first day:**
- [ ] Environment setup complete
- [ ] Can run `git status` successfully
- [ ] Python environment activated
- [ ] Read README, CONTRIBUTING, and your role doc
- [ ] Introduced yourself to team
- [ ] Asked at least one question
- [ ] Created your first test branch

**Ready to contribute!** 🎉

---

<div align="center">
  <strong>Welcome to the team! 🚀</strong>
</div>
