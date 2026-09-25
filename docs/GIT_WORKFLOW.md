# Git Workflow Guide for CoffeeGuard AI

This document provides a practical, step-by-step guide for using Git in the CoffeeGuard AI project.

---

## 📋 Table of Contents

1. [Quick Reference](#quick-reference)
2. [Initial Setup](#initial-setup)
3. [Daily Workflow](#daily-workflow)
4. [Common Scenarios](#common-scenarios)
5. [Branch Management](#branch-management)
6. [Troubleshooting](#troubleshooting)

---

## ⚡ Quick Reference

### Common Commands

```bash
# Check current status
git status

# See what branch you're on
git branch

# Update from remote
git pull origin main

# Create and switch to new branch
git checkout -b feature/my-feature

# Stage changes
git add .                    # All files
git add path/to/file.py      # Specific file

# Commit changes
git commit -m "feat: add new feature"

# Push to remote
git push origin feature/my-feature

# Switch branches
git checkout main
git checkout feature/other-feature

# Delete local branch
git branch -d feature/old-feature
```

---

## 🚀 Initial Setup

### 1. Clone the Repository

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/coffee-guard-ai.git

# Navigate into the project
cd coffee-guard-ai
```

### 2. Configure Git

```bash
# Set your name and email (first time only)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Verify configuration
git config --list
```

### 3. Verify Setup

```bash
# Check remote connection
git remote -v

# Should show:
# origin  https://github.com/YOUR_USERNAME/coffee-guard-ai.git (fetch)
# origin  https://github.com/YOUR_USERNAME/coffee-guard-ai.git (push)
```

---

## 📆 Daily Workflow

### Morning: Start Your Day

```bash
# 1. Make sure you're on main
git checkout main

# 2. Get latest changes from remote
git pull origin main

# 3. Create your feature branch
git checkout -b feature/your-feature-name
```

### During Development

```bash
# 1. Make changes to files
# ... edit files in your editor ...

# 2. Check what changed
git status
git diff                      # See detailed changes

# 3. Stage your changes
git add .                     # Or stage specific files

# 4. Commit your changes
git commit -m "feat(scope): description of changes"

# 5. Push to remote (first time)
git push -u origin feature/your-feature-name

# 6. Push subsequent changes
git push
```

### End of Day: Save Your Work

```bash
# Even if not finished, commit and push
git add .
git commit -m "wip: partial implementation of feature X"
git push

# This ensures your work is backed up
```

---

## 🔄 Common Scenarios

### Scenario 1: Starting a New Feature

```bash
# Update main branch
git checkout main
git pull origin main

# Create feature branch from main
git checkout -b feature/data-validation

# Start coding
# ... make changes ...

# Commit and push
git add .
git commit -m "feat(data): implement image validator"
git push -u origin feature/data-validation
```

### Scenario 2: Someone Else Updated Main

```bash
# Your feature branch is behind main
# First, commit your current work
git add .
git commit -m "feat: my current changes"

# Switch to main and update
git checkout main
git pull origin main

# Switch back to your feature branch
git checkout feature/your-feature

# Merge main into your branch
git merge main

# If conflicts occur, see "Resolving Conflicts" section
```

### Scenario 3: Creating a Pull Request

```bash
# 1. Make sure all changes are committed and pushed
git status  # Should show "nothing to commit"
git push

# 2. Go to GitHub repository in browser
# 3. Click "Pull requests" tab
# 4. Click "New pull request"
# 5. Select your branch
# 6. Fill out PR template
# 7. Click "Create pull request"
# 8. Request reviews from team members
```

### Scenario 4: Updating PR After Review

```bash
# Make requested changes in your editor
# ... fix issues ...

# Commit and push (automatically updates PR)
git add .
git commit -m "fix: address review comments"
git push

# No need to create a new PR!
```

### Scenario 5: After PR is Merged

```bash
# Switch to main
git checkout main

# Update main with merged changes
git pull origin main

# Delete your local feature branch
git branch -d feature/your-feature

# The remote branch is typically deleted automatically on GitHub
```

### Scenario 6: Working on Multiple Features

```bash
# Save current work
git add .
git commit -m "wip: feature A progress"
git push

# Switch to different feature
git checkout main
git pull origin main
git checkout -b feature/different-feature

# Work on new feature
# ... make changes ...

# Switch back to first feature
git checkout feature/first-feature

# Continue where you left off
```

---

## 🌿 Branch Management

### Creating Branches

```bash
# Feature branch
git checkout -b feature/grad-cam-visualization

# Bug fix branch
git checkout -b bugfix/image-loader-error

# Experiment branch
git checkout -b experiment/mobilenet-comparison

# Documentation branch
git checkout -b docs/api-documentation
```

### Viewing Branches

```bash
# List local branches
git branch

# List all branches (including remote)
git branch -a

# See current branch
git branch --show-current
```

### Switching Branches

```bash
# Switch to existing branch
git checkout main
git checkout feature/my-feature

# Create and switch in one command
git checkout -b feature/new-feature
```

### Deleting Branches

```bash
# Delete local branch (safe - won't delete if unmerged)
git branch -d feature/old-feature

# Force delete (careful!)
git branch -D feature/old-feature

# Delete remote branch
git push origin --delete feature/old-feature
```

---

## 🛠 Troubleshooting

### Problem: "Merge Conflict"

When merging, Git may find conflicting changes:

```bash
# After git merge main shows conflicts:
# 1. Open conflicted files (marked in git status)
# 2. Look for conflict markers:
#    <<<<<<< HEAD
#    Your changes
#    =======
#    Their changes
#    >>>>>>> main

# 3. Edit file to keep correct version
# 4. Remove conflict markers

# 5. Stage resolved files
git add path/to/resolved/file.py

# 6. Complete merge
git commit -m "merge: resolve conflicts with main"
```

### Problem: "Accidentally Committed to Wrong Branch"

```bash
# If you haven't pushed yet:

# 1. Undo last commit but keep changes
git reset --soft HEAD~1

# 2. Stash changes
git stash

# 3. Switch to correct branch
git checkout correct-branch

# 4. Apply stashed changes
git stash pop

# 5. Commit on correct branch
git add .
git commit -m "feat: my changes"
```

### Problem: "Want to Undo Last Commit"

```bash
# Undo commit but keep changes (most common)
git reset --soft HEAD~1

# Undo commit and discard changes (careful!)
git reset --hard HEAD~1

# If already pushed (creates new commit)
git revert HEAD
git push
```

### Problem: "Accidentally Modified Files, Want to Discard"

```bash
# Discard changes to specific file
git checkout -- path/to/file.py

# Discard all uncommitted changes (careful!)
git reset --hard HEAD

# See what would be removed first
git clean -n

# Remove untracked files
git clean -f
```

### Problem: "Need to Save Work But Not Ready to Commit"

```bash
# Stash changes
git stash

# Do other work (switch branches, etc.)

# Restore stashed changes
git stash pop

# List all stashes
git stash list

# Apply specific stash
git stash apply stash@{0}
```

### Problem: "Can't Pull - Uncommitted Changes Conflict"

```bash
# Option 1: Commit your changes first
git add .
git commit -m "wip: progress"
git pull

# Option 2: Stash, pull, then unstash
git stash
git pull
git stash pop
```

### Problem: "Pushed Sensitive Data (API Key, Password)"

```bash
# 1. Remove from file immediately
# 2. Commit removal
git add .
git commit -m "fix: remove sensitive data"
git push

# 3. Rotate the exposed secret/key immediately
# 4. (Advanced) Use git-filter-branch to remove from history
#    (Ask team lead for help)
```

---

## 📝 Best Practices

### DO ✅

- Commit early and often
- Write descriptive commit messages
- Pull before starting new work
- Keep branches focused on one feature
- Delete branches after merging
- Ask for help when stuck

### DON'T ❌

- Commit directly to main
- Push untested code
- Force push to shared branches (`git push -f`)
- Commit secrets or credentials
- Ignore merge conflicts
- Make huge commits with many unrelated changes

---

## 🆘 Getting Help

### Check Status

```bash
# See current state
git status

# See commit history
git log --oneline -10

# See remote branches
git branch -r
```

### Ask Team

If you're stuck:
1. Describe what you were trying to do
2. Share the error message
3. Show output of `git status`
4. Ask in team chat or create GitHub issue

---

## 📚 Additional Resources

- [Official Git Documentation](https://git-scm.com/doc)
- [GitHub Flow Guide](https://guides.github.com/introduction/flow/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Oh Shit, Git!?!](https://ohshitgit.com/) - Fixing common mistakes

---

## Cheat Sheet Summary

```bash
# Start work
git checkout main && git pull origin main
git checkout -b feature/my-feature

# During work
git add .
git commit -m "feat: description"
git push -u origin feature/my-feature  # First time
git push                                # After first time

# End work
git add .
git commit -m "wip: end of day"
git push

# Update from main
git checkout main && git pull origin main
git checkout feature/my-feature
git merge main

# Create PR: Go to GitHub website

# After PR merged
git checkout main && git pull origin main
git branch -d feature/my-feature
```

---

<div align="center">
  <strong>Happy Coding! 🚀</strong>
</div>
