# Git Workflow Handbook for ML Team

## 📋 Table of Contents
- [Quick Start](#quick-start)
- [Daily Workflow](#daily-workflow)
- [Branch Strategy](#branch-strategy)
- [Common Commands](#common-commands)
- [Handling Conflicts](#handling-conflicts)
- [Pull Request Process](#pull-request-process)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### First Time Setup
```bash
# Clone the repository
git clone https://github.com/UnicornnGrg/AppliedMachineLearning.git
cd AppliedMachineLearning

# Configure your identity
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Check your configuration
git config --list
```

### Install Dependencies
```bash
# Install Python packages
pip install -r requirements.txt

# Install Git LFS (for large files)
git lfs install
git lfs pull
```

---

## 📅 Daily Workflow

### Start Your Day
```bash
# 1. Switch to main branch
git checkout main

# 2. Get latest changes from GitHub
git pull origin main

# 3. Create or switch to your feature branch
git checkout -b your-feature-name
# OR if branch already exists:
git checkout your-feature-name

# 4. Merge latest main into your branch
git merge main
```

### While Working
```bash
# Check what files you've changed
git status

# See specific changes
git diff

# Stage files for commit
git add <filename>                    # Add specific file
git add .                             # Add all changed files
git add data_cleaning/                # Add entire folder

# Commit your changes
git commit -m "Clear description of what you did"

# Push to GitHub (backup + team visibility)
git push origin your-feature-name
```

### End of Day / Task Complete
```bash
# 1. Stage and commit final changes
git add .
git commit -m "Complete [feature name]: brief description"

# 2. Push to GitHub
git push origin your-feature-name

# 3. Create Pull Request on GitHub for team review
# Go to: https://github.com/UnicornnGrg/AppliedMachineLearning/pulls
```

---

## 🌿 Branch Strategy

### Branch Naming Convention
```
main                    # Production-ready code (protected)
├── data-cleaning       # Data preprocessing and cleaning
├── feature-engineering # Feature creation and selection
├── model-training      # Model development and training
├── model-evaluation    # Testing and metrics
└── visualization       # Plots and dashboards
```

### Creating a New Branch
```bash
# Create and switch to new branch
git checkout -b branch-name

# Push new branch to GitHub
git push -u origin branch-name
```

### Switching Between Branches
```bash
# List all branches
git branch -a

# Switch to existing branch
git checkout branch-name

# Switch back to main
git checkout main
```

### Deleting a Branch
```bash
# Delete local branch (after merging)
git branch -d branch-name

# Delete remote branch on GitHub
git push origin --delete branch-name

# Force delete (if not merged)
git branch -D branch-name
```

### Merging Feature Branch to Main

#### Method 1: Via Pull Request (Recommended)
```bash
# 1. Ensure your feature branch is up to date
git checkout your-feature-name
git pull origin main
git push origin your-feature-name

# 2. Create Pull Request on GitHub
# Go to: https://github.com/UnicornnGrg/AppliedMachineLearning/pulls
# Click "New pull request"
# Select: base: main <- compare: your-feature-name
# Add description and request reviews

# 3. After approval, merge on GitHub
# Click "Merge pull request" → "Confirm merge"

# 4. Update your local main branch
git checkout main
git pull origin main

# 5. Delete the merged branch
git branch -d your-feature-name
git push origin --delete your-feature-name
```

#### Method 2: Direct Merge (Command Line)
```bash
# 1. Update main branch
git checkout main
git pull origin main

# 2. Merge feature branch into main
git merge your-feature-name

# 3. Resolve any conflicts if they occur
# (See "Handling Conflicts" section)

# 4. Push merged changes to GitHub
git push origin main

# 5. Delete the merged feature branch
git branch -d your-feature-name
git push origin --delete your-feature-name
```

#### Method 3: Squash Merge (Clean History)
```bash
# 1. Update main branch
git checkout main
git pull origin main

# 2. Squash merge feature branch (combines all commits into one)
git merge --squash your-feature-name

# 3. Commit the squashed changes
git commit -m "Add feature: brief description of all changes"

# 4. Push to GitHub
git push origin main

# 5. Delete the feature branch
git branch -d your-feature-name
git push origin --delete your-feature-name
```

#### Pre-Merge Checklist
Before merging your feature branch to main:
- [ ] All tests pass
- [ ] Code has been reviewed (if using PR)
- [ ] Feature branch is up to date with main
- [ ] No merge conflicts
- [ ] Documentation is updated
- [ ] Committed all changes
- [ ] Code follows team standards

#### Post-Merge Steps
```bash
# 1. Notify team that main has been updated
# (via Slack, Teams, or your communication channel)

# 2. Team members should update their local main
git checkout main
git pull origin main

# 3. Team members should update their feature branches
git checkout their-feature-branch
git merge main
```

---

## 💻 Common Commands

### Checking Status
```bash
# See modified files
git status

# See changes in files
git diff

# See commit history
git log
git log --oneline
git log --graph --oneline --all

# See who changed what
git blame filename
```

### Committing Changes
```bash
# Stage specific files
git add file1.py file2.py

# Stage all Python files
git add *.py

# Stage all files in a directory
git add data_cleaning/

# Stage everything
git add .

# Commit with message
git commit -m "Your message here"

# Commit with detailed message
git commit
# Opens editor for multi-line message
```

### Pushing and Pulling
```bash
# Push to GitHub
git push origin branch-name

# Pull latest from GitHub
git pull origin branch-name

# Pull from main
git pull origin main

# Fetch without merging
git fetch origin
```

### Undoing Changes
```bash
# Discard changes in working directory
git restore filename

# Unstage a file (keep changes)
git restore --staged filename

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1

# Revert a specific commit
git revert <commit-hash>
```

### Stashing Changes
```bash
# Save changes temporarily
git stash

# List all stashes
git stash list

# Apply most recent stash
git stash apply

# Apply and remove stash
git stash pop

# Drop a stash
git stash drop
```

---

## ⚔️ Handling Conflicts

### When Merge Conflicts Occur
```bash
# 1. Try to merge or pull
git merge main
# OR
git pull origin main

# 2. If conflict occurs, check status
git status

# 3. Open conflicted files and look for:
# <<<<<<< HEAD
# Your changes
# =======
# Their changes
# >>>>>>> main

# 4. Edit the file to resolve conflicts
# Remove conflict markers and keep desired code

# 5. Stage the resolved files
git add resolved-file.py

# 6. Complete the merge
git commit -m "Resolve merge conflict in resolved-file.py"

# 7. Push changes
git push origin your-branch
```

### Preventing Conflicts
```bash
# Update your branch regularly
git checkout your-branch
git pull origin main

# Before starting work
git pull origin main

# Communicate with team about file ownership
```

---

## 🔄 Pull Request Process

### Creating a Pull Request

1. **Push your branch to GitHub**
```bash
git push origin your-feature-name
```

2. **On GitHub**
   - Go to: https://github.com/UnicornnGrg/AppliedMachineLearning
   - Click "Pull requests" → "New pull request"
   - Select your branch
   - Add title and description
   - Request reviewers
   - Click "Create pull request"

3. **PR Description Template**
```markdown
## What changed?
Brief description of changes

## Why?
Reason for changes

## How to test?
Steps to verify changes work

## Checklist
- [ ] Code runs without errors
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No merge conflicts
```

### Reviewing a Pull Request

```bash
# 1. Fetch the branch
git fetch origin

# 2. Check out the PR branch
git checkout pr-branch-name

# 3. Test the code
python script.py

# 4. Return to your branch
git checkout your-branch
```

### Merging a Pull Request

**On GitHub (Recommended):**
- Click "Merge pull request"
- Choose merge type (usually "Squash and merge")
- Confirm merge
- Delete branch

**Via Command Line:**
```bash
# 1. Switch to main
git checkout main

# 2. Merge the feature branch
git merge feature-branch

# 3. Push to GitHub
git push origin main

# 4. Delete merged branch
git branch -d feature-branch
git push origin --delete feature-branch
```

---

## ✅ Best Practices

### Commit Messages
```bash
# ✅ Good commit messages
git commit -m "Add outlier removal to data cleaning pipeline"
git commit -m "Fix missing value imputation bug in WAGP column"
git commit -m "Update feature engineering with age groups"

# ❌ Bad commit messages
git commit -m "update"
git commit -m "fixed stuff"
git commit -m "asdf"
```

### When to Commit
- ✅ After completing a logical unit of work
- ✅ When code is working
- ✅ Before switching tasks
- ✅ At least once per day
- ❌ Don't commit broken code
- ❌ Don't commit generated files (.pyc, __pycache__)

### What NOT to Commit
```bash
# Add to .gitignore
__pycache__/
*.pyc
.env
*.log
.DS_Store
*.swp
*.swo
.vscode/
.idea/
```

### Branch Hygiene
- Keep branches short-lived (2-3 days max)
- Delete branches after merging
- One feature per branch
- Update from main daily
- Don't push directly to main

---

## 🛠️ Troubleshooting

### "Your branch is behind 'origin/main'"
```bash
git pull origin main
```

### "Your branch is ahead of 'origin/main'"
```bash
git push origin your-branch
```

### "Your branch has diverged from 'origin/main'"
```bash
# Option 1: Merge
git pull origin main

# Option 2: Rebase (cleaner history)
git pull --rebase origin main
```

### Accidentally Committed to Wrong Branch
```bash
# 1. Copy the commit hash
git log --oneline

# 2. Switch to correct branch
git checkout correct-branch

# 3. Cherry-pick the commit
git cherry-pick <commit-hash>

# 4. Go back to wrong branch
git checkout wrong-branch

# 5. Remove the commit
git reset --hard HEAD~1
```

### Need to Switch Branches with Uncommitted Changes
```bash
# Option 1: Stash changes
git stash
git checkout other-branch
# Later: git stash pop

# Option 2: Commit changes
git add .
git commit -m "WIP: work in progress"
git checkout other-branch
```

### Accidentally Deleted Important Files
```bash
# Recover deleted file
git restore filename

# Recover all deleted files
git restore .
```

### Need to Undo a Pushed Commit
```bash
# ⚠️ Only if you're the only one on the branch
git reset --hard HEAD~1
git push --force origin your-branch

# ✅ Better: Create new commit that undoes changes
git revert <commit-hash>
git push origin your-branch
```

---

## 🔍 Advanced Commands

### View Changes Between Branches
```bash
# See what's different
git diff main..your-branch

# See file names only
git diff --name-only main..your-branch
```

### Sync Fork with Original Repository
```bash
# Add upstream remote
git remote add upstream https://github.com/original/repo.git

# Fetch upstream changes
git fetch upstream

# Merge upstream changes
git checkout main
git merge upstream/main
git push origin main
```

### Interactive Rebase (Clean Up Commits)
```bash
# Combine last 3 commits
git rebase -i HEAD~3

# In editor, change 'pick' to 'squash' for commits to combine
# Save and close editor
```

### Search Commit History
```bash
# Find commits with specific text
git log --grep="search term"

# Find when a function was changed
git log -S "function_name"

# Find commits by author
git log --author="Your Name"
```

---

## 📚 Useful Aliases

Add these to your `~/.gitconfig` or set with `git config --global`:

```bash
# Create aliases
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.unstage 'reset HEAD --'
git config --global alias.last 'log -1 HEAD'
git config --global alias.visual 'log --graph --oneline --all'

# Now use them:
git st          # instead of git status
git co main     # instead of git checkout main
git visual      # pretty commit graph
```

---

## 👥 Team Coordination

### File Ownership During Development
- **Person 1**: `data_cleaning/`, `data/processed/`
- **Person 2**: `feature_engineering/`, `notebooks/features.ipynb`
- **Person 3**: `model_training/`, `models/`
- **Person 4**: `evaluation/`, `results/`

### Communication Checklist
- [ ] Daily standup: "I'm working on X files today"
- [ ] Before editing shared files: Check with team
- [ ] PR created: Tag reviewers
- [ ] Merge conflicts: Coordinate resolution
- [ ] Main updated: Notify team to pull

---

## 📖 Resources

- [Git Documentation](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com/)
- [Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)
- [Interactive Git Tutorial](https://learngitbranching.js.org/)

---

## 🆘 Quick Help

```bash
# Get help for any command
git help <command>
git <command> --help

# Examples:
git help commit
git merge --help
```

---

**Last Updated**: November 18, 2025  
**Repository**: https://github.com/UnicornnGrg/AppliedMachineLearning  
**Team Size**: 4 people
