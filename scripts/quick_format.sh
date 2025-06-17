#!/bin/bash
# Quick code formatting script for immediate use
# This script applies formatting step-by-step with progress reporting

set -e  # Exit on any error

echo " Starting Quick Code Formatting Pipeline"
echo "Project: $(pwd)"
echo "Timestamp: $(date)"

# Initialize conda for bash (more robust activation)
echo " Setting up conda environment..."
eval "$(conda shell.bash hook)"
conda activate main

# Install dev dependencies if needed
echo " Installing development dependencies..."
pip install -r requirements-dev.txt

# Define target directories
TARGETS="app.py shared_utils.py app_elements/ app_utils/ callbacks/"

echo ""
echo "="*60
echo "STEP 1: Import Sorting with isort"
echo "="*60

for target in $TARGETS; do
    if [ -e "$target" ]; then
        echo " Sorting imports in: $target"
        isort --diff "$target" || true
        isort "$target"
        echo " Import sorting applied to: $target"
    fi
done

echo ""
echo "="*60
echo "STEP 2: Code Formatting with Black"
echo "="*60

for target in $TARGETS; do
    if [ -e "$target" ]; then
        echo " Formatting code in: $target"
        black --diff "$target" || true
        black "$target"
        echo " Black formatting applied to: $target"
    fi
done

echo ""
echo "="*60
echo "STEP 3: Linting with flake8"
echo "="*60

for target in $TARGETS; do
    if [ -e "$target" ]; then
        echo " Linting: $target"
        flake8 "$target" || echo "  Linting issues found in $target (review above)"
    fi
done

echo ""
echo "="*60
echo "STEP 4: Security Scan with bandit (optional)"
echo "="*60

echo " Running security scan..."
bandit -r . -f json -o bandit_report.json || echo "  Security issues found (check bandit_report.json)"

echo ""
echo " Quick formatting pipeline completed!"
echo " Check any warnings above and review the output files."
echo " Next steps: Review changes and run tests" 