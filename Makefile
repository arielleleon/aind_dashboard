# Makefile for AIND Dashboard Code Quality Management
# Provides convenient targets for formatting, linting, and quality checks

.PHONY: help install-dev format lint type-check security-scan clean test quality-full quality-quick

# Default target
help:
	@echo "Available targets:"
	@echo "  install-dev     Install development dependencies"
	@echo "  format          Apply code formatting (isort + black)"
	@echo "  lint            Run linting with flake8"
	@echo "  type-check      Run type checking with mypy"
	@echo "  security-scan   Run security analysis with bandit"
	@echo "  quality-quick   Run quick quality checks (format + lint)"
	@echo "  quality-full    Run full quality pipeline"
	@echo "  test            Run all tests"
	@echo "  clean           Clean up generated files"

# Install development dependencies
install-dev:
	@echo " Installing development dependencies..."
	conda run -n main pip install -r requirements-dev.txt

# Code formatting
format:
	@echo " Applying code formatting..."
	conda run -n main isort app.py shared_utils.py app_elements/ app_utils/ callbacks/
	conda run -n main black app.py shared_utils.py app_elements/ app_utils/ callbacks/

# Linting
lint:
	@echo " Running flake8 linting..."
	conda run -n main flake8 app.py shared_utils.py app_elements/ app_utils/ callbacks/

# Type checking
type-check:
	@echo " Running mypy type checking..."
	conda run -n main mypy app.py shared_utils.py

# Security scanning
security-scan:
	@echo " Running bandit security scan..."
	conda run -n main bandit -r . -f json -o bandit_report.json

# Quick quality checks (format + lint)
quality-quick:
	@echo " Running quick quality pipeline..."
	$(MAKE) format
	$(MAKE) lint

# Full quality pipeline
quality-full:
	@echo " Running full quality pipeline..."
	$(MAKE) install-dev
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) security-scan
	@echo " Full quality pipeline completed!"

# Run tests
test:
	@echo " Running tests..."
	conda run -n main python -m pytest

# Clean up generated files
clean:
	@echo " Cleaning up generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -f bandit_report.json
	rm -f quality_check_results_*.json
	rm -f .coverage 