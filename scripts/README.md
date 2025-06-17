# Code Quality Management Scripts

This directory contains scripts and tools for maintaining code quality in the AIND Dashboard project.

## Quick Start

For immediate code formatting and quality checks:

```bash
# Quick approach using Makefile
make quality-quick

# Or use the shell script directly
./scripts/quick_format.sh

# Or use the comprehensive Python script
python scripts/format_code.py
```

## Available Tools

### 1. Makefile Targets

The Makefile provides convenient shortcuts for common tasks:

```bash
make help           # Show all available targets
make install-dev    # Install development dependencies
make format         # Apply formatting (isort + black)
make lint           # Run flake8 linting
make type-check     # Run mypy type checking
make security-scan  # Run bandit security analysis
make quality-quick  # Quick pipeline (format + lint)
make quality-full   # Full pipeline (all tools)
make test           # Run tests
make clean          # Clean up generated files
```

### 2. Shell Script (`quick_format.sh`)

**Reasoning**: Provides immediate formatting without complex setup. Best for quick fixes.

```bash
./scripts/quick_format.sh
```

Features:
- Activates conda environment automatically
- Installs dependencies if needed
- Applies formatting step-by-step with progress reporting
- Handles errors gracefully

### 3. Python Script (`format_code.py`)

**Reasoning**: Most comprehensive option with detailed reporting and incremental processing to manage context limits.

```bash
# Run full pipeline
python scripts/format_code.py

# Skip specific tools
python scripts/format_code.py --skip mypy bandit

# Custom project root
python scripts/format_code.py --project-root /path/to/project
```

Features:
- Incremental processing to avoid context limits
- Detailed JSON reporting
- Error handling and rollback capabilities
- Customizable tool selection

## Configuration Files

### Tool Configurations

- **`.flake8`**: Linting rules and exclusions
- **`pyproject.toml`**: Configuration for black, isort, and mypy
- **`requirements-dev.txt`**: Development dependencies

### Key Settings

**Black**: 88 character line length, compatible with flake8
**isort**: Black-compatible profile, organized import sections
**flake8**: Relaxed rules for Dash application patterns
**mypy**: Gradual typing with external library ignores

## Incremental Application Strategy

To manage context limits and ensure smooth progression:

### Phase 1: Core Files
1. `app.py` - Main application file
2. `shared_utils.py` - Shared utilities

### Phase 2: App Components
3. `app_elements/` - UI components
4. `callbacks/` - Dash callbacks

### Phase 3: Utilities
5. `app_utils/` - Application utilities

### Phase 4: Tests and Documentation
6. `tests/` - Test files (optional)

## Handling Common Issues

### Context Limit Management

**Problem**: Large codebases can overwhelm AI context limits
**Solution**: Process files incrementally by directory/module

```bash
# Process specific directories
python scripts/format_code.py --target-paths app_elements/ app_utils/
```

### Flake8 Errors

**Common Issues**:
- Import order: Fixed by isort (run first)
- Line length: Fixed by black
- Star imports: Allowed in `__init__.py` and `app.py` per config

### MyPy Type Errors

**Strategy**: Start with main files, gradually expand coverage
```bash
# Start with core files only
python scripts/format_code.py --skip bandit --target-paths app.py shared_utils.py
```

### Security Scan Issues

**Bandit** may flag false positives in Dash apps:
- Review `bandit_report.json` for actual issues
- Use `# nosec` comments for confirmed false positives

## Pre-Deployment Checklist

1. **Install dependencies**: `make install-dev`
2. **Format code**: `make format`
3. **Fix linting issues**: `make lint`
4. **Address type issues**: `make type-check` (optional)
5. **Review security scan**: `make security-scan`
6. **Run tests**: `make test`
7. **Clean up**: `make clean`

## Integration with Development Workflow

### Git Hooks (Optional)

Consider adding pre-commit hooks:

```bash
# Install pre-commit
conda activate main
pip install pre-commit

# Set up hooks
pre-commit install
```

### CI/CD Integration

The Makefile targets can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions step
- name: Code Quality Check
  run: |
    conda activate main
    make quality-full
```

## Troubleshooting

### Common Environment Issues

1. **Conda activation fails**: Ensure conda is properly initialized
2. **Package conflicts**: Use separate environment for development tools
3. **Permission errors**: Ensure scripts are executable (`chmod +x`)

### Performance Optimization

- Use `--skip` flags to avoid time-consuming tools during development
- Process directories separately for large codebases
- Cache results using the Python script's JSON output

## Reporting and Monitoring

All tools generate reports:
- **Flake8**: Console output with line-by-line issues
- **MyPy**: Type error reports
- **Bandit**: JSON security report (`bandit_report.json`)
- **Python script**: Comprehensive JSON report (`quality_check_results_*.json`)

Use these reports to track quality improvements over time. 