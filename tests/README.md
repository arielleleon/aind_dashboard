# AIND Dashboard Testing Infrastructure

This directory contains the testing infrastructure for the AIND Dashboard, designed to support large-scale refactoring while maintaining code quality and functionality.

## 🧪 Test Organization

The tests are organized by **functionality** rather than **file structure** to support future refactoring:

```
tests/
├── conftest.py                    # Shared fixtures and configuration
├── unit/                          # Unit tests for individual functions
│   ├── test_data_operations/      # Data loading and processing
│   ├── test_statistical_analysis/ # Statistical functions and analysis
│   ├── test_ui_components/        # UI component logic
│   ├── test_callback_logic/       # Callback function logic
│   └── test_utilities/            # Shared utility functions
└── e2e/                          # End-to-end tests
    ├── test_app_smoke.py         # Basic functionality verification
    ├── test_user_workflows/      # Complete user interaction flows
    └── test_integration/         # Component integration tests
```

## 🚀 Quick Start

### 1. Install Testing Dependencies

```bash
# Install testing dependencies
pip install -r requirements-test.txt

# Or use the test runner to install automatically
python run_tests.py --install-deps
```

### 2. Run Tests

```bash
# Quick baseline verification (recommended first step)
python run_tests.py --smoke

# Run unit tests only
python run_tests.py --unit

# Run end-to-end tests
python run_tests.py --e2e

# Run all tests
python run_tests.py --all

# Fast tests (no E2E, good for development)
python run_tests.py --fast

# With coverage reporting
python run_tests.py --unit --coverage
```

### 3. Alternative pytest commands

```bash
# Direct pytest usage
pytest tests/e2e/test_app_smoke.py -v        # Smoke tests
pytest tests/unit/ -v                        # Unit tests
pytest tests/ -m "not e2e" -v               # Fast tests
pytest tests/ --cov --cov-report=html       # With coverage
```

## Test Types

### Smoke Tests (`--smoke`)
**Purpose**: Verify the app works in its current state before refactoring
- App starts without errors
- Main UI elements render
- Data loading completes
- No immediate crashes

**When to run**: Before starting any refactoring work, after major changes

### Unit Tests (`--unit`)
**Purpose**: Test individual functions and components in isolation
- Data processing functions
- Statistical calculations
- Utility functions
- Component logic

**When to run**: During development, before commits

### End-to-End Tests (`--e2e`)
**Purpose**: Test complete user workflows through browser automation
- User interaction flows
- Data visualization updates
- Error handling scenarios

**When to run**: Before releases, after major feature changes

### Fast Tests (`--fast`)
**Purpose**: Quick feedback during development (unit + integration, no E2E)
- Combines unit and integration tests
- Skips slower browser-based tests

**When to run**: Frequently during development

## Test Infrastructure Design

### Refactor-Friendly Organization
- Tests are organized by **functionality**, not current file structure
- When you refactor and move code, tests can be easily reorganized
- Function-focused tests remain valid regardless of module restructuring

### Fixture Design
- **Reusable fixtures** in `conftest.py` for common test data
- **Mock utilities** to test functions without full data loading
- **Sample data** that mimics real data structure but with predictable values

### Testing Patterns
- **Isolated unit tests** for pure functions
- **Mocked dependencies** to avoid external data requirements
- **Integration tests** for component interactions
- **Browser automation** for complete user workflows

## Adding New Tests

### For New Functionality
1. **Identify the function type**: Data processing, statistical, UI logic, etc.
2. **Choose the appropriate directory**: Based on functionality, not current file location
3. **Use existing fixtures**: Leverage shared test data and utilities
4. **Follow naming conventions**: `test_*` for functions, `Test*` for classes

### Example: Testing a New Statistical Function

```python
# tests/unit/test_statistical_analysis/test_new_calculation.py
import pytest
from your_module import your_new_function

def test_new_statistical_calculation(sample_statistical_data):
    """Test the new statistical calculation function"""
    result = your_new_function(sample_statistical_data['values'])
    
    # Test with predictable data
    assert result is not None
    assert isinstance(result, float)
    assert 0 <= result <= 1  # or whatever bounds make sense
```

### For Refactored Code
1. **Move tests** to match new organization
2. **Update imports** in test files
3. **Keep test logic unchanged** if function behavior is the same
4. **Update only the import paths**, not the test assertions

## Coverage and Quality

### Coverage Goals
- **Minimum 50%** overall coverage (configured in `pytest.ini`)
- **Focus on critical paths**: Data processing, statistical functions, core logic
- **Less focus on UI rendering**: More on business logic

### Quality Checks
- **No critical JavaScript errors** in smoke tests
- **Consistent test data** using fixtures
- **Isolated tests** that don't depend on external state
- **Fast execution** for unit tests (< 1s per test ideal)

## Configuration

### pytest.ini
- Test discovery patterns
- Coverage settings
- Markers for test categorization
- Timeout settings

### conftest.py
- Shared fixtures for test data
- Mock utilities
- App instance for E2E tests

### requirements-test.txt
- Testing framework dependencies
- Browser automation tools
- Coverage and reporting tools

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Make sure you're in the project root
cd /path/to/aind_dashboard
python run_tests.py --smoke
```

**Missing Dependencies**
```bash
# Install testing dependencies
pip install -r requirements-test.txt
```

**Browser Issues (E2E tests)**
```bash
# Make sure Chrome/Firefox is installed
# E2E tests use headless browsers by default
```

**Slow Tests**
```bash
# Use fast tests during development
python run_tests.py --fast

# Save E2E tests for major milestones
python run_tests.py --e2e
```

## Best Practices

1. **Start with smoke tests** before any refactoring
2. **Write tests for new functionality** as you develop
3. **Keep tests focused** on single responsibilities
4. **Use mocks liberally** to isolate functionality
5. **Run fast tests frequently** during development
6. **Run full test suite** before major commits/releases
7. **Update test organization** as you refactor code structure 