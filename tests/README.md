# AIND Dashboard Test Suite

## Overview
This directory contains the complete test suite for the AIND Dashboard application, featuring comprehensive unit, integration, and end-to-end tests.

## Updated Test Architecture (2024-12-20)

### **New: Realistic Test Fixtures**
The test suite now uses **realistic sample data extracted from the actual app** instead of synthetic test data. This ensures tests accurately reflect production behavior.

#### Key Improvements:
- **Real strata formats**: `"Uncoupled Without Baiting_BEGINNER_v1"` → `"UWBB1"`
- **Actual column structures**: All columns match production data
- **Correct data types**: Real timestamps, subject IDs, and metrics
- **Proper abbreviations**: Tests use actual strata abbreviation logic

## Test Structure

```
tests/
├── fixtures/                    # Centralized realistic test data
│   ├── __init__.py
│   └── sample_data.py          # Real app data for all tests
├── unit/
│   ├── test_ui_components/     # UI layer tests with real data
│   ├── test_data_operations/   # Data loading and processing
│   ├── test_utilities/         # Helper functions
│   ├── test_callback_logic/    # Dash callback tests
│   └── test_statistical_analysis/
├── e2e/                        # End-to-end browser tests
├── conftest.py                 # 🔄 Updated pytest fixtures
└── README.md                   # This file
```

## Using Test Fixtures

### Available Fixtures

#### Core Data Fixtures
```python
@pytest.fixture
def sample_session_data():
    """Realistic session data with 10 sessions from 5 real subjects"""
    
@pytest.fixture  
def simple_session_data():
    """Lightweight data for basic tests"""

@pytest.fixture
def real_strata_data():
    """Real strata names and abbreviations from production"""
```

#### Mock Fixtures
```python
@pytest.fixture
def mock_app_utils():
    """Mock AppUtils with realistic return values"""

@pytest.fixture
def mock_bootstrap_manager():
    """Mock bootstrap manager for statistical tests"""
```

### Using Real Data in Your Tests

#### Example: Testing Strata Abbreviations
```python
from tests.fixtures import get_strata_test_cases

def test_strata_abbreviation():
    """Test with real app strata formats"""
    test_cases = get_strata_test_cases()
    for case in test_cases:
        result = ui_manager.get_strata_abbreviation(case['input'])
        assert result == case['expected']
```

#### Example: Testing with Real Session Data
```python
def test_my_feature(sample_session_data):
    """Test using realistic session data"""
    # sample_session_data contains:
    # - Real subject IDs: '690494', '690486', etc.
    # - Real strata: 'Uncoupled Without Baiting_BEGINNER_v1'
    # - All production columns and data types
    
    result = my_function(sample_session_data)
    assert len(result) == 5  # 5 unique subjects
```

### Real Data Specifications

#### Subject Data (sample_session_data)
- **Subjects**: 5 real subjects (690494, 690486, 702200, 697929, 700708)
- **Sessions**: 10 total sessions with real session numbers
- **Strata**: 7 real strata formats from production
- **Columns**: All production columns including features, metadata, percentiles

#### Strata Formats (real_strata_data)
- `'Uncoupled Without Baiting_BEGINNER_v1'` → `'UWBB1'`
- `'Coupled Baiting_ADVANCED_v2'` → `'CBA2'`
- `'Coupled Baiting_INTERMEDIATE_v1'` → `'CBI1'`
- And 4 more real formats...

## Running Tests

### All Tests
```bash
conda activate main
pytest tests/
```

### Specific Test Categories
```bash
# UI Component tests with real data
pytest tests/unit/test_ui_components/ -v

# Data operations tests
pytest tests/unit/test_data_operations/ -v

# Single test with real fixtures
pytest tests/unit/test_ui_components/test_ui_data_manager.py::TestUIDataManager::test_get_strata_abbreviation -v
```

### Test Coverage
```bash
pytest tests/ --cov=app_utils --cov-report=html
```

## Writing New Tests

### Best Practices with Real Data

1. **Use Real Fixtures**: Always prefer `sample_session_data` over creating synthetic data
2. **Test Real Scenarios**: Use actual strata names and subject IDs from fixtures
3. **Verify Production Behavior**: Tests should match how the app actually works
4. **Use Type Hints**: New test fixtures include proper type annotations

### Example Test Template
```python
import pytest
from tests.fixtures import get_realistic_session_data, get_strata_test_cases

class TestMyFeature:
    def setup_method(self):
        self.sample_data = get_realistic_session_data()
    
    def test_with_real_data(self, sample_session_data):
        """Test description with real app data"""
        # Test implementation using real production data
        result = my_feature_function(sample_session_data)
        
        # Assertions should match real data expectations
        assert '690494' in result  # Real subject ID
        assert result['strata'] == 'Uncoupled Without Baiting_BEGINNER_v1'  # Real strata
```

## Migration Notes

### Previous vs Current Test Data

#### Before (Synthetic):
```python
'subject_id': ['S001', 'S002', 'S003']
'strata': ['A_B_v1', 'A_B_v2', 'A_C_v1']
```

#### After (Real):
```python
'subject_id': ['690494', '690486', '702200']
'strata': ['Uncoupled Without Baiting_BEGINNER_v1', 'Coupled Baiting_ADVANCED_v1']
```

### Updated Test Expectations

## Test Results

### Current Status: All Passing
- **Unit Tests**: 37/37 passing
- **UI Data Manager**: 14/14 passing  
- **Data Operations**: 18/18 passing
- **Helper Functions**: 5/5 passing

### Key Improvements Verified
1. **Strata abbreviation logic** works with real formats
2. **Session data processing** handles production data correctly
3. **UI optimized structures** created with real subject/strata data
4. **Error handling** graceful with realistic edge cases

## Contributing

When adding new tests:

1. **Use existing fixtures** from `tests/fixtures/`
2. **Add realistic test cases** to `sample_data.py` if needed
3. **Update this README** if you add new fixture types
4. **Ensure all tests pass** before submitting

## Fixture Generation

The realistic fixtures were generated using `sample_data_generator.py` which:
1. Loads real app data via `shared_utils`
2. Extracts actual strata names and abbreviations
3. Creates sample data with proper structure
4. Validates against real app behavior

To regenerate fixtures (if needed):
```python
# In Jupyter notebook
from sample_data_generator import main
main()
```

---

**Last Updated**: 2025-06-06
**Real Data Source**: AIND Dashboard Production App  
**Test Coverage**: 37 unit tests, all passing with realistic data 