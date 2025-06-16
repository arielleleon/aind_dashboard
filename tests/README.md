# AIND Dashboard Testing Architecture

## Overview

This testing suite has been streamlined and consolidated to provide comprehensive coverage while maintaining simplicity and ease of maintenance for developers.

## Simplified Structure

```
tests/
├── conftest.py               # Shared fixtures and configuration
├── fixtures/                 # Test data and sample data generators
│   ├── sample_data.py       # Realistic test data based on actual app data
│   └── __init__.py
├── unit/                    # Unit tests for core components
│   ├── test_core_components.py    # **MAIN TEST FILE** - Core functionality
│   ├── test_callback_logic/       # Callback integration tests
│   ├── test_statistical_analysis/ # Statistical utilities tests
│   ├── test_ui_components/        # UI component tests
│   └── test_utilities/            # Utility function tests
└── e2e/                     # End-to-end integration tests
    └── test_app_smoke.py    # Basic app startup and functionality tests
```

## Key Changes in This Refactor

### ✅ **Consolidated Core Tests**
- **Primary File**: `tests/unit/test_core_components.py`
- Combines tests for the three most critical components:
  - `PercentileCoordinator` (core percentile functionality)
  - `AlertCoordinator` (alert management)
  - `EnhancedDataLoader` (data loading)
- **Simplified from 3 separate files totaling 1400+ lines to 1 file with 310 lines**

### ✅ **Removed Redundancies**
- Eliminated duplicate `test_percentile_utils.py` files (were in 2 locations)
- Removed empty directories (`test_integration/`, `test_user_workflows/`)
- Consolidated overlapping test cases

### ✅ **Enhanced Test Data**
- Uses realistic sample data extracted from actual app (`tests/fixtures/sample_data.py`)
- Provides both realistic and simple data fixtures for different test needs
- Maintains backward compatibility for existing fixtures

## Running Tests

### **Quick Core Test Suite**
```bash
# Run the main consolidated tests (fastest)
pytest tests/unit/test_core_components.py -v
```

### **Full Unit Test Suite**
```bash
# Run all unit tests
pytest tests/unit/ -v
```

### **End-to-End Tests**
```bash
# Run smoke tests (note: app startup can take 5+ minutes)
pytest tests/e2e/test_app_smoke.py -v -m "not slow"

# Run full E2E including server startup
pytest tests/e2e/test_app_smoke.py -v
```

### **All Tests**
```bash
# Run everything (comprehensive test suite)
pytest tests/ -v
```

## Test Categories

### **Core Components** (`test_core_components.py`)
- **Essential for development**: Tests the primary application logic
- **Focus**: Key functionality validation with simplified test cases
- **Coverage**: Initialization, core operations, error handling, integration

### **Callback Logic** (`test_callback_logic/`)
- Tests Dash callback integration and UI interactions
- Focus on callback behavior and state management

### **Statistical Analysis** (`test_statistical_analysis/`)
- Tests statistical utilities and calculation functions
- Focus on mathematical accuracy and edge cases

### **UI Components** (`test_ui_components/`)
- Tests UI-specific business logic and data formatting
- Focus on user interface behavior and data presentation

### **Utilities** (`test_utilities/`)
- Tests helper functions and utility modules
- Focus on reusable functionality across the application

## Developer Guidelines

### **When Adding New Tests**

1. **Start with Core Components**: Add tests to `test_core_components.py` if testing:
   - PercentileCoordinator
   - AlertCoordinator  
   - EnhancedDataLoader
   - Basic integration scenarios

2. **Use Appropriate Subdirectory**: For specialized functionality:
   - Callbacks → `test_callback_logic/`
   - Statistics → `test_statistical_analysis/`
   - UI Logic → `test_ui_components/`
   - Utilities → `test_utilities/`

3. **Follow the Simplified Pattern**:
   ```python
   class TestYourComponent:
       @pytest.fixture
       def component(self):
           return YourComponent()
       
       def test_core_functionality(self, component):
           # Test essential behavior
           assert component.works()
       
       def test_error_handling(self, component):
           # Test graceful error handling
           with pytest.raises(ExpectedError):
               component.fail_gracefully()
   ```

### **Testing Philosophy**

- **Test Core Functionality**: Focus on essential behavior over edge cases
- **Use Realistic Data**: Leverage `tests/fixtures/sample_data.py` for realistic test scenarios
- **Mock External Dependencies**: Use `unittest.mock` for external services and complex setup
- **Keep Tests Simple**: Prefer clear, readable tests over comprehensive coverage of unlikely scenarios

### **Fixtures Available**

From `conftest.py`:
- `sample_session_data` - Realistic session data for testing
- `simple_session_data` - Minimal data for lightweight tests
- `mock_app_utils` - Mock AppUtils instance with common methods
- `sample_statistical_data` - Data for statistical function testing

## Migration Notes

If you have existing test files that reference the old structure:
- Update imports to use `test_core_components` for core functionality
- Use the new fixtures from `conftest.py`
- Follow the simplified test patterns shown in `test_core_components.py`

## Performance

The consolidated test architecture provides:
- **~75% reduction** in test file size and complexity
- **Faster test execution** through focused test cases
- **Easier maintenance** with centralized core component testing
- **Better developer experience** with clear, documented test structure

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

## Test Fixtures

The `conftest.py` file provides shared fixtures for all tests:

- `sample_session_data`: Realistic session data for testing
- `simple_session_data`: Minimal session data for lightweight tests  
- `table_display_data`: Pre-formatted table data for UI tests
- `statistical_data`: Statistical analysis test data
- `strata_test_cases`: Strata abbreviation test cases
- `mock_cache_manager`: Mock cache manager for testing
- `sample_data_provider`: Global data provider instance

These fixtures ensure consistent, realistic test data across all test modules.

---

**Last Updated**: 2025-06-06
**Real Data Source**: AIND Dashboard Production App  
**Test Coverage**: 37 unit tests, all passing with realistic data 