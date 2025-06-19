# AIND Dashboard

A comprehensive performance monitoring dashboard for the Allen Institute for Neural Dynamics (AIND) dynamic foraging behavioral experiments. This application provides real-time analysis, statistical monitoring, and alert systems for tracking subject performance across experimental conditions.

## High-Level Overview

The AIND Dashboard transforms raw behavioral session data from dynamic foraging experiments into actionable insights through:

- **Intelligent Data Stratification**: Groups subjects by experimental conditions for fair performance comparisons
- **Robust Statistical Analysis**: Uses Wilson confidence intervals and weighted calculations for reliable percentile estimates  
- **Real-Time Performance Monitoring**: Generates automated alerts for subjects requiring attention
- **Interactive Visualization**: Provides conditional formatting and detailed subject analysis tools

### Data Source
Raw behavioral data is pulled from Allen AIND Amazon S3 storage via the `aind_analysis_arch_result_access` package, focusing on dynamic foraging experimental sessions.

---

## Statistical Computations & Data Pipeline

### Strata Assignment System

The dynamic foraging task involves complex experimental parameters. To enable fair performance comparisons, the dashboard groups subjects into **strata** based on three key experimental conditions:

#### 1. **Task/Curriculum Types** (Most common in last 90 days):
- **Uncoupled Baiting**: 1021 sessions
- **Uncoupled Without Baiting**: 570 sessions  
- **Coupled Baiting**: 283 sessions

#### 2. **Stage Simplification**:
- **Beginner**: Stages 1, 2
- **Intermediate**: Stages 3, 4
- **Advanced**: Stage final and graduated

#### 3. **Version Grouping**:
- **v1, v2, v3**: Grouped by relative effects on experimental parameters (block length, contrast, etc.)

**Example Strata ID**: `Uncoupled Baiting_ADVANCED_v3`
- Task: Uncoupled Baiting
- Stage: STAGE_4 (Advanced)  
- Version: 2.3 (v3 group)

### Statistical Analysis Pipeline

#### Wilson Confidence Intervals
The dashboard employs **Wilson confidence intervals** for robust percentile estimation, providing more accurate confidence bounds than normal approximations, especially for extreme percentiles and small sample sizes.

#### Dual Weighting System

**1. Outlier Session Weighting**
- **Detection Method**: IQR-based outlier identification (factor: 1.5)
- **Handling**: Outlier sessions receive **0.5 weight** instead of full exclusion
- **Purpose**: Maintains data while reducing influence of anomalous sessions

**2. Exponential Decay Rolling Averages**
- **Temporal Weighting**: Recent sessions weighted more heavily through exponential decay
- **Purpose**: Percentile estimates prioritize current performance trends
- **Implementation**: Applied to cumulative rolling averages across all features

#### Percentile Calculation & Categories

**Default Percentile Thresholds** (configurable):
- **SB (Severely Below)**: < 6.5 percentile
- **B (Below)**: 6.5-28 percentile  
- **N (Normal)**: 28-72 percentile (no highlighting)
- **G (Good)**: 72-93.5 percentile
- **SG (Severely Good)**: > 93.5 percentile
- **NS**: Not Scored
- **T**: Threshold alerts (session counts, water intake)

#### Uncertainty & Certainty Scores
Point estimate calculations provide **certainty scores** that indicate the reliability of percentile assignments, with color-coded visualization reflecting confidence levels in the statistical estimates.

---

## Application Architecture & Data Flow

### Core Components

```
app.py (Entry Point)
├── shared_utils.py (Singleton AppUtils)
├── app_elements/ (UI Components)
│   ├── app_main.py (Layout)
│   ├── app_content/ (Main Dashboard)
│   ├── app_filter/ (Filtering Controls)
│   └── app_subject_detail/ (Subject Analysis)
├── app_utils/ (Core Logic)
│   ├── app_data_load/ (Data Loading)
│   ├── app_analysis/ (Statistical Pipeline)
│   ├── app_alerts/ (Alert Generation)
│   └── [percentile_utils, ui_utils, cache_utils, etc.]
└── callbacks/ (Interactive Logic)
```

### Data Flow Pipeline

#### 1. **Data Origin → Raw Processing**
```
AIND S3 Storage 
→ aind_analysis_arch_result_access.han_pipeline.get_session_table()
→ EnhancedDataLoader.load()
→ Cache: "raw_data"
```

#### 2. **Raw Data → Statistical Processing**
```
Raw Session Data
→ DataPipelineManager.process_data_pipeline()
├── ReferenceProcessor (outlier detection, eligibility)
├── Strata Assignment (task/stage/version grouping)
├── OverallPercentileCalculator (Wilson CI percentiles)
└── Rolling Average Calculation (exponential decay weighting)
→ Cache: "session_level_data"
```

#### 3. **Processed Data → Alert Generation**
```
Session-Level Data
→ AlertCoordinator.initialize_alert_service()
├── PercentileCoordinator (percentile-based alerts)
├── ThresholdAnalyzer (session count/water intake alerts)
└── Alert Combination & Classification
→ Cache: "unified_alerts"
```

#### 4. **Data → UI Optimization**
```
Processed Data + Alerts
→ UIDataManager.optimize_session_data_storage()
├── Table Display Data (optimized for DataTable)
├── Subject Detail Data (time series, metadata)
└── Visualization Data (plots, heatmaps)
→ Cache: "optimized_storage", "ui_structures"
```

### Component Communication

**AppUtils (Central Coordinator)**
- **CacheManager**: Intelligent caching with invalidation
- **DataPipelineManager**: Unified statistical processing
- **AlertCoordinator**: Alert generation and management
- **UIDataManager**: UI-optimized data structures

**Callbacks (Interactive Layer)**
- `filter_callbacks.py`: Dynamic filtering and data selection
- `table_callbacks.py`: DataTable interactions and conditional formatting
- `subject_detail_callbacks.py`: Subject-specific analysis views
- `tooltip_callbacks.py`: Contextual information display
- `visualization_callbacks.py`: Interactive plots and charts

---

## User Interface Features

### Conditional Row Highlighting

The DataTable implements comprehensive conditional formatting based on performance alerts:

#### Color Schema & Alert Categories
- **🟠 Orange Tones** (Performance Issues):
  - **SB (Severely Below)**: Dark orange with white text (< 6.5 percentile)
  - **B (Below)**: Light orange with white text (6.5-28 percentile)

- **🔵 Blue Tones** (Good Performance):
  - **G (Good)**: Light blue with white text (72-93.5 percentile)  
  - **SG (Severely Good)**: Dark blue with white text (> 93.5 percentile)

- **🟤 Brown Tone** (Threshold Alerts):
  - **T (Threshold Only)**: Brown highlighting for session count/water intake alerts

- **Combined Alerts**: Enhanced styling with darker colors and borders for subjects with multiple alert types (e.g., "SB, T")

#### Visual Elements
- **Severity Mapping**: Color brightness indicates severity (darker = more severe)
- **Certainty Score Color Coding**: Opacity/saturation reflects statistical confidence in percentile estimates
- **Border Indicators**: Left border in alert color for visual consistency
- **Enhanced Key Columns**: Alert-related columns receive stronger highlighting
- **Hover Effects**: Smooth transitions and elevated appearance

### Interactive Components

#### Subject Detail Views
- **Time Series Analysis**: Historical performance trends with rolling averages
- **Session Metadata**: Detailed session information and experimental parameters
- **Strata Context**: Subject positioning within experimental condition groups

#### Filtering & Search
- **Dynamic Filtering**: Real-time data filtering by alert category, strata, date ranges
- **Multi-Selection**: Multiple subjects, experimental conditions, performance categories
- **Search Integration**: Subject ID and experimental parameter search

#### Tooltips & Context
- **Statistical Context**: Hover details for percentile calculations and confidence intervals
- **Alert Explanations**: Detailed reasoning for alert generation
- **Experimental Context**: Strata information and experimental parameter details

---

## Testing Architecture

### Comprehensive Test Structure

```
tests/
├── conftest.py               # Shared fixtures and realistic test data
├── fixtures/                 # Real app data for testing
│   └── sample_data.py       # Extracted from actual dashboard data
├── unit/                    # Core component testing
│   ├── test_core_components.py    # **MAIN** - PercentileCoordinator, AlertCoordinator, EnhancedDataLoader
│   ├── test_callback_logic/       # Dash callback integration
│   ├── test_statistical_analysis/ # Statistical utilities and calculations
│   ├── test_ui_components/        # UI logic and data formatting
│   └── test_utilities/            # Helper functions and utilities
└── e2e/                     # End-to-end integration tests
    └── test_app_smoke.py    # App startup and basic functionality
```

### Testing Philosophy

- **Realistic Data**: Uses actual app data patterns and strata formats
- **Core Functionality Focus**: Prioritizes essential behavior over edge cases
- **Consolidated Approach**: Main functionality tested in `test_core_components.py`
- **Mock External Dependencies**: Clean isolation of testable components

### Running Tests

```bash
# Quick core functionality tests
pytest tests/unit/test_core_components.py -v

# Full unit test suite
pytest tests/unit/ -v

# End-to-end tests (note: app startup takes 5+ minutes)
pytest tests/e2e/test_app_smoke.py -v

# Complete test suite
pytest tests/ -v
```

---

## Development Guidelines & Contribution

### Code Quality Standards

The project uses pre-configured tools for consistent code quality:

- **black**: Code formatting (88 character line length)
- **isort**: Import sorting with project-specific configurations
- **flake8**: Linting with custom rules
- **mypy**: Type checking for enhanced code reliability
- **vulture**: Dead code detection

### Development Workflow

#### 1. **Code Contribution Process**

**Branch Strategy**:
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes following code standards
# Run pre-commit checks
black .
isort .
flake8
mypy .

# Run tests
pytest tests/unit/test_core_components.py -v
```

**Pull Request Requirements**:
- [ ] All tests pass
- [ ] Code follows black/isort formatting
- [ ] Type hints added for new functions
- [ ] Updated tests for new functionality
- [ ] Documentation updated if needed

#### 2. **Adding New Features**

**Statistical Components**:
- Add new analysis in `app_utils/app_analysis/`
- Integrate with `DataPipelineManager`
- Add corresponding tests in `test_statistical_analysis/`

**UI Components**:
- New elements in `app_elements/`
- Callbacks in `callbacks/`
- UI tests in `test_ui_components/`

**Alert Logic**:
- Extend `AlertCoordinator` or `AlertService`
- Update alert classification logic
- Add tests in `test_core_components.py`

#### 3. **Testing New Code**

**Test Location Guidelines**:
- Core functionality → `test_core_components.py`
- Statistical calculations → `test_statistical_analysis/`
- UI behavior → `test_ui_components/`
- Callback interactions → `test_callback_logic/`

**Use Realistic Fixtures**:
```python
def test_your_feature(sample_session_data):
    # Use provided realistic data
    result = your_function(sample_session_data)
    assert result.meets_expectations()
```

### Configuration Management

#### Alert Thresholds
Percentile thresholds are configurable for different sensitivity requirements:

```python
config = {
    "percentile_categories": {
        "severely_below": 6.5,
        "below": 28,
        "above": 72,
        "severely_above": 93.5
    }
}
```

#### Statistical Parameters
Key parameters can be adjusted:
- **Outlier detection factor**: IQR multiplier (default: 1.5)
- **Minimum sessions**: Eligibility threshold (default: 5)
- **Rolling average decay**: Exponential weighting factor
- **Confidence level**: Wilson CI confidence (default: 95%)

---

## Performance & Deployment

### Caching Strategy
- **Multi-tier caching**: Raw data, processed data, UI-optimized structures
- **Intelligent invalidation**: Automatic cache refresh on data updates
- **Memory optimization**: Compressed storage for large datasets

### Scalability Considerations
- **Optimized data structures**: UI-specific data formatting
- **Lazy loading**: On-demand computation for detailed views
- **Efficient filtering**: Pre-computed indices for fast subset operations

### Future Deployment
This dashboard is designed for **public server hosting** with:
- Clean separation of computation and presentation layers
- Stateless design for multi-user access
- Configurable data refresh intervals
- Security considerations for public access

---

## License

Licensed under the terms specified in the LICENSE file.

## Support

For questions about statistical methodology, experimental design, or technical implementation, please refer to the comprehensive test suite and inline documentation throughout the codebase.