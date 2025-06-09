"""
Shared pytest configuration and fixtures for AIND Dashboard tests

This module provides common fixtures and configuration that can be used
across all test modules, regardless of future refactoring.

Updated to use realistic sample data extracted from the actual app.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the new realistic fixtures
from tests.fixtures import (
    get_realistic_session_data,
    get_simple_session_data,
    get_table_display_data,
    get_statistical_data,
    sample_data_provider
)

@pytest.fixture
def sample_session_data():
    """
    Fixture providing realistic sample session data for testing
    
    This uses real app data structure with actual column names,
    strata formats, and data types for accurate testing.
    """
    return get_realistic_session_data()

@pytest.fixture
def simple_session_data():
    """
    Fixture providing simple session data for lightweight tests
    
    This provides minimal data for tests that don't need the full structure.
    """
    return get_simple_session_data()

@pytest.fixture
def sample_table_data():
    """
    Fixture providing realistic table display data for UI testing
    
    Updated to use real app data structure and strata formats.
    """
    table_data = get_table_display_data()
    # Convert to DataFrame format for backward compatibility
    return pd.DataFrame(table_data)

@pytest.fixture
def mock_app_utils():
    """
    Fixture providing a mock AppUtils instance for testing
    
    This allows testing of functions that depend on AppUtils without
    requiring the full data loading pipeline.
    """
    mock_utils = Mock()
    mock_utils.get_session_data.return_value = get_realistic_session_data()
    mock_utils.process_data_pipeline.return_value = get_realistic_session_data()
    mock_utils.get_table_display_data.return_value = get_table_display_data()
    mock_utils._cache = {}
    
    # Add UI data manager mock for delegation tests
    mock_utils.ui_data_manager = Mock()
    mock_utils.ui_data_manager.get_strata_abbreviation.return_value = 'UWBB1'
    mock_utils.ui_data_manager.optimize_session_data_storage.return_value = {
        'subjects': {},
        'strata_reference': {},
        'metadata': {'total_subjects': 5}
    }
    
    return mock_utils

@pytest.fixture
def sample_statistical_data():
    """
    Fixture providing data specifically for testing statistical functions
    
    Updated to include real feature names and strata references.
    """
    return get_statistical_data()

@pytest.fixture
def real_strata_data():
    """
    Fixture providing real strata names and abbreviations for testing
    
    This ensures tests use the actual strata formats from the app.
    """
    return {
        'strata_names': sample_data_provider.real_strata,
        'abbreviations': sample_data_provider.real_strata_abbreviations,
        'features': sample_data_provider.features
    }

@pytest.fixture
def mock_bootstrap_manager():
    """
    Fixture providing a mock bootstrap manager for testing
    """
    return sample_data_provider.create_mock_bootstrap_manager()

@pytest.fixture
def mock_cache_manager():
    """
    Fixture providing a mock cache manager for testing
    """
    return sample_data_provider.create_mock_cache_manager()

@pytest.fixture(scope="session")
def app_instance():
    """
    Session-scoped fixture for Dash app instance used in E2E tests
    
    This fixture creates the app once per test session to speed up E2E testing.
    """
    # Import here to avoid circular imports during unit testing
    from app import app
    return app

# Legacy fixtures for backward compatibility
@pytest.fixture
def legacy_sample_session_data():
    """
    DEPRECATED: Legacy fixture for backward compatibility
    
    Use sample_session_data instead for realistic app data.
    """
    return pd.DataFrame({
        'subject_id': ['S001', 'S001', 'S002', 'S002', 'S003'],
        'session_id': ['sess_1', 'sess_2', 'sess_1', 'sess_2', 'sess_1'],
        'performance_metric': [0.75, 0.80, 0.65, 0.70, 0.85],
        'reaction_time': [0.5, 0.45, 0.6, 0.55, 0.4],
        'session_date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-01', '2024-01-02', '2024-01-01']),
        'training_day': [1, 2, 1, 2, 1],
        'total_trials': [100, 120, 90, 110, 130]
    }) 