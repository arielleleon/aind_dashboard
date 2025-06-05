"""
Shared pytest configuration and fixtures for AIND Dashboard tests

This module provides common fixtures and configuration that can be used
across all test modules, regardless of future refactoring.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def sample_session_data():
    """
    Fixture providing sample session data for testing
    
    This mimics the structure of real session data but with predictable values
    for testing statistical functions and data processing operations.
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

@pytest.fixture
def sample_table_data():
    """
    Fixture providing sample table display data for UI testing
    """
    return pd.DataFrame({
        'Subject ID': ['S001', 'S002', 'S003'],
        'Total Sessions': [2, 2, 1],
        'Avg Performance': [0.775, 0.675, 0.85],
        'Avg Reaction Time': [0.475, 0.575, 0.4],
        'Last Session': ['2024-01-02', '2024-01-02', '2024-01-01']
    })

@pytest.fixture
def mock_app_utils():
    """
    Fixture providing a mock AppUtils instance for testing
    
    This allows testing of functions that depend on AppUtils without
    requiring the full data loading pipeline.
    """
    mock_utils = Mock()
    mock_utils.get_session_data.return_value = pd.DataFrame()
    mock_utils.process_data_pipeline.return_value = pd.DataFrame()
    mock_utils.get_table_display_data.return_value = pd.DataFrame()
    mock_utils._cache = {}
    return mock_utils

@pytest.fixture
def sample_statistical_data():
    """
    Fixture providing data specifically for testing statistical functions
    """
    np.random.seed(42)  # For reproducible tests
    return {
        'values': np.random.normal(100, 15, 1000),
        'percentiles': [25, 50, 75, 90, 95],
        'bootstrap_samples': 1000,
        'confidence_level': 0.95
    }

@pytest.fixture(scope="session")
def app_instance():
    """
    Session-scoped fixture for Dash app instance used in E2E tests
    
    This fixture creates the app once per test session to speed up E2E testing.
    """
    # Import here to avoid circular imports during unit testing
    from app import app
    return app 