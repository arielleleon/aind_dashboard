"""
Test fixtures package for AIND Dashboard

This package provides centralized test fixtures and sample data
for all test modules in the AIND Dashboard project.
"""

from .sample_data import (
    SampleDataProvider,
    sample_data_provider,
    get_realistic_session_data,
    get_simple_session_data,
    get_strata_test_cases,
    get_table_display_data,
    get_statistical_data
)

__all__ = [
    'SampleDataProvider',
    'sample_data_provider', 
    'get_realistic_session_data',
    'get_simple_session_data',
    'get_strata_test_cases',
    'get_table_display_data',
    'get_statistical_data'
] 