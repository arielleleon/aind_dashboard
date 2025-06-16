"""
Unit tests for enhanced StatisticalUtils module

This module tests all statistical utility functions including Wilson confidence intervals
and percentile analysis methods that are used in the AIND Dashboard.

Tests cover:
- Statistical methods (confidence intervals, outlier detection, etc.)
- Wilson CI methodology for uncertainty quantification
- Error handling and edge cases
- Integration with external dependencies (cache manager, etc.)

Uses realistic fixtures from sample_data.py to ensure tests match actual app data structures.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Ensure we can import from the project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from app_utils.app_analysis.statistical_utils import StatisticalUtils
from tests.fixtures.sample_data import (
    get_realistic_session_data, 
    get_simple_session_data, 
    get_statistical_data,
    sample_data_provider
)


class TestBasicStatisticalMethods:
    """Tests for existing statistical utility methods"""
    
    def test_calculate_percentile_confidence_interval_normal_case(self):
        """Test percentile confidence interval calculation with realistic statistical data"""
        # Use fixture data
        stat_data = get_statistical_data()
        values = stat_data['values']
        
        # Calculate CI for 50th percentile
        lower, upper = StatisticalUtils.calculate_percentile_confidence_interval(
            values, percentile=50, confidence_level=stat_data['confidence_level']
        )
        
        # Should return valid bounds
        assert not np.isnan(lower)
        assert not np.isnan(upper)
        assert lower < upper
        assert 0 <= lower <= 100
        assert 0 <= upper <= 100
        
    def test_calculate_percentile_confidence_interval_insufficient_data(self):
        """Test percentile CI calculation with insufficient data"""
        # Test with too few values
        values = np.array([1, 2])
        
        lower, upper = StatisticalUtils.calculate_percentile_confidence_interval(
            values, percentile=50, confidence_level=0.95
        )
        
        # Should return NaN for insufficient data
        assert np.isnan(lower)
        assert np.isnan(upper)
        
    def test_calculate_percentile_confidence_interval_with_nans(self):
        """Test percentile CI calculation with NaN values"""
        # Use fixture data and add NaN values
        stat_data = get_statistical_data()
        values = stat_data['values'][:20].copy()  # Use subset for faster test
        values[::5] = np.nan  # Add NaN values at regular intervals
        
        lower, upper = StatisticalUtils.calculate_percentile_confidence_interval(
            values, percentile=50, confidence_level=0.95
        )
        
        # Should handle NaN values and return valid bounds
        assert not np.isnan(lower)
        assert not np.isnan(upper)
        assert lower < upper
        
    def test_detect_outliers_iqr_normal_case(self):
        """Test IQR outlier detection with realistic feature values"""
        # Use realistic finished_trials data from fixtures
        session_data = get_realistic_session_data()
        data = session_data['finished_trials'].values.astype(float)
        
        outlier_mask, weights = StatisticalUtils.detect_outliers_iqr(data, factor=1.5)
        
        # Check results
        assert len(outlier_mask) == len(data)
        assert len(weights) == len(data)
        assert isinstance(outlier_mask, np.ndarray)
        assert isinstance(weights, np.ndarray)
        # All weights should be between 0 and 1
        assert np.all((weights >= 0) & (weights <= 1))
        
    def test_detect_outliers_iqr_insufficient_data(self):
        """Test IQR outlier detection with insufficient data"""
        # Test with too few values
        data = np.array([1, 2, 3])
        
        outlier_mask, weights = StatisticalUtils.detect_outliers_iqr(data)
        
        # Should return no outliers for insufficient data
        assert len(outlier_mask) == len(data)
        assert len(weights) == len(data)
        assert np.all(outlier_mask == False)
        assert np.all(weights == 1.0)
        
    def test_detect_outliers_modified_zscore(self):
        """Test modified Z-score outlier detection with realistic data"""
        # Use realistic foraging_performance data from fixtures
        session_data = get_realistic_session_data()
        data = session_data['foraging_performance'].values.astype(float)
        
        outlier_mask, weights = StatisticalUtils.detect_outliers_modified_zscore(data, threshold=3.5)
        
        # Check results
        assert len(outlier_mask) == len(data)
        assert len(weights) == len(data)
        assert isinstance(outlier_mask, np.ndarray)
        assert isinstance(weights, np.ndarray)
        # All weights should be between 0 and 1
        assert np.all((weights >= 0) & (weights <= 1))
        
    def test_calculate_weighted_percentile_rank(self):
        """Test weighted percentile rank calculation with realistic feature data"""
        # Use realistic ignore_rate data from fixtures
        session_data = get_realistic_session_data()
        reference_values = session_data['ignore_rate'].values.astype(float)
        reference_weights = np.ones(len(reference_values))
        target_value = np.median(reference_values)  # Use median as target
        
        rank = StatisticalUtils.calculate_weighted_percentile_rank(
            reference_values, reference_weights, target_value
        )
        
        # Should return valid percentile rank
        assert not np.isnan(rank)
        assert 0 <= rank <= 100
        # Should be around 50th percentile since we used median
        assert 35 <= rank <= 65
        
    def test_calculate_weighted_percentile_rank_with_weights(self):
        """Test weighted percentile rank with realistic outlier weights"""
        # Use realistic data with outlier weights from fixtures
        session_data = get_realistic_session_data()
        reference_values = session_data['abs(bias_naive)'].values.astype(float)
        reference_weights = session_data['outlier_weight'].values.astype(float)
        target_value = reference_values[0]  # Use first value as target
        
        rank = StatisticalUtils.calculate_weighted_percentile_rank(
            reference_values, reference_weights, target_value
        )
        
        # Should return valid percentile rank influenced by weights
        assert not np.isnan(rank)
        assert 0 <= rank <= 100


class TestEdgeCasesAndErrorHandling:
    """Tests for edge cases and error handling"""
    
    def test_percentile_ci_with_all_nans(self):
        """Test percentile CI calculation with all NaN values"""
        values = np.array([np.nan, np.nan, np.nan])
        
        lower, upper = StatisticalUtils.calculate_percentile_confidence_interval(values, 50)
        
        assert np.isnan(lower)
        assert np.isnan(upper)
        
    def test_outlier_detection_with_identical_values(self):
        """Test outlier detection with identical values"""
        data = np.array([5, 5, 5, 5, 5])
        outlier_mask, outlier_weights = StatisticalUtils.detect_outliers_iqr(data)
        
        # No outliers expected in identical values
        assert np.all(outlier_mask == False)
        assert len(outlier_weights) == len(data)
        assert all(w == 1.0 for w in outlier_weights)

    def test_weighted_percentile_rank_empty_reference(self):
        """Test weighted percentile rank with empty reference data"""
        reference_data = np.array([])
        reference_weights = np.array([])
        test_value = 5.0
        
        rank = StatisticalUtils.calculate_weighted_percentile_rank(
            reference_data, reference_weights, test_value
        )
        
        # Should return NaN for empty data
        assert np.isnan(rank)


class TestParameterValidation:
    """Tests for parameter validation and bounds checking"""
    
    def test_confidence_level_bounds(self):
        """Test confidence level parameter validation"""
        # Edge case: very low confidence level
        data = np.random.normal(50, 10, 100)
        lower, upper = StatisticalUtils.calculate_percentile_confidence_interval(
            data, 50, confidence_level=0.01
        )
        
        # Should still return valid bounds, just very narrow
        assert not np.isnan(lower)
        assert not np.isnan(upper)
        assert lower < upper
        
        # Edge case: very high confidence level
        lower, upper = StatisticalUtils.calculate_percentile_confidence_interval(
            data, 50, confidence_level=0.999
        )
        
        # Should return valid bounds, very wide
        assert not np.isnan(lower)
        assert not np.isnan(upper)
        assert lower < upper
    
    def test_percentile_bounds(self):
        """Test percentile parameter bounds"""
        data = np.random.normal(50, 10, 100)
        
        # Test extreme percentiles
        for percentile in [0.1, 99.9]:
            lower, upper = StatisticalUtils.calculate_percentile_confidence_interval(
                data, percentile
            )
            assert not np.isnan(lower)
            assert not np.isnan(upper)
            assert 0 <= lower <= 100
            assert 0 <= upper <= 100


class TestHeatmapDataProcessing:
    """Test the new heatmap data processing functions extracted from UI component"""
    
    def test_validate_percentile_data_basic(self):
        """Test basic percentile data validation"""
        # Test with mixed valid and invalid data
        test_percentiles = [50.0, -1, 75.0, -1, 90.0, 25.0]
        validated = StatisticalUtils.validate_percentile_data(test_percentiles)
        
        expected = [50.0, np.nan, 75.0, np.nan, 90.0, 25.0]
        
        assert len(validated) == len(expected)
        for i, (actual, expect) in enumerate(zip(validated, expected)):
            if np.isnan(expect):
                assert np.isnan(actual), f"Position {i}: expected NaN, got {actual}"
            else:
                assert actual == expect, f"Position {i}: expected {expect}, got {actual}"
    
    def test_validate_percentile_data_custom_marker(self):
        """Test percentile validation with custom invalid marker"""
        test_percentiles = [50.0, -999, 75.0, -999, 90.0]
        validated = StatisticalUtils.validate_percentile_data(test_percentiles, invalid_marker=-999)
        
        expected = [50.0, np.nan, 75.0, np.nan, 90.0]
        
        assert len(validated) == len(expected)
        for i, (actual, expect) in enumerate(zip(validated, expected)):
            if np.isnan(expect):
                assert np.isnan(actual)
            else:
                assert actual == expect
    
    def test_validate_percentile_data_all_valid(self):
        """Test percentile validation with all valid data"""
        test_percentiles = [10.5, 25.0, 50.0, 75.0, 90.5]
        validated = StatisticalUtils.validate_percentile_data(test_percentiles)
        
        assert validated == test_percentiles
    
    def test_validate_percentile_data_all_invalid(self):
        """Test percentile validation with all invalid data"""
        test_percentiles = [-1, -1, -1, -1]
        validated = StatisticalUtils.validate_percentile_data(test_percentiles)
        
        assert all(np.isnan(v) for v in validated)
        assert len(validated) == 4
    
    def test_process_heatmap_matrix_data_basic(self):
        """Test basic heatmap matrix data processing"""
        # Setup test data
        features_config = {
            'finished_trials': False,
            'ignore_rate': True,
            'foraging_performance': False
        }
        
        time_series_data = {
            'finished_trials_percentiles': [50.0, 60.0, 70.0],
            'ignore_rate_percentiles': [30.0, -1, 40.0],
            'foraging_performance_percentiles': [80.0, 85.0, 90.0],
            'overall_percentiles': [55.0, 62.5, 67.5]
        }
        
        # Process the data
        heatmap_data, feature_names = StatisticalUtils.process_heatmap_matrix_data(
            time_series_data, features_config
        )
        
        # Verify structure
        assert len(heatmap_data) == 4  # 3 features + overall
        assert len(feature_names) == 4
        
        # Verify feature names are properly formatted
        expected_names = ['Finished Trials', 'Ignore Rate', 'Foraging Performance', 'Overall Percentile']
        assert feature_names == expected_names
        
        # Verify data integrity
        assert heatmap_data[0] == [50.0, 60.0, 70.0]  # finished_trials
        assert heatmap_data[1][0] == 30.0 and np.isnan(heatmap_data[1][1]) and heatmap_data[1][2] == 40.0  # ignore_rate with NaN
        assert heatmap_data[2] == [80.0, 85.0, 90.0]  # foraging_performance
        assert heatmap_data[3] == [55.0, 62.5, 67.5]  # overall
    
    def test_process_heatmap_matrix_data_missing_features(self):
        """Test heatmap processing when some features are missing from data"""
        features_config = {
            'finished_trials': False,
            'missing_feature': True,
            'ignore_rate': True
        }
        
        time_series_data = {
            'finished_trials_percentiles': [50.0, 60.0, 70.0],
            'ignore_rate_percentiles': [30.0, 35.0, 40.0],
            # missing_feature_percentiles is not present
        }
        
        heatmap_data, feature_names = StatisticalUtils.process_heatmap_matrix_data(
            time_series_data, features_config
        )
        
        # Should only include features that exist in data
        assert len(heatmap_data) == 2
        assert len(feature_names) == 2
        assert feature_names == ['Finished Trials', 'Ignore Rate']
    
    def test_process_heatmap_matrix_data_all_invalid_features(self):
        """Test heatmap processing when all feature data is invalid"""
        features_config = {
            'finished_trials': False,
            'ignore_rate': True
        }
        
        time_series_data = {
            'finished_trials_percentiles': [-1, -1, -1],
            'ignore_rate_percentiles': [-1, -1, -1],
        }
        
        heatmap_data, feature_names = StatisticalUtils.process_heatmap_matrix_data(
            time_series_data, features_config
        )
        
        # Should return empty lists when no valid data
        assert len(heatmap_data) == 0
        assert len(feature_names) == 0
    
    def test_process_heatmap_matrix_data_no_overall(self):
        """Test heatmap processing without overall percentiles"""
        features_config = {
            'finished_trials': False
        }
        
        time_series_data = {
            'finished_trials_percentiles': [50.0, 60.0, 70.0],
            # No overall_percentiles
        }
        
        heatmap_data, feature_names = StatisticalUtils.process_heatmap_matrix_data(
            time_series_data, features_config
        )
        
        assert len(heatmap_data) == 1
        assert len(feature_names) == 1
        assert feature_names == ['Finished Trials']
    
    def test_format_feature_display_name(self):
        """Test feature name formatting for display"""
        test_cases = [
            ('finished_trials', 'Finished Trials'),
            ('abs(bias_naive)', '|Bias Naive|'),
            ('ignore_rate', 'Ignore Rate'),
            ('total_trials', 'Total Trials'),
            ('foraging_performance', 'Foraging Performance'),
            ('simple', 'Simple'),
            ('complex_feature_name', 'Complex Feature Name')
        ]
        
        for input_name, expected in test_cases:
            result = StatisticalUtils.format_feature_display_name(input_name)
            assert result == expected, f"Expected '{expected}', got '{result}' for input '{input_name}'"
    
    def test_calculate_session_highlighting_coordinates(self):
        """Test session highlighting coordinate calculation"""
        sessions = [1, 3, 5, 7, 9, 11]
        
        # Test valid session
        result = StatisticalUtils.calculate_session_highlighting_coordinates(sessions, 5)
        assert result == 2  # Index of session 5
        
        # Test another valid session
        result = StatisticalUtils.calculate_session_highlighting_coordinates(sessions, 1)
        assert result == 0  # Index of session 1
        
        # Test session not in list
        result = StatisticalUtils.calculate_session_highlighting_coordinates(sessions, 6)
        assert result is None
        
        # Test with empty sessions list
        result = StatisticalUtils.calculate_session_highlighting_coordinates([], 5)
        assert result is None
    
    def test_calculate_session_highlighting_coordinates_edge_cases(self):
        """Test session highlighting with edge cases"""
        # Single session
        sessions = [1]
        result = StatisticalUtils.calculate_session_highlighting_coordinates(sessions, 1)
        assert result == 0
        
        # Large session numbers
        sessions = [100, 200, 300]
        result = StatisticalUtils.calculate_session_highlighting_coordinates(sessions, 200)
        assert result == 1

# ... existing code ... 