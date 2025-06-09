"""
Unit tests for enhanced StatisticalUtils module

This module tests all statistical utility functions including the new bootstrap-related
methods that were added during the Module 5 refactoring.

Tests cover:
- Existing statistical methods (confidence intervals, outlier detection, etc.)
- New bootstrap methods (session CIs, distributions, coverage stats, etc.)  
- Error handling and edge cases
- Integration with external dependencies (bootstrap manager, cache manager, etc.)

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


class TestBootstrapMethods:
    """Tests for new bootstrap-related methods"""
    
    def test_generate_bootstrap_reference_distribution(self):
        """Test bootstrap reference distribution generation with realistic strata data"""
        # Use realistic strata reference data from fixtures
        stat_data = get_statistical_data()
        strata_name = list(stat_data['strata_reference'].keys())[0]
        reference_data = stat_data['strata_reference'][strata_name]
        
        result = StatisticalUtils.generate_bootstrap_reference_distribution(
            reference_data=reference_data,
            n_bootstrap=100,  # Reduced for faster testing
            random_state=42
        )
        
        # Check result structure
        assert isinstance(result, dict)
        assert 'bootstrap_enabled' in result
        assert 'percentile_grid' in result
        assert 'percentile_values' in result
        
        # Should be enabled for sufficient data
        assert result['bootstrap_enabled'] == True
        assert len(result['percentile_grid']) > 0
        assert len(result['percentile_values']) > 0
        assert len(result['percentile_grid']) == len(result['percentile_values'])
        
    def test_generate_bootstrap_reference_distribution_insufficient_data(self):
        """Test bootstrap distribution with insufficient data"""
        # Create insufficient data
        reference_data = np.array([1, 2])
        
        result = StatisticalUtils.generate_bootstrap_reference_distribution(
            reference_data=reference_data,
            n_bootstrap=100
        )
        
        # Should be disabled for insufficient data
        assert result['bootstrap_enabled'] == False
        assert 'error' in result
        assert len(result['percentile_grid']) == 0
        
    def test_calculate_bootstrap_raw_value_ci(self):
        """Test bootstrap confidence interval for realistic rolling averages"""
        # Use realistic rolling average data from fixtures
        session_data = get_realistic_session_data()
        reference_data = session_data['finished_trials_processed_rolling_avg'].values.astype(float)
        target_value = reference_data[0]  # Use first rolling average as target
        
        lower, upper = StatisticalUtils.calculate_bootstrap_raw_value_ci(
            reference_data=reference_data,
            target_value=target_value,
            confidence_level=0.95,
            n_bootstrap=100,  # Reduced for faster testing
            random_state=42
        )
        
        # Should return valid bounds
        assert not np.isnan(lower)
        assert not np.isnan(upper)
        assert lower < upper
        # Target should typically be within CI bounds (but not guaranteed due to bootstrap variance)
        
    def test_calculate_bootstrap_raw_value_ci_insufficient_data(self):
        """Test bootstrap CI with insufficient reference data"""
        # Create insufficient data
        reference_data = np.array([1, 2, 3])
        target_value = 2.0
        
        lower, upper = StatisticalUtils.calculate_bootstrap_raw_value_ci(
            reference_data=reference_data,
            target_value=target_value
        )
        
        # Should return NaN for insufficient data
        assert np.isnan(lower)
        assert np.isnan(upper)
        
    def test_validate_percentile_monotonicity(self):
        """Test percentile monotonicity validation with realistic percentile grids"""
        # Create valid monotonic percentiles using fixture percentiles
        stat_data = get_statistical_data()
        percentile_grid = np.array(stat_data['percentiles'])
        percentile_values = np.array([10, 25, 45, 70, 85])  # Monotonic values
        
        result = StatisticalUtils.validate_percentile_monotonicity(
            percentile_values, percentile_grid
        )
        
        # Should pass validation
        assert result['monotonicity_valid'] == True
        assert result['overall_monotonic'] == True
        assert len(result['violations']) == 0
        
    def test_validate_percentile_monotonicity_violations(self):
        """Test percentile monotonicity validation with violations"""
        # Create non-monotonic percentiles
        percentile_values = np.array([10, 30, 20, 40, 50])  # 30 > 20 is violation
        percentile_grid = np.array([10, 25, 50, 75, 90])
        
        result = StatisticalUtils.validate_percentile_monotonicity(
            percentile_values, percentile_grid
        )
        
        # Should detect violations
        assert result['overall_monotonic'] == False
        assert len(result['violations']) > 0
        
    def test_compare_bootstrap_vs_standard(self):
        """Test comparison between bootstrap and standard percentile calculations"""
        # Use realistic strata reference data
        stat_data = get_statistical_data()
        strata_name = list(stat_data['strata_reference'].keys())[0]
        reference_data = stat_data['strata_reference'][strata_name]
        
        # Create mock bootstrap distribution with realistic structure
        bootstrap_distribution = {
            'bootstrap_enabled': True,
            'percentile_grid': np.array(stat_data['percentiles']),
            'percentile_values': np.percentile(reference_data, stat_data['percentiles'])
        }
        
        result = StatisticalUtils.compare_bootstrap_vs_standard(
            reference_data, bootstrap_distribution
        )
        
        # Should return comparison results
        assert isinstance(result, dict)
        assert 'bootstrap_available' in result
        assert result['bootstrap_available'] == True
        assert 'percentile_comparisons' in result
        assert 'quality_metrics' in result
        assert 'recommendation' in result


class TestBootstrapIntegrationMethods:
    """Tests for bootstrap methods that integrate with external dependencies"""
    
    def test_calculate_session_bootstrap_cis_no_manager(self):
        """Test session bootstrap CI calculation without bootstrap manager"""
        # Use realistic session data from fixtures
        session_data = get_realistic_session_data()
        
        result = StatisticalUtils.calculate_session_bootstrap_cis(
            session_data=session_data,
            bootstrap_manager=None
        )
        
        # Should return original data when no manager available
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(session_data)
        assert list(result.columns) == list(session_data.columns)
        
    def test_calculate_session_bootstrap_cis_with_mocks(self):
        """Test session bootstrap CI calculation with mocked dependencies"""
        # Use realistic session data from fixtures
        session_data = get_realistic_session_data()
        
        # Create mock bootstrap manager
        mock_bootstrap_manager = sample_data_provider.create_mock_bootstrap_manager()
        
        # Create mock reference processor with realistic features
        mock_reference_processor = Mock()
        mock_reference_processor.features_config = {
            feature: False for feature in sample_data_provider.features  # Use real feature names
        }
        
        # Create mock quantile analyzer with realistic strata data
        mock_quantile_analyzer = Mock()
        # Create realistic strata data using actual feature columns
        mock_strata_data = pd.DataFrame({
            f"{feature}_processed": np.random.normal(0, 1, 50)
            for feature in sample_data_provider.features
        })
        mock_quantile_analyzer.percentile_data = {
            sample_data_provider.real_strata[0]: mock_strata_data
        }
        
        result = StatisticalUtils.calculate_session_bootstrap_cis(
            session_data=session_data,
            bootstrap_manager=mock_bootstrap_manager,
            reference_processor=mock_reference_processor,
            quantile_analyzer=mock_quantile_analyzer
        )
        
        # Should process successfully
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(session_data)
        
    def test_generate_bootstrap_distributions_no_manager(self):
        """Test bootstrap distribution generation without bootstrap manager"""
        result = StatisticalUtils.generate_bootstrap_distributions(
            bootstrap_manager=None
        )
        
        # Should return error for no manager
        assert isinstance(result, dict)
        assert 'error' in result
        assert result['bootstrap_enabled_count'] == 0
        
    def test_generate_bootstrap_distributions_with_mocks(self):
        """Test bootstrap distribution generation with mocked dependencies"""
        # Create mocks using realistic strata names
        mock_bootstrap_manager = Mock()
        mock_bootstrap_manager.generate_bootstrap_for_all_strata.return_value = {
            'bootstrap_enabled_count': 2,
            'total_strata': len(sample_data_provider.real_strata)
        }
        
        mock_quantile_analyzer = Mock()
        # Use realistic strata names and feature data
        mock_quantile_analyzer.stratified_data = {
            strata: pd.DataFrame({
                feature: np.random.normal(50, 15, 100)
                for feature in sample_data_provider.features
            })
            for strata in sample_data_provider.real_strata[:2]  # Use first 2 strata
        }
        
        mock_reference_processor = Mock()
        
        # Create properly mocked cache manager using fixture
        mock_cache_manager = sample_data_provider.create_mock_cache_manager()
        
        result = StatisticalUtils.generate_bootstrap_distributions(
            bootstrap_manager=mock_bootstrap_manager,
            quantile_analyzer=mock_quantile_analyzer,
            reference_processor=mock_reference_processor,
            cache_manager=mock_cache_manager
        )
        
        # Should call bootstrap manager
        assert isinstance(result, dict)
        mock_bootstrap_manager.generate_bootstrap_for_all_strata.assert_called_once()
        
    def test_get_bootstrap_coverage_stats_no_cache(self):
        """Test bootstrap coverage stats without cache manager"""
        result = StatisticalUtils.get_bootstrap_coverage_stats(cache_manager=None)
        
        # Should return empty dict
        assert isinstance(result, dict)
        assert len(result) == 0
        
    def test_get_bootstrap_coverage_stats_with_cache(self):
        """Test bootstrap coverage stats with mocked cache manager"""
        # Create mock cache manager with realistic strata coverage data
        mock_cache_manager = Mock()
        mock_cache_manager.has.return_value = True
        mock_cache_manager.get.return_value = {
            sample_data_provider.real_strata[0]: {'coverage': 0.85},
            sample_data_provider.real_strata[1]: {'coverage': 0.92}
        }
        
        result = StatisticalUtils.get_bootstrap_coverage_stats(
            cache_manager=mock_cache_manager,
            use_cache=True
        )
        
        # Should return cached data
        assert isinstance(result, dict)
        assert sample_data_provider.real_strata[0] in result
        mock_cache_manager.has.assert_called_with('bootstrap_coverage_stats')
        
    def test_get_bootstrap_enabled_strata_no_cache(self):
        """Test bootstrap enabled strata without cache manager"""
        result = StatisticalUtils.get_bootstrap_enabled_strata(cache_manager=None)
        
        # Should return empty set
        assert isinstance(result, set)
        assert len(result) == 0
        
    def test_get_bootstrap_enabled_strata_with_cache(self):
        """Test bootstrap enabled strata with mocked cache manager"""
        # Create mock cache manager with realistic strata names
        mock_cache_manager = Mock()
        mock_cache_manager.has.side_effect = lambda key: key == 'optimized_storage'
        mock_cache_manager.get.return_value = {
            'metadata': {
                'bootstrap_enabled_strata_list': sample_data_provider.real_strata[:3]
            }
        }
        
        result = StatisticalUtils.get_bootstrap_enabled_strata(
            cache_manager=mock_cache_manager,
            use_cache=True
        )
        
        # Should return realistic strata set
        assert isinstance(result, set)
        for strata in sample_data_provider.real_strata[:3]:
            assert strata in result
        assert len(result) == 3
        
    def test_get_bootstrap_enhancement_summary_no_data(self):
        """Test bootstrap enhancement summary without session data"""
        # Create mock cache manager with no data
        mock_cache_manager = Mock()
        mock_cache_manager.has.return_value = False
        
        result = StatisticalUtils.get_bootstrap_enhancement_summary(
            cache_manager=mock_cache_manager
        )
        
        # Should return error
        assert isinstance(result, dict)
        assert 'error' in result
        
    def test_get_bootstrap_enhancement_summary_with_data(self):
        """Test bootstrap enhancement summary with realistic session data"""
        # Use realistic session data from fixtures
        session_data = get_realistic_session_data()
        
        # Create mocks with realistic configurations
        mock_cache_manager = Mock()
        mock_cache_manager.has.return_value = True
        mock_cache_manager.get.return_value = session_data
        
        mock_bootstrap_manager = sample_data_provider.create_mock_bootstrap_manager()
        
        mock_reference_processor = Mock()
        mock_reference_processor.features_config = {
            feature: False for feature in sample_data_provider.features
        }
        
        result = StatisticalUtils.get_bootstrap_enhancement_summary(
            cache_manager=mock_cache_manager,
            bootstrap_manager=mock_bootstrap_manager,
            reference_processor=mock_reference_processor
        )
        
        # Should return complete summary with realistic structure
        assert isinstance(result, dict)
        assert 'total_sessions' in result
        assert 'total_subjects' in result
        assert 'feature_enhancement' in result
        assert 'overall_enhancement' in result
        assert result['total_sessions'] == len(session_data)
        assert result['total_subjects'] == session_data['subject_id'].nunique()


class TestEdgeCasesAndErrorHandling:
    """Tests for edge cases and error handling"""
    
    def test_percentile_ci_with_all_nans(self):
        """Test percentile CI calculation with all NaN values"""
        values = np.array([np.nan, np.nan, np.nan])
        
        lower, upper = StatisticalUtils.calculate_percentile_confidence_interval(values, 50)
        
        assert np.isnan(lower)
        assert np.isnan(upper)
        
    def test_outlier_detection_with_identical_values(self):
        """Test outlier detection when all values are identical"""
        data = np.array([5, 5, 5, 5, 5])
        
        outlier_mask, weights = StatisticalUtils.detect_outliers_iqr(data)
        
        # No outliers should be detected
        assert np.all(outlier_mask == False)
        assert np.all(weights == 1.0)
        
    def test_bootstrap_ci_with_nan_target(self):
        """Test bootstrap CI calculation with NaN target value"""
        # Use realistic reference data from fixtures
        session_data = get_realistic_session_data()
        reference_data = session_data['total_trials'].values.astype(float)[:5]
        target_value = np.nan
        
        lower, upper = StatisticalUtils.calculate_bootstrap_raw_value_ci(
            reference_data, target_value
        )
        
        assert np.isnan(lower)
        assert np.isnan(upper)
        
    def test_weighted_percentile_rank_empty_reference(self):
        """Test weighted percentile rank with empty reference data"""
        reference_values = np.array([])
        reference_weights = np.array([])
        target_value = 5.0
        
        rank = StatisticalUtils.calculate_weighted_percentile_rank(
            reference_values, reference_weights, target_value
        )
        
        assert np.isnan(rank)
        
    def test_bootstrap_distribution_with_weights(self):
        """Test bootstrap distribution generation with realistic outlier weights"""
        # Use realistic data with some outlier weights
        session_data = get_realistic_session_data()
        reference_data = session_data['foraging_performance'].values.astype(float)
        reference_weights = session_data['outlier_weight'].values.astype(float)
        
        result = StatisticalUtils.generate_bootstrap_reference_distribution(
            reference_data=reference_data,
            reference_weights=reference_weights,
            n_bootstrap=50,  # Reduced for faster testing
            random_state=42
        )
        
        # Should handle weights properly
        assert result['bootstrap_enabled'] == True
        assert len(result['percentile_values']) > 0


class TestParameterValidation:
    """Tests for parameter validation and bounds checking"""
    
    def test_confidence_level_bounds(self):
        """Test confidence level parameter bounds with realistic data"""
        # Use realistic statistical data from fixtures
        stat_data = get_statistical_data()
        values = stat_data['values'][:50]  # Use subset for faster testing
        
        # Test valid confidence levels
        for cl in [0.90, 0.95, 0.99]:
            lower, upper = StatisticalUtils.calculate_percentile_confidence_interval(
                values, percentile=50, confidence_level=cl
            )
            assert not np.isnan(lower)
            assert not np.isnan(upper)
            
    def test_percentile_bounds(self):
        """Test percentile parameter bounds with realistic statistical data"""
        # Use realistic statistical data from fixtures
        stat_data = get_statistical_data()
        values = stat_data['values'][:50]  # Use subset for faster testing
        
        # Test edge percentiles
        for p in [1, 50, 99]:
            lower, upper = StatisticalUtils.calculate_percentile_confidence_interval(
                values, percentile=p, confidence_level=0.95
            )
            assert not np.isnan(lower)
            assert not np.isnan(upper)
            assert 0 <= lower <= 100
            assert 0 <= upper <= 100
            
    def test_bootstrap_samples_parameter(self):
        """Test bootstrap samples parameter with realistic strata data"""
        # Use realistic strata reference data from fixtures
        stat_data = get_statistical_data()
        strata_name = list(stat_data['strata_reference'].keys())[0]
        reference_data = stat_data['strata_reference'][strata_name]
        
        # Test different bootstrap sample counts
        for n_bootstrap in [10, 50, 100]:
            result = StatisticalUtils.generate_bootstrap_reference_distribution(
                reference_data=reference_data,
                n_bootstrap=n_bootstrap,
                random_state=42
            )
            assert result['bootstrap_enabled'] == True
            assert result['bootstrap_samples'] <= n_bootstrap 