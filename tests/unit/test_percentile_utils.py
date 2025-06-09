"""
Unit Tests for PercentileCoordinator

This module tests the percentile coordination functionality extracted from AppUtils.
Tests cover percentile calculation coordination, data retrieval, and caching behavior.

REFACTORING: Created alongside Module 6 extraction to ensure reliable percentile coordination.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from app_utils.percentile_utils import PercentileCoordinator
from tests.fixtures.sample_data import get_realistic_session_data, get_simple_session_data


class TestPercentileCoordinator:
    """Test suite for PercentileCoordinator class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create mock dependencies
        self.mock_cache_manager = Mock()
        self.mock_pipeline_manager = Mock()
        
        # Initialize coordinator with mocks
        self.coordinator = PercentileCoordinator(
            cache_manager=self.mock_cache_manager,
            pipeline_manager=self.mock_pipeline_manager
        )
        
        # Set up sample data
        self.sample_session_data = get_realistic_session_data()
        self.simple_session_data = get_simple_session_data()
    
    def test_initialization(self):
        """Test PercentileCoordinator initialization"""
        # Test with dependencies
        coordinator = PercentileCoordinator(
            cache_manager=self.mock_cache_manager,
            pipeline_manager=self.mock_pipeline_manager
        )
        
        assert coordinator.cache_manager == self.mock_cache_manager
        assert coordinator.pipeline_manager == self.mock_pipeline_manager
        assert coordinator.percentile_calculator is not None
        
        # Test without dependencies
        standalone_coordinator = PercentileCoordinator()
        assert standalone_coordinator.cache_manager is None
        assert standalone_coordinator.pipeline_manager is None
        assert standalone_coordinator.percentile_calculator is not None
    
    def test_get_session_overall_percentiles_with_cache(self):
        """Test getting session percentiles using cached data"""
        # Mock cache has data
        self.mock_cache_manager.has.return_value = True
        self.mock_cache_manager.get.return_value = self.sample_session_data
        
        result = self.coordinator.get_session_overall_percentiles(use_cache=True)
        
        # Verify cache was checked
        self.mock_cache_manager.has.assert_called_with('session_level_data')
        self.mock_cache_manager.get.assert_called_with('session_level_data')
        
        # Verify result structure
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'subject_id' in result.columns
    
    def test_get_session_overall_percentiles_without_cache(self):
        """Test getting session percentiles without cache"""
        # Mock cache has no session-level data, but has raw data
        def mock_has_side_effect(key):
            if key == 'session_level_data':
                return False
            elif key == 'raw_data':
                return True
            return False
        
        def mock_get_side_effect(key):
            if key == 'raw_data':
                return self.sample_session_data
            return None
        
        self.mock_cache_manager.has.side_effect = mock_has_side_effect
        self.mock_cache_manager.get.side_effect = mock_get_side_effect
        
        # Mock pipeline manager to return processed data
        self.mock_pipeline_manager.process_data_pipeline.return_value = self.sample_session_data
        
        result = self.coordinator.get_session_overall_percentiles(use_cache=False)
        
        # Verify pipeline manager was called
        self.mock_pipeline_manager.process_data_pipeline.assert_called_once()
        
        # Verify result
        assert isinstance(result, pd.DataFrame)
    
    def test_get_session_overall_percentiles_with_subject_filter(self):
        """Test filtering percentiles by subject IDs"""
        self.mock_cache_manager.has.return_value = True
        self.mock_cache_manager.get.return_value = self.sample_session_data
        
        subject_ids = ['690494', '690486']
        result = self.coordinator.get_session_overall_percentiles(
            subject_ids=subject_ids,
            use_cache=True
        )
        
        # Verify filtering worked
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            unique_subjects = set(result['subject_id'].unique())
            assert unique_subjects.issubset(set(subject_ids))
    
    def test_get_session_level_data_from_cache(self):
        """Test retrieving session-level data from cache"""
        self.mock_cache_manager.has.return_value = True
        self.mock_cache_manager.get.return_value = self.sample_session_data
        
        result = self.coordinator._get_session_level_data(use_cache=True)
        
        # Verify cache access
        self.mock_cache_manager.has.assert_called_with('session_level_data')
        assert result.equals(self.sample_session_data)
    
    def test_get_session_level_data_from_pipeline(self):
        """Test retrieving session-level data from pipeline manager"""
        # Mock cache behavior: no session-level data, but has raw data
        def mock_has_side_effect(key):
            if key == 'session_level_data':
                return False
            elif key == 'raw_data':
                return True
            return False
        
        def mock_get_side_effect(key):
            if key == 'raw_data':
                return self.sample_session_data
            return None
        
        self.mock_cache_manager.has.side_effect = mock_has_side_effect
        self.mock_cache_manager.get.side_effect = mock_get_side_effect
        
        # Mock pipeline manager to return processed data
        self.mock_pipeline_manager.process_data_pipeline.return_value = self.sample_session_data
        
        result = self.coordinator._get_session_level_data(use_cache=True)
        
        # Verify pipeline processing was called
        self.mock_pipeline_manager.process_data_pipeline.assert_called_once()
        assert result.equals(self.sample_session_data)
    
    def test_get_session_level_data_no_data_available(self):
        """Test handling when no data is available"""
        # Cache has no data
        self.mock_cache_manager.has.return_value = False
        
        # Pipeline manager has no data
        self.mock_pipeline_manager.cache_manager.get.return_value = None
        
        result = self.coordinator._get_session_level_data(use_cache=True)
        
        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_calculate_percentiles_for_sessions_success(self):
        """Test successful percentile calculation for sessions"""
        # Mock percentile calculator methods
        self.coordinator.percentile_calculator.calculate_session_overall_percentile = Mock(
            return_value=self.sample_session_data
        )
        self.coordinator.percentile_calculator.calculate_session_overall_rolling_average = Mock(
            return_value=self.sample_session_data
        )
        
        result = self.coordinator.calculate_percentiles_for_sessions(self.simple_session_data)
        
        # Verify calculations were called
        self.coordinator.percentile_calculator.calculate_session_overall_percentile.assert_called_once()
        self.coordinator.percentile_calculator.calculate_session_overall_rolling_average.assert_called_once()
        
        assert isinstance(result, pd.DataFrame)
    
    def test_calculate_percentiles_for_sessions_empty_data(self):
        """Test percentile calculation with empty data"""
        empty_df = pd.DataFrame()
        result = self.coordinator.calculate_percentiles_for_sessions(empty_df)
        
        assert result.empty
    
    def test_calculate_percentiles_for_sessions_error_handling(self):
        """Test error handling in percentile calculation"""
        # Mock percentile calculator to raise exception
        self.coordinator.percentile_calculator.calculate_session_overall_percentile = Mock(
            side_effect=Exception("Calculation error")
        )
        
        result = self.coordinator.calculate_percentiles_for_sessions(self.simple_session_data)
        
        # Should return original data on error
        assert result.equals(self.simple_session_data)
    
    def test_get_percentiles_by_strata(self):
        """Test getting percentiles for specific strata"""
        self.mock_cache_manager.has.return_value = True
        self.mock_cache_manager.get.return_value = self.sample_session_data
        
        strata_name = 'Uncoupled Without Baiting_BEGINNER_v1'
        result = self.coordinator.get_percentiles_by_strata(strata_name)
        
        # Verify strata filtering
        if result is not None and not result.empty:
            assert all(result['strata'] == strata_name)
    
    def test_get_percentiles_by_strata_no_data(self):
        """Test getting percentiles for strata when no data available"""
        self.mock_cache_manager.has.return_value = False
        self.mock_pipeline_manager.cache_manager.get.return_value = None
        
        result = self.coordinator.get_percentiles_by_strata('NonexistentStrata')
        
        assert result is None
    
    def test_get_subject_percentile_history(self):
        """Test getting percentile history for a subject"""
        self.mock_cache_manager.has.return_value = True
        self.mock_cache_manager.get.return_value = self.sample_session_data
        
        subject_id = '690494'
        result = self.coordinator.get_subject_percentile_history(subject_id)
        
        # Verify subject filtering and sorting
        if result is not None and not result.empty:
            assert all(result['subject_id'] == subject_id)
            # Check if sorted by session_date
            dates = result['session_date'].tolist()
            assert dates == sorted(dates)
    
    def test_get_subject_percentile_history_no_subject(self):
        """Test getting history for non-existent subject"""
        self.mock_cache_manager.has.return_value = True
        self.mock_cache_manager.get.return_value = self.sample_session_data
        
        result = self.coordinator.get_subject_percentile_history('NonexistentSubject')
        
        assert result is None
    
    def test_clear_percentile_cache(self):
        """Test clearing percentile calculator cache"""
        # Mock percentile calculator with clear_cache method
        self.coordinator.percentile_calculator.clear_cache = Mock()
        
        self.coordinator.clear_percentile_cache()
        
        self.coordinator.percentile_calculator.clear_cache.assert_called_once()
    
    def test_clear_percentile_cache_no_method(self):
        """Test clearing cache when calculator has no clear_cache method"""
        # Should not raise an error
        self.coordinator.clear_percentile_cache()
    
    def test_get_percentile_summary_stats(self):
        """Test getting percentile summary statistics"""
        self.mock_cache_manager.has.return_value = True
        self.mock_cache_manager.get.return_value = self.sample_session_data
        
        result = self.coordinator.get_percentile_summary_stats()
        
        assert isinstance(result, dict)
        
        # Check for percentile columns in result
        percentile_columns = [col for col in self.sample_session_data.columns if 'percentile' in col.lower()]
        if percentile_columns:
            assert len(result) > 0
            for col in percentile_columns:
                if col in result:
                    stats = result[col]
                    assert 'mean' in stats
                    assert 'median' in stats
                    assert 'std' in stats
                    assert 'min' in stats
                    assert 'max' in stats
                    assert 'count' in stats
    
    def test_get_percentile_summary_stats_no_data(self):
        """Test summary stats with no data"""
        self.mock_cache_manager.has.return_value = False
        self.mock_pipeline_manager.cache_manager.get.return_value = None
        
        result = self.coordinator.get_percentile_summary_stats()
        
        assert result == {}
    
    def test_validate_percentile_calculations_valid(self):
        """Test validation of valid percentile calculations"""
        # Create data with valid percentiles
        valid_data = self.simple_session_data.copy()
        valid_data['test_percentile'] = [75.0, 25.0, 95.0]
        
        result = self.coordinator.validate_percentile_calculations(valid_data)
        
        assert result['valid'] is True
        assert len(result['issues']) == 0
    
    def test_validate_percentile_calculations_no_percentile_columns(self):
        """Test validation when no percentile columns exist"""
        # Create data without percentile columns
        data_no_percentiles = pd.DataFrame({
            'subject_id': ['A', 'B', 'C'],
            'session_date': [datetime.now()] * 3
        })
        
        result = self.coordinator.validate_percentile_calculations(data_no_percentiles)
        
        assert result['valid'] is False
        assert "No percentile columns found" in result['issues']
    
    def test_validate_percentile_calculations_invalid_values(self):
        """Test validation with invalid percentile values"""
        # Create data with invalid percentiles
        invalid_data = self.simple_session_data.copy()
        invalid_data['bad_percentile'] = [-10.0, 150.0, 50.0]  # Invalid values
        
        result = self.coordinator.validate_percentile_calculations(invalid_data)
        
        assert result['valid'] is False
        assert any("Invalid percentile values" in issue for issue in result['issues'])
    
    def test_validate_percentile_calculations_high_nan_rate(self):
        """Test validation with high NaN rate"""
        # Create data with high NaN rate
        data_with_nans = self.simple_session_data.copy()
        data_with_nans['high_nan_percentile'] = [np.nan, np.nan, 50.0]  # 67% NaN
        
        result = self.coordinator.validate_percentile_calculations(data_with_nans)
        
        # Should have warnings about high NaN rate
        assert any("High NaN rate" in warning for warning in result['warnings'])
    
    def test_coordinator_without_dependencies(self):
        """Test coordinator functionality without external dependencies"""
        standalone_coordinator = PercentileCoordinator()
        
        # Should handle gracefully when no pipeline manager available
        result = standalone_coordinator._get_session_level_data()
        assert result.empty
        
        # Should still be able to validate data
        validation = standalone_coordinator.validate_percentile_calculations(self.simple_session_data)
        assert isinstance(validation, dict)
        assert 'valid' in validation


class TestPercentileCoordinatorIntegration:
    """Integration tests for PercentileCoordinator with real dependencies"""
    
    def test_integration_with_real_calculator(self):
        """Test integration with real OverallPercentileCalculator"""
        coordinator = PercentileCoordinator()
        
        # Test that percentile calculator was properly initialized
        assert coordinator.percentile_calculator is not None
        assert hasattr(coordinator.percentile_calculator, '__class__')
        assert coordinator.percentile_calculator.__class__.__name__ == 'OverallPercentileCalculator'
    
    def test_percentile_calculation_workflow(self):
        """Test the complete percentile calculation workflow"""
        coordinator = PercentileCoordinator()
        
        # Use simple test data
        test_data = get_simple_session_data()
        
        # Test validation first
        validation = coordinator.validate_percentile_calculations(test_data)
        assert isinstance(validation, dict)
        
        # Test summary stats
        summary = coordinator.get_percentile_summary_stats(use_cache=False)
        assert isinstance(summary, dict)


if __name__ == '__main__':
    pytest.main([__file__])