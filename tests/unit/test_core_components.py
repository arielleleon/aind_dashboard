"""
Consolidated Core Components Tests

This module combines tests for the most critical components:
- PercentileCoordinator (core percentile functionality)
- AlertCoordinator (alert management)
- EnhancedDataLoader (data loading)

Simplified to focus on essential functionality and reduce test complexity.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

# Core imports
from app_utils.percentile_utils import PercentileCoordinator
from app_utils.app_alerts.alert_coordinator import AlertCoordinator
from app_utils.app_data_load import EnhancedDataLoader
from tests.fixtures.sample_data import get_realistic_session_data, get_simple_session_data


class TestPercentileCoordinator:
    """Simplified tests for PercentileCoordinator focusing on core functionality"""
    
    @pytest.fixture
    def coordinator(self):
        """Create a coordinator with mock dependencies"""
        cache_manager = Mock()
        pipeline_manager = Mock()
        return PercentileCoordinator(cache_manager=cache_manager, pipeline_manager=pipeline_manager)
    
    @pytest.fixture
    def sample_data(self):
        """Get sample session data for testing"""
        return get_realistic_session_data()
    
    def test_initialization(self, coordinator):
        """Test coordinator initializes correctly"""
        assert coordinator.cache_manager is not None
        assert coordinator.pipeline_manager is not None
        assert coordinator.percentile_calculator is not None
    
    def test_get_session_percentiles_with_cache(self, coordinator, sample_data):
        """Test getting percentiles from cache"""
        coordinator.cache_manager.has.return_value = True
        coordinator.cache_manager.get.return_value = sample_data
        
        result = coordinator.get_session_overall_percentiles(use_cache=True)
        
        coordinator.cache_manager.has.assert_called_with('session_level_data')
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
    
    def test_subject_filtering(self, coordinator, sample_data):
        """Test filtering by specific subjects"""
        coordinator.cache_manager.has.return_value = True
        coordinator.cache_manager.get.return_value = sample_data
        
        subject_ids = ['690494', '690486']
        result = coordinator.get_session_overall_percentiles(subject_ids=subject_ids)
        
        if not result.empty:
            unique_subjects = set(result['subject_id'].unique())
            assert unique_subjects.issubset(set(subject_ids))
    
    def test_calculate_percentiles(self, coordinator):
        """Test percentile calculation delegation"""
        test_data = get_simple_session_data()
        
        # Mock the calculator methods
        coordinator.percentile_calculator.calculate_session_overall_percentile = Mock(return_value=test_data)
        coordinator.percentile_calculator.calculate_session_overall_rolling_average = Mock(return_value=test_data)
        
        result = coordinator.calculate_percentiles_for_sessions(test_data)
        
        coordinator.percentile_calculator.calculate_session_overall_percentile.assert_called_once()
        assert isinstance(result, pd.DataFrame)
    
    def test_validation(self, coordinator):
        """Test data validation functionality"""
        valid_data = pd.DataFrame({
            'subject_id': ['690494', '690495'],
            'overall_percentile': [75.0, 85.0]
        })
        
        result = coordinator.validate_percentile_calculations(valid_data)
        
        assert result['valid'] is True
        assert 'issues' in result
        assert 'warnings' in result


class TestAlertCoordinator:
    """Simplified tests for AlertCoordinator focusing on core functionality"""
    
    @pytest.fixture
    def coordinator(self):
        """Create coordinator with mock dependencies"""
        cache_manager = Mock()
        pipeline_manager = Mock()
        return AlertCoordinator(cache_manager=cache_manager, pipeline_manager=pipeline_manager)
    
    def test_initialization(self, coordinator):
        """Test coordinator initializes correctly"""
        assert coordinator.cache_manager is not None
        assert coordinator.pipeline_manager is not None
        assert coordinator.alert_service is None
    
    def test_alert_service_initialization(self, coordinator):
        """Test alert service initialization"""
        mock_app_utils = Mock()
        
        alert_service = coordinator.initialize_alert_service(mock_app_utils)
        
        assert coordinator.alert_service is not None
        assert alert_service == coordinator.alert_service
    
    def test_get_alerts_without_initialization(self, coordinator):
        """Test that methods raise error when service not initialized"""
        with pytest.raises(ValueError, match="Alert service not initialized"):
            coordinator.get_quantile_alerts()
        
        with pytest.raises(ValueError, match="Alert service not initialized"):
            coordinator.get_unified_alerts()
    
    def test_unified_alerts_with_cache(self, coordinator):
        """Test unified alerts using cache"""
        # Initialize service first
        coordinator.initialize_alert_service(Mock())
        
        # Setup cache
        cached_alerts = {'test_subject': {'alert_category': 'G'}}
        coordinator.cache_manager.has.return_value = True
        coordinator.cache_manager.get.return_value = cached_alerts
        
        result = coordinator.get_unified_alerts(use_cache=True)
        
        coordinator.cache_manager.has.assert_called_with('unified_alerts')
        assert result == cached_alerts
    
    def test_alert_summary_stats(self, coordinator):
        """Test alert summary statistics"""
        # Initialize service
        coordinator.initialize_alert_service(Mock())
        
        # Mock alerts
        test_alerts = {
            'sub1': {'alert_category': 'G'},
            'sub2': {'alert_category': 'B'},
            'sub3': {'alert_category': 'G'}
        }
        coordinator.get_unified_alerts = Mock(return_value=test_alerts)
        
        result = coordinator.get_alert_summary_stats()
        
        assert result['total_subjects'] == 3
        assert result['category_counts'] == {'G': 2, 'B': 1}
        assert result['category_percentages']['G'] == pytest.approx(66.67, abs=0.01)
    
    def test_configuration_validation(self, coordinator):
        """Test alert configuration validation"""
        valid_config = {
            'percentile_categories': {
                'SB': 6.5, 'B': 28.0, 'N': 72.0, 'G': 93.5, 'SG': 100.0
            }
        }
        
        result = coordinator.validate_alert_configuration(valid_config)
        
        assert result['valid'] is True
        assert len(result['errors']) == 0


class TestEnhancedDataLoader:
    """Simplified tests for EnhancedDataLoader focusing on core functionality"""
    
    @pytest.fixture
    def mock_session_data(self):
        """Create mock session data"""
        return pd.DataFrame({
            'subject_id': ['690494', '690495', '690494'],
            'session': [1, 1, 2],
            'session_date': [
                datetime(2024, 1, 1),
                datetime(2024, 1, 1),
                datetime(2024, 1, 2)
            ]
        })
    
    def test_initialization(self, mock_session_data):
        """Test loader initializes correctly"""
        with patch('app_utils.app_data_load.data_loader.get_session_table') as mock_get:
            mock_get.return_value = mock_session_data
            
            loader = EnhancedDataLoader()
            
            assert loader.session_table is not None
            assert loader.last_load_time is not None
            assert loader.load_parameters == {'load_bpod': False}
    
    def test_get_subject_sessions(self, mock_session_data):
        """Test getting sessions for specific subject"""
        with patch('app_utils.app_data_load.data_loader.get_session_table') as mock_get:
            mock_get.return_value = mock_session_data
            
            loader = EnhancedDataLoader()
            result = loader.get_subject_sessions('690494')
            
            assert result is not None
            assert len(result) == 2  # Two sessions for sub1
            assert all(result['subject_id'] == '690494')
    
    def test_get_most_recent_sessions(self, mock_session_data):
        """Test getting most recent session per subject"""
        with patch('app_utils.app_data_load.data_loader.get_session_table') as mock_get:
            mock_get.return_value = mock_session_data
            
            loader = EnhancedDataLoader()
            result = loader.get_most_recent_subject_sessions()
            
            assert len(result) == 2  # Two unique subjects
            # sub1 should have session 2 (most recent)
            sub1_row = result[result['subject_id'] == '690494'].iloc[0]
            assert sub1_row['session'] == 2
    
    def test_data_validation(self, mock_session_data):
        """Test data validation"""
        with patch('app_utils.app_data_load.data_loader.get_session_table') as mock_get:
            mock_get.return_value = mock_session_data
            
            loader = EnhancedDataLoader()
            result = loader.validate_data()
            
            assert result['valid'] is True
            assert len(result['issues']) == 0
    
    def test_backward_compatibility(self, mock_session_data):
        """Test AppLoadData alias works correctly"""
        with patch('app_utils.app_data_load.data_loader.get_session_table') as mock_get:
            mock_get.return_value = mock_session_data
            
            from app_utils.app_data_load import AppLoadData
            
            loader = AppLoadData()
            assert isinstance(loader, EnhancedDataLoader)
            assert hasattr(loader, 'get_subject_sessions')


class TestIntegration:
    """Integration tests for component interactions"""
    
    def test_percentile_coordinator_standalone(self):
        """Test coordinator works without external dependencies"""
        coordinator = PercentileCoordinator()

        result = coordinator._get_session_level_data()
        assert result.empty
        
        # Should still validate data
        test_data = pd.DataFrame({'subject_id': ['690494'], 'overall_percentile': [50.0]})
        validation = coordinator.validate_percentile_calculations(test_data)
        assert isinstance(validation, dict)
        assert 'valid' in validation
    
    def test_alert_coordinator_error_handling(self):
        """Test alert coordinator handles errors gracefully"""
        coordinator = AlertCoordinator()
        
        # Should return error message when service not initialized
        result = coordinator.get_alert_summary_stats()
        assert 'error' in result
        assert result['total_subjects'] == 0
    
    def test_data_loader_error_handling(self):
        """Test data loader handles errors gracefully"""
        with patch('app_utils.app_data_load.data_loader.get_session_table') as mock_get:
            mock_get.side_effect = Exception("Connection failed")
            
            with pytest.raises(ValueError, match="Failed to load session table"):
                EnhancedDataLoader()


# Utility test functions for common patterns
class TestUtilities:
    """Common utility tests that can be reused"""
    
    def test_empty_dataframe_handling(self):
        """Test that components handle empty DataFrames gracefully"""
        coordinator = PercentileCoordinator()
        empty_df = pd.DataFrame()
        
        # Should return empty DataFrame without errors
        result = coordinator.calculate_percentiles_for_sessions(empty_df)
        assert result.empty
    
    def test_mock_validation(self):
        """Test that mocking works correctly across components"""
        mock_cache = Mock()
        mock_pipeline = Mock()
        
        # Both coordinators should accept mocks
        perc_coord = PercentileCoordinator(mock_cache, mock_pipeline)
        alert_coord = AlertCoordinator(mock_cache, mock_pipeline)
        
        assert perc_coord.cache_manager == mock_cache
        assert alert_coord.cache_manager == mock_cache 