"""
Unit tests for AlertCoordinator class

This module tests the alert coordination functionality including
service initialization, alert retrieval, caching, and configuration validation.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from app_utils.app_alerts.alert_coordinator import AlertCoordinator
from app_utils.app_alerts.alert_service import AlertService
from tests.fixtures.sample_data import sample_data_provider


class TestAlertCoordinator:
    """Test the AlertCoordinator class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.mock_cache_manager = Mock()
        self.mock_pipeline_manager = Mock()
        self.alert_coordinator = AlertCoordinator(
            cache_manager=self.mock_cache_manager,
            pipeline_manager=self.mock_pipeline_manager
        )
    
    def test_initialization(self):
        """Test AlertCoordinator initialization"""
        # Test with no dependencies
        coordinator = AlertCoordinator()
        assert coordinator.cache_manager is None
        assert coordinator.pipeline_manager is None
        assert coordinator.alert_service is None
        
        # Test with dependencies
        coordinator = AlertCoordinator(
            cache_manager=self.mock_cache_manager,
            pipeline_manager=self.mock_pipeline_manager
        )
        assert coordinator.cache_manager == self.mock_cache_manager
        assert coordinator.pipeline_manager == self.mock_pipeline_manager
        assert coordinator.alert_service is None
    
    def test_initialize_alert_service(self):
        """Test alert service initialization"""
        # Mock app_utils
        mock_app_utils = Mock()
        
        # Test initialization without config
        alert_service = self.alert_coordinator.initialize_alert_service(mock_app_utils)
        
        # Verify service was created
        assert self.alert_coordinator.alert_service is not None
        assert isinstance(self.alert_coordinator.alert_service, AlertService)
        assert alert_service == self.alert_coordinator.alert_service
        
        # Test initialization with config
        config = {'percentile_categories': {'SB': 5.0, 'B': 25.0}}
        alert_service_with_config = self.alert_coordinator.initialize_alert_service(
            mock_app_utils, config
        )
        
        assert alert_service_with_config is not None
    
    def test_initialize_alert_service_with_force_reset(self):
        """Test alert service initialization with force reset capability"""
        mock_app_utils = Mock()
        
        # Mock alert service with force_reset method
        with patch('app_utils.app_alerts.alert_coordinator.AlertService') as mock_alert_service_class:
            mock_alert_service = Mock()
            mock_alert_service.force_reset = Mock()
            mock_alert_service_class.return_value = mock_alert_service
            
            # Initialize service
            result = self.alert_coordinator.initialize_alert_service(mock_app_utils)
            
            # Verify force_reset was called
            mock_alert_service.force_reset.assert_called_once()
            assert result == mock_alert_service
    
    def test_get_quantile_alerts_without_initialization(self):
        """Test get_quantile_alerts raises error when service not initialized"""
        with pytest.raises(ValueError, match="Alert service not initialized"):
            self.alert_coordinator.get_quantile_alerts()
    
    def test_get_quantile_alerts_with_service(self):
        """Test get_quantile_alerts delegates properly"""
        # Initialize alert service
        mock_app_utils = Mock()
        self.alert_coordinator.initialize_alert_service(mock_app_utils)
        
        # Mock the alert service method
        expected_alerts = {'690494': {'alert_category': 'G'}}
        self.alert_coordinator.alert_service.get_quantile_alerts = Mock(
            return_value=expected_alerts
        )
        
        # Test without subject_ids
        result = self.alert_coordinator.get_quantile_alerts()
        assert result == expected_alerts
        self.alert_coordinator.alert_service.get_quantile_alerts.assert_called_with(None)
        
        # Test with subject_ids
        subject_ids = ['690494', '690486']
        result = self.alert_coordinator.get_quantile_alerts(subject_ids)
        self.alert_coordinator.alert_service.get_quantile_alerts.assert_called_with(subject_ids)
    
    def test_get_unified_alerts_without_initialization(self):
        """Test get_unified_alerts raises error when service not initialized"""
        with pytest.raises(ValueError, match="Alert service not initialized"):
            self.alert_coordinator.get_unified_alerts()
    
    def test_get_unified_alerts_with_cache(self):
        """Test get_unified_alerts uses cache when available"""
        # Initialize alert service first
        mock_app_utils = Mock()
        self.alert_coordinator.initialize_alert_service(mock_app_utils)
        
        # Setup cache to return data
        cached_alerts = {'690494': {'alert_category': 'B'}}
        self.mock_cache_manager.has.return_value = True
        self.mock_cache_manager.get.return_value = cached_alerts
        
        # Test with cache
        result = self.alert_coordinator.get_unified_alerts(use_cache=True)
        
        # Verify cache was used
        self.mock_cache_manager.has.assert_called_with('unified_alerts')
        self.mock_cache_manager.get.assert_called_with('unified_alerts')
        assert result == cached_alerts
    
    def test_get_unified_alerts_without_cache(self):
        """Test get_unified_alerts when cache is not available"""
        # Setup cache to not have data
        self.mock_cache_manager.has.return_value = False
        
        # Initialize alert service
        mock_app_utils = Mock()
        self.alert_coordinator.initialize_alert_service(mock_app_utils)
        
        # Mock the alert service method
        expected_alerts = {'690494': {'alert_category': 'N'}}
        self.alert_coordinator.alert_service.get_unified_alerts = Mock(
            return_value=expected_alerts
        )
        
        # Test without cache
        result = self.alert_coordinator.get_unified_alerts()
        
        # Verify cache was checked but service was used
        self.mock_cache_manager.has.assert_called_with('unified_alerts')
        self.alert_coordinator.alert_service.get_unified_alerts.assert_called_with(None)
        
        # Verify result was cached
        self.mock_cache_manager.set.assert_called_with('unified_alerts', expected_alerts)
        assert result == expected_alerts
    
    def test_get_unified_alerts_with_specific_subjects(self):
        """Test get_unified_alerts with specific subject IDs (no caching)"""
        # Initialize alert service
        mock_app_utils = Mock()
        self.alert_coordinator.initialize_alert_service(mock_app_utils)
        
        # Mock the alert service method
        expected_alerts = {'690494': {'alert_category': 'SG'}}
        self.alert_coordinator.alert_service.get_unified_alerts = Mock(
            return_value=expected_alerts
        )
        
        # Test with specific subjects (should not use cache)
        subject_ids = ['690494']
        result = self.alert_coordinator.get_unified_alerts(
            subject_ids=subject_ids, 
            use_cache=True
        )
        
        # Verify cache was not checked for specific subjects
        self.mock_cache_manager.has.assert_not_called()
        self.alert_coordinator.alert_service.get_unified_alerts.assert_called_with(subject_ids)
        
        # Verify result was not cached for specific subjects
        self.mock_cache_manager.set.assert_not_called()
        assert result == expected_alerts
    
    def test_clear_alert_cache(self):
        """Test clearing alert caches"""
        # Setup cache manager
        self.mock_cache_manager.has.return_value = True
        
        # Initialize alert service with internal cache
        mock_app_utils = Mock()
        self.alert_coordinator.initialize_alert_service(mock_app_utils)
        self.alert_coordinator.alert_service._quantile_alerts = {'test': 'data'}
        
        # Clear caches
        self.alert_coordinator.clear_alert_cache()
        
        # Verify cache was checked
        self.mock_cache_manager.has.assert_called_with('unified_alerts')
        
        # Verify alert service internal cache was cleared
        assert self.alert_coordinator.alert_service._quantile_alerts == {}
    
    def test_clear_alert_cache_without_cache_manager(self):
        """Test clearing alert cache when no cache manager"""
        coordinator = AlertCoordinator()  # No cache manager
        
        # Should not raise error
        coordinator.clear_alert_cache()
    
    def test_get_alert_summary_stats_without_service(self):
        """Test alert summary stats when service not initialized"""
        result = self.alert_coordinator.get_alert_summary_stats()
        
        expected = {
            'error': 'Alert service not initialized',
            'total_subjects': 0,
            'category_counts': {}
        }
        assert result == expected
    
    def test_get_alert_summary_stats_with_service(self):
        """Test alert summary statistics calculation"""
        # Initialize alert service
        mock_app_utils = Mock()
        self.alert_coordinator.initialize_alert_service(mock_app_utils)
        
        # Mock get_unified_alerts to return test data
        test_alerts = {
            'subject1': {'alert_category': 'G'},
            'subject2': {'alert_category': 'B'},
            'subject3': {'alert_category': 'G'},
            'subject4': {'alert_category': 'N'},
            'subject5': {'alert_category': 'NS'}
        }
        
        self.alert_coordinator.get_unified_alerts = Mock(return_value=test_alerts)
        
        # Get summary stats
        result = self.alert_coordinator.get_alert_summary_stats()
        
        # Verify results
        assert result['total_subjects'] == 5
        assert result['category_counts'] == {'G': 2, 'B': 1, 'N': 1, 'NS': 1}
        assert result['category_percentages']['G'] == 40.0
        assert result['category_percentages']['B'] == 20.0
        assert result['category_percentages']['N'] == 20.0
        assert result['category_percentages']['NS'] == 20.0
        assert set(result['categories_found']) == {'G', 'B', 'N', 'NS'}
    
    def test_get_alert_summary_stats_with_error(self):
        """Test alert summary stats with error handling"""
        # Initialize alert service
        mock_app_utils = Mock()
        self.alert_coordinator.initialize_alert_service(mock_app_utils)
        
        # Mock get_unified_alerts to raise exception
        self.alert_coordinator.get_unified_alerts = Mock(
            side_effect=Exception("Test error")
        )
        
        # Get summary stats
        result = self.alert_coordinator.get_alert_summary_stats()
        
        # Verify error handling
        assert 'error' in result
        assert 'Test error' in result['error']
        assert result['total_subjects'] == 0
        assert result['category_counts'] == {}
    
    def test_validate_alert_configuration_valid(self):
        """Test validation of valid alert configuration"""
        valid_config = {
            'percentile_categories': {
                'SB': 6.5,
                'B': 28.0,
                'N': 72.0,
                'G': 93.5,
                'SG': 100.0
            },
            'feature_config': {
                'finished_trials': {'threshold': 400}
            }
        }
        
        result = self.alert_coordinator.validate_alert_configuration(valid_config)
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
        assert len(result['warnings']) == 0
    
    def test_validate_alert_configuration_missing_categories(self):
        """Test validation with missing category thresholds"""
        invalid_config = {
            'percentile_categories': {
                'SB': 6.5,
                'B': 28.0,
                # Missing N, G, SG
            }
        }
        
        result = self.alert_coordinator.validate_alert_configuration(invalid_config)
        
        assert result['valid'] is False
        assert len(result['errors']) == 3  # Missing N, G, SG
        assert any('Missing category threshold: N' in error for error in result['errors'])
        assert any('Missing category threshold: G' in error for error in result['errors'])
        assert any('Missing category threshold: SG' in error for error in result['errors'])
    
    def test_validate_alert_configuration_invalid_types(self):
        """Test validation with invalid threshold types"""
        invalid_config = {
            'percentile_categories': {
                'SB': 'invalid',  # Should be numeric
                'B': 28.0,
                'N': 72.0,
                'G': 93.5,
                'SG': 100.0
            }
        }
        
        result = self.alert_coordinator.validate_alert_configuration(invalid_config)
        
        assert result['valid'] is False
        assert len(result['errors']) == 1
        assert 'Invalid threshold type for SB: must be numeric' in result['errors'][0]
    
    def test_validate_alert_configuration_wrong_order(self):
        """Test validation with wrong threshold order"""
        config_wrong_order = {
            'percentile_categories': {
                'SB': 6.5,
                'B': 93.5,  # Wrong order
                'N': 28.0,  # Wrong order
                'G': 72.0,  # Wrong order
                'SG': 100.0
            }
        }
        
        result = self.alert_coordinator.validate_alert_configuration(config_wrong_order)
        
        assert result['valid'] is True  # Still valid, just warning
        assert len(result['warnings']) == 1
        assert 'Category thresholds may not be in expected order' in result['warnings'][0]
    
    def test_validate_alert_configuration_invalid_feature_config(self):
        """Test validation with invalid feature configuration"""
        invalid_config = {
            'feature_config': 'not_a_dict'  # Should be dict
        }
        
        result = self.alert_coordinator.validate_alert_configuration(invalid_config)
        
        assert result['valid'] is False
        assert 'feature_config must be a dictionary' in result['errors']

    def test_filter_by_alert_category_threshold_alerts(self):
        """Test filtering by threshold alerts using AlertCoordinator"""
        
        # Create test dataframe
        test_data = {
            'subject_id': ['001', '002', '003', '004'],
            'threshold_alert': ['T', None, None, None],
            'total_sessions_alert': [None, 'T | Test', None, None],
            'overall_percentile_category': ['T', 'B', 'G', 'NS']
        }
        df = pd.DataFrame(test_data)
        
        # Mock app_utils
        mock_app_utils = Mock()
        mock_app_utils.quantile_analyzer = Mock()
        mock_app_utils.percentile_calculator = Mock()
        
        # Initialize alert service
        self.alert_coordinator.initialize_alert_service(mock_app_utils)
        
        # Test threshold alert filtering
        result = self.alert_coordinator.filter_by_alert_category(df, 'T')
        
        # Should include subjects with threshold alerts (001 and 002)
        assert len(result) == 2
        assert set(result['subject_id']) == {'001', '002'}
    
    def test_filter_by_alert_category_percentile_categories(self):
        """Test filtering by percentile categories using AlertCoordinator"""
        
        # Create test dataframe
        test_data = {
            'subject_id': ['001', '002', '003', '004', '005'],
            'overall_percentile_category': ['NS', 'B', 'G', 'SB', 'SG']
        }
        df = pd.DataFrame(test_data)
        
        # Mock app_utils
        mock_app_utils = Mock()
        mock_app_utils.quantile_analyzer = Mock()
        mock_app_utils.percentile_calculator = Mock()
        
        # Initialize alert service
        self.alert_coordinator.initialize_alert_service(mock_app_utils)
        
        # Test each category
        categories_to_test = ['NS', 'B', 'G', 'SB', 'SG']
        
        for category in categories_to_test:
            result = self.alert_coordinator.filter_by_alert_category(df, category)
            assert len(result) == 1
            assert result['overall_percentile_category'].iloc[0] == category
    
    def test_filter_by_alert_category_all_returns_original(self):
        """Test that 'all' category returns original dataframe"""
        
        test_data = {
            'subject_id': ['001', '002', '003'],
            'overall_percentile_category': ['B', 'G', 'NS']
        }
        df = pd.DataFrame(test_data)
        
        # Mock app_utils
        mock_app_utils = Mock()
        self.alert_coordinator.initialize_alert_service(mock_app_utils)
        
        result = self.alert_coordinator.filter_by_alert_category(df, 'all')
        
        # Should return all data unchanged
        assert len(result) == len(df)
        pd.testing.assert_frame_equal(result, df)
    
    def test_aggregate_alert_categories(self):
        """Test alert category aggregation functionality"""
        
        # Create test dataframe
        test_data = {
            'subject_id': ['001', '002', '003', '004', '005', '006'],
            'threshold_alert': ['T', None, None, None, None, None],
            'total_sessions_alert': [None, 'T | Test', None, None, None, None],
            'overall_percentile_category': ['T', 'B', 'G', 'NS', 'SB', 'SG']
        }
        df = pd.DataFrame(test_data)
        
        # Mock app_utils
        mock_app_utils = Mock()
        mock_app_utils.quantile_analyzer = Mock()
        mock_app_utils.percentile_calculator = Mock()
        
        # Initialize alert service
        self.alert_coordinator.initialize_alert_service(mock_app_utils)
        
        # Test aggregation
        result = self.alert_coordinator.aggregate_alert_categories(df)
        
        # Should have counts for each category
        assert isinstance(result, dict)
        assert result.get('T', 0) >= 2  # Threshold alerts from pattern matching
        assert result.get('B', 0) == 1
        assert result.get('G', 0) == 1
        assert result.get('NS', 0) == 1
        assert result.get('SB', 0) == 1
        assert result.get('SG', 0) == 1
    
    def test_filter_by_alert_category_without_alert_service(self):
        """Test that filtering raises error when alert service not initialized"""
        
        test_data = {
            'subject_id': ['001'],
            'overall_percentile_category': ['B']
        }
        df = pd.DataFrame(test_data)
        
        # Don't initialize alert service
        with pytest.raises(ValueError, match="Alert service not initialized"):
            self.alert_coordinator.filter_by_alert_category(df, 'B')
    
    def test_aggregate_alert_categories_fallback(self):
        """Test aggregation fallback when alert service not available"""
        
        test_data = {
            'subject_id': ['001', '002', '003'],
            'overall_percentile_category': ['B', 'G', 'NS']
        }
        df = pd.DataFrame(test_data)
        
        # Don't initialize alert service
        result = self.alert_coordinator.aggregate_alert_categories(df)
        
        # Should fall back to basic counting
        assert result == {'B': 1, 'G': 1, 'NS': 1}


class TestAlertCoordinatorEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_clear_cache_without_alert_service(self):
        """Test clearing cache when alert service not initialized"""
        mock_cache_manager = Mock()
        coordinator = AlertCoordinator(cache_manager=mock_cache_manager)
        
        # Should not raise error
        coordinator.clear_alert_cache()
    
    def test_summary_stats_empty_alerts(self):
        """Test summary stats with empty alerts"""
        mock_cache_manager = Mock()
        coordinator = AlertCoordinator(cache_manager=mock_cache_manager)
        
        # Initialize alert service
        mock_app_utils = Mock()
        coordinator.initialize_alert_service(mock_app_utils)
        
        # Mock empty alerts
        coordinator.get_unified_alerts = Mock(return_value={})
        
        result = coordinator.get_alert_summary_stats()
        
        assert result['total_subjects'] == 0
        assert result['category_counts'] == {}
        assert result['category_percentages'] == {}
        assert result['categories_found'] == []
    
    def test_validate_configuration_empty_config(self):
        """Test validation with empty configuration"""
        coordinator = AlertCoordinator()
        
        result = coordinator.validate_alert_configuration({})
        
        # Empty config should be valid (uses defaults)
        assert result['valid'] is True
        assert len(result['errors']) == 0 