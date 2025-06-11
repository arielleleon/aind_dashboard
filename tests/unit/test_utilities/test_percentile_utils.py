"""
Tests for PercentileCoordinator class

This module contains tests for the percentile coordination functionality
that was extracted from the main AppUtils class during refactoring.

Tests focus on:
- Session-level percentile calculations
- Data coordination between pipeline and calculators
- Cache management for percentile results
- Error handling and edge cases
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from app_utils.percentile_utils import PercentileCoordinator, calculate_heatmap_colorscale, format_feature_display_names


class TestPercentileCoordinator:
    """Test the PercentileCoordinator class functionality"""
    
    def test_initialization(self):
        """Test coordinator initialization with dependencies"""
        cache_manager = Mock()
        pipeline_manager = Mock()
        
        coordinator = PercentileCoordinator(
            cache_manager=cache_manager,
            pipeline_manager=pipeline_manager
        )
        
        assert coordinator.cache_manager == cache_manager
        assert coordinator.pipeline_manager == pipeline_manager
        assert coordinator.percentile_calculator is not None
    
    def test_initialization_without_dependencies(self):
        """Test coordinator initialization without dependencies"""
        coordinator = PercentileCoordinator()
        
        assert coordinator.cache_manager is None
        assert coordinator.pipeline_manager is None
        assert coordinator.percentile_calculator is not None
    
    def test_get_session_overall_percentiles_with_cache(self):
        """Test getting session percentiles with cache enabled"""
        # Setup mocks
        cache_manager = Mock()
        pipeline_manager = Mock()
        
        # Setup cached session data
        cached_data = pd.DataFrame({
            'subject_id': ['sub1', 'sub2', 'sub3'],
            'session_date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'overall_percentile': [75.0, 85.0, 65.0]
        })
        
        cache_manager.has.return_value = True
        cache_manager.get.return_value = cached_data
        
        coordinator = PercentileCoordinator(
            cache_manager=cache_manager,
            pipeline_manager=pipeline_manager
        )
        
        # Test retrieval
        result = coordinator.get_session_overall_percentiles(use_cache=True)
        
        # Verify cache was used
        cache_manager.has.assert_called_with('session_level_data')
        cache_manager.get.assert_called_with('session_level_data')
        
        # Verify result structure
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'subject_id' in result.columns
    
    def test_get_session_overall_percentiles_specific_subjects(self):
        """Test getting percentiles for specific subjects"""
        cache_manager = Mock()
        pipeline_manager = Mock()
        
        # Setup test data with multiple subjects
        session_data = pd.DataFrame({
            'subject_id': ['sub1', 'sub2', 'sub3', 'sub1', 'sub2'],
            'session_date': pd.to_datetime([
                '2023-01-01', '2023-01-01', '2023-01-01',
                '2023-01-02', '2023-01-02'
            ]),
            'overall_percentile': [75.0, 85.0, 65.0, 80.0, 90.0]
        })
        
        cache_manager.has.return_value = True
        cache_manager.get.return_value = session_data
        
        coordinator = PercentileCoordinator(
            cache_manager=cache_manager,
            pipeline_manager=pipeline_manager
        )
        
        # Test filtering for specific subjects
        result = coordinator.get_session_overall_percentiles(
            subject_ids=['sub1', 'sub2'], 
            use_cache=True
        )
        
        # Verify filtering worked
        assert set(result['subject_id'].unique()) <= {'sub1', 'sub2'}
        assert 'sub3' not in result['subject_id'].values
    
    def test_get_session_overall_percentiles_no_cache(self):
        """Test getting percentiles without cache"""
        cache_manager = Mock()
        pipeline_manager = Mock()
        
        cache_manager.has.return_value = False
        
        coordinator = PercentileCoordinator(
            cache_manager=cache_manager,
            pipeline_manager=pipeline_manager
        )
        
        result = coordinator.get_session_overall_percentiles(use_cache=False)
        
        # Should return empty DataFrame when no cache and no pipeline data
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_get_session_overall_percentiles_empty_filter(self):
        """Test getting percentiles with filter that matches no subjects"""
        cache_manager = Mock()
        pipeline_manager = Mock()
        
        session_data = pd.DataFrame({
            'subject_id': ['sub1', 'sub2'],
            'session_date': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'overall_percentile': [75.0, 85.0]
        })
        
        cache_manager.has.return_value = True
        cache_manager.get.return_value = session_data
        
        coordinator = PercentileCoordinator(
            cache_manager=cache_manager,
            pipeline_manager=pipeline_manager
        )
        
        # Test with subjects not in data
        result = coordinator.get_session_overall_percentiles(
            subject_ids=['nonexistent'], 
            use_cache=True
        )
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_calculate_percentiles_for_sessions(self):
        """Test percentile calculation for session data"""
        coordinator = PercentileCoordinator()
        
        # Create test session data
        session_data = pd.DataFrame({
            'subject_id': ['sub1', 'sub2'],
            'session_date': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'finished_trials': [100, 150],
            'ignore_rate': [0.1, 0.05]
        })
        
        # Mock the percentile calculator
        with patch.object(coordinator.percentile_calculator, 'calculate_session_overall_percentile') as mock_calc:
            enhanced_data = session_data.copy()
            enhanced_data['overall_percentile'] = [75.0, 85.0]
            mock_calc.return_value = enhanced_data
            
            result = coordinator.calculate_percentiles_for_sessions(session_data)
            
            # Verify calculator was called
            mock_calc.assert_called_once_with(session_data)
            
            # Verify enhanced data returned
            assert 'overall_percentile' in result.columns
            assert result['overall_percentile'].tolist() == [75.0, 85.0]
    
    def test_calculate_percentiles_for_sessions_empty_data(self):
        """Test percentile calculation with empty session data"""
        coordinator = PercentileCoordinator()
        
        empty_data = pd.DataFrame()
        result = coordinator.calculate_percentiles_for_sessions(empty_data)
        
        # Should return the same empty DataFrame
        assert result.empty
        assert result.equals(empty_data)
    
    def test_get_percentiles_by_strata(self):
        """Test getting percentiles filtered by strata"""
        cache_manager = Mock()
        pipeline_manager = Mock()
        
        session_data = pd.DataFrame({
            'subject_id': ['sub1', 'sub2', 'sub3'],
            'strata': ['strata_A', 'strata_B', 'strata_A'],
            'overall_percentile': [75.0, 85.0, 65.0]
        })
        
        cache_manager.has.return_value = True
        cache_manager.get.return_value = session_data
        
        coordinator = PercentileCoordinator(
            cache_manager=cache_manager,
            pipeline_manager=pipeline_manager
        )
        
        result = coordinator.get_percentiles_by_strata('strata_A')
        
        # Verify filtering
        assert result is not None
        assert len(result) == 2  # Two subjects in strata_A
        assert all(result['strata'] == 'strata_A')
    
    def test_get_percentiles_by_strata_not_found(self):
        """Test getting percentiles for non-existent strata"""
        cache_manager = Mock()
        pipeline_manager = Mock()
        
        session_data = pd.DataFrame({
            'subject_id': ['sub1', 'sub2'],
            'strata': ['strata_A', 'strata_B'],
            'overall_percentile': [75.0, 85.0]
        })
        
        cache_manager.has.return_value = True
        cache_manager.get.return_value = session_data
        
        coordinator = PercentileCoordinator(
            cache_manager=cache_manager,
            pipeline_manager=pipeline_manager
        )
        
        result = coordinator.get_percentiles_by_strata('nonexistent_strata')
        
        assert result is None
    
    def test_get_subject_percentile_history(self):
        """Test getting percentile history for a subject"""
        cache_manager = Mock()
        pipeline_manager = Mock()
        
        session_data = pd.DataFrame({
            'subject_id': ['sub1', 'sub1', 'sub2', 'sub1'],
            'session_date': pd.to_datetime([
                '2023-01-01', '2023-01-02', '2023-01-01', '2023-01-03'
            ]),
            'overall_percentile': [70.0, 75.0, 85.0, 80.0]
        })
        
        cache_manager.has.return_value = True
        cache_manager.get.return_value = session_data
        
        coordinator = PercentileCoordinator(
            cache_manager=cache_manager,
            pipeline_manager=pipeline_manager
        )
        
        result = coordinator.get_subject_percentile_history('sub1')
        
        # Should return only sub1 sessions
        assert result is not None
        assert len(result) == 3  # Three sessions for sub1
        assert all(result['subject_id'] == 'sub1')
    
    def test_clear_percentile_cache(self):
        """Test clearing percentile cache"""
        # Create a mock percentile calculator with clear_cache method
        mock_calculator = Mock()
        mock_calculator.clear_cache = Mock()
        
        cache_manager = Mock()
        coordinator = PercentileCoordinator(cache_manager=cache_manager)
        
        # Replace the coordinator's calculator with our mock
        coordinator.percentile_calculator = mock_calculator
        
        coordinator.clear_percentile_cache()
        
        # Verify the calculator's clear_cache method was called
        mock_calculator.clear_cache.assert_called_once()
    
    def test_clear_percentile_cache_no_manager(self):
        """Test clearing cache without cache manager"""
        coordinator = PercentileCoordinator()
        
        # Should not raise exception
        coordinator.clear_percentile_cache()
    
    def test_validate_percentile_calculations(self):
        """Test percentile calculation validation"""
        coordinator = PercentileCoordinator()
        
        # Valid session data
        session_data = pd.DataFrame({
            'subject_id': ['sub1', 'sub2'],
            'overall_percentile': [75.0, 85.0],
            'finished_trials': [100, 150]
        })
        
        result = coordinator.validate_percentile_calculations(session_data)
        
        assert 'valid' in result
        assert 'issues' in result
        assert 'warnings' in result
        assert result['valid'] == True


class TestHeatmapColorscaleFunctions:
    """Test the colorscale functions extracted from the heatmap component"""
    
    def test_calculate_heatmap_colorscale_binned(self):
        """Test binned colorscale calculation"""
        colorscale = calculate_heatmap_colorscale('binned')
        
        # Verify structure
        assert isinstance(colorscale, list)
        assert len(colorscale) == 10  # Should have 10 color stops
        
        # Verify format - each entry should be [position, color]
        for entry in colorscale:
            assert isinstance(entry, list)
            assert len(entry) == 2
            assert isinstance(entry[0], (int, float))  # Position
            assert isinstance(entry[1], str)  # Color hex code
            assert entry[1].startswith('#')  # Valid hex color
        
        # Verify key thresholds are present
        positions = [entry[0] for entry in colorscale]
        assert 0.0 in positions
        assert 0.065 in positions
        assert 0.28 in positions
        assert 0.72 in positions
        assert 0.935 in positions
        assert 1.0 in positions
    
    def test_calculate_heatmap_colorscale_continuous(self):
        """Test continuous colorscale calculation"""
        colorscale = calculate_heatmap_colorscale('continuous')
        
        # Verify structure
        assert isinstance(colorscale, list)
        assert len(colorscale) == 11  # Should have 11 color stops for smooth transitions
        
        # Verify format
        for entry in colorscale:
            assert isinstance(entry, list)
            assert len(entry) == 2
            assert isinstance(entry[0], (int, float))
            assert isinstance(entry[1], str)
            assert entry[1].startswith('#')
        
        # Verify smooth progression
        positions = [entry[0] for entry in colorscale]
        assert positions == sorted(positions)  # Should be in ascending order
        assert positions[0] == 0.0
        assert positions[-1] == 1.0
    
    def test_calculate_heatmap_colorscale_default(self):
        """Test default colorscale calculation"""
        # Should default to binned mode
        default_colorscale = calculate_heatmap_colorscale()
        binned_colorscale = calculate_heatmap_colorscale('binned')
        
        assert default_colorscale == binned_colorscale
    
    def test_calculate_heatmap_colorscale_invalid_mode(self):
        """Test colorscale with invalid mode"""
        # Should fall back to binned for any unrecognized mode
        colorscale = calculate_heatmap_colorscale('invalid_mode')
        expected = calculate_heatmap_colorscale('binned')
        
        assert colorscale == expected
    
    def test_format_feature_display_names(self):
        """Test feature display name formatting"""
        features_config = {
            'finished_trials': False,
            'ignore_rate': True,
            'abs(bias_naive)': True,
            'foraging_performance': False,
            'total_trials': False
        }
        
        result = format_feature_display_names(features_config)
        
        # Verify structure
        assert isinstance(result, dict)
        assert len(result) == len(features_config)
        
        # Verify specific transformations
        expected_mappings = {
            'finished_trials': 'Finished Trials',
            'ignore_rate': 'Ignore Rate',
            'abs(bias_naive)': '|Bias Naive|',
            'foraging_performance': 'Foraging Performance',
            'total_trials': 'Total Trials'
        }
        
        for feature, expected_display in expected_mappings.items():
            assert result[feature] == expected_display
    
    def test_format_feature_display_names_empty(self):
        """Test feature display name formatting with empty config"""
        result = format_feature_display_names({})
        
        assert result == {}
    
    def test_format_feature_display_names_edge_cases(self):
        """Test feature display name formatting with edge cases"""
        features_config = {
            'simple': True,
            'multiple_underscores_here': False,
            'abs(complex_formula)': True,
            'already_formatted': False
        }
        
        result = format_feature_display_names(features_config)
        
        expected = {
            'simple': 'Simple',
            'multiple_underscores_here': 'Multiple Underscores Here',
            'abs(complex_formula)': '|Complex Formula|',
            'already_formatted': 'Already Formatted'
        }
        
        assert result == expected


class TestPercentileCoordinatorIntegration:
    """Integration tests for PercentileCoordinator with other components"""
    
    def test_coordinator_with_pipeline_integration(self):
        """Test coordinator integration with data pipeline"""
        cache_manager = Mock()
        pipeline_manager = Mock()
        
        # Setup pipeline to return processed data
        processed_data = pd.DataFrame({
            'subject_id': ['sub1', 'sub2'],
            'session_date': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'overall_percentile': [75.0, 85.0]
        })
        
        cache_manager.has.side_effect = lambda key: key == 'raw_data'
        cache_manager.get.side_effect = lambda key: pd.DataFrame({'raw': [1, 2]}) if key == 'raw_data' else None
        pipeline_manager.process_data_pipeline.return_value = processed_data
        
        coordinator = PercentileCoordinator(
            cache_manager=cache_manager,
            pipeline_manager=pipeline_manager
        )
        
        result = coordinator.get_session_overall_percentiles(use_cache=True)
        
        # Verify pipeline was used when session cache missing
        pipeline_manager.process_data_pipeline.assert_called_once()
        assert not result.empty


class TestPercentileUtilsEdgeCases:
    """Test edge cases and error handling for percentile utilities"""
    
    def test_colorscale_color_values(self):
        """Test that colorscale colors are valid hex codes"""
        binned = calculate_heatmap_colorscale('binned')
        continuous = calculate_heatmap_colorscale('continuous')
        
        def is_valid_hex_color(color):
            return len(color) == 7 and color.startswith('#') and all(c in '0123456789ABCDEFabcdef' for c in color[1:])
        
        # Test binned colors
        for position, color in binned:
            assert is_valid_hex_color(color), f"Invalid hex color: {color} at position {position}"
        
        # Test continuous colors
        for position, color in continuous:
            assert is_valid_hex_color(color), f"Invalid hex color: {color} at position {position}"
    
    def test_colorscale_position_bounds(self):
        """Test that colorscale positions are within valid bounds"""
        binned = calculate_heatmap_colorscale('binned')
        continuous = calculate_heatmap_colorscale('continuous')
        
        # Test binned positions
        for position, color in binned:
            assert 0.0 <= position <= 1.0, f"Position {position} out of bounds [0,1]"
        
        # Test continuous positions
        for position, color in continuous:
            assert 0.0 <= position <= 1.0, f"Position {position} out of bounds [0,1]"
    
    def test_colorscale_alert_category_mapping(self):
        """Test that colorscale matches expected alert category thresholds"""
        binned = calculate_heatmap_colorscale('binned')
        
        # Extract key threshold positions
        positions = [entry[0] for entry in binned]
        
        # Verify alert category thresholds are present
        assert 0.065 in positions  # SB -> B threshold (6.5%)
        assert 0.28 in positions   # B -> N threshold (28%)
        assert 0.72 in positions   # N -> G threshold (72%)
        assert 0.935 in positions  # G -> SG threshold (93.5%)

# ... existing code ... 