"""
Unit tests for dataframe business logic functions

These tests verify the DataFrame manipulation and processing logic
extracted from UI components, ensuring data transformations work correctly.

Updated to use realistic fixtures from sample_data.py for consistency.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

# Ensure we can import from the project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from app_utils.ui_utils import (
    create_empty_dataframe_structure,
    format_strata_abbreviations,
    get_optimized_table_data,
    process_unified_alerts_integration
)

# Import realistic fixtures
from tests.fixtures.sample_data import get_realistic_session_data, get_simple_session_data


class TestDataFrameBusinessLogic(unittest.TestCase):
    """Test the extracted dataframe business logic functions"""

    def setUp(self):
        """Set up test fixtures using realistic sample data"""
        self.sample_session_data = get_realistic_session_data()
        # Also create a simple version for lightweight tests
        self.simple_session_data = get_simple_session_data()

    def test_create_empty_dataframe_structure(self):
        """Test empty dataframe creation"""
        result = create_empty_dataframe_structure()
        
        # Verify structure
        expected_columns = [
            'subject_id', 'combined_alert', 'percentile_category', 
            'overall_percentile', 'session_date', 'session'
        ]
        self.assertEqual(list(result.columns), expected_columns)
        self.assertEqual(len(result), 0)

    def test_format_strata_abbreviations_basic(self):
        """Test basic strata abbreviation formatting using realistic data"""
        test_df = self.sample_session_data.copy()
        result = format_strata_abbreviations(test_df)
        
        # Verify abbreviations are created
        self.assertIn('strata_abbr', result.columns)
        # Check that abbreviations are reasonable (not empty for valid strata)
        non_empty_strata = result[result['strata'].notna() & (result['strata'] != '')]
        if len(non_empty_strata) > 0:
            self.assertTrue(all(abbr != '' for abbr in non_empty_strata['strata_abbr']))

    def test_format_strata_abbreviations_edge_cases(self):
        """Test strata abbreviation with edge cases"""
        test_df = pd.DataFrame({
            'strata': ['', 'Simple Name', 'Complex_Name_With_Multiple_Parts']
        })
        
        result = format_strata_abbreviations(test_df)
        
        # Check empty string handling
        self.assertEqual(result['strata_abbr'].iloc[0], '')
        # Check simple name handling
        self.assertEqual(result['strata_abbr'].iloc[1], 'SimpleName')

    def test_format_strata_abbreviations_existing_column(self):
        """Test strata abbreviation when column already exists"""
        test_df = self.sample_session_data.copy()[:3]  # Use first 3 rows
        test_df['strata_abbr'] = ['UBA3', 'CBB1', 'UWBI2']
        
        result = format_strata_abbreviations(test_df)
        
        # Should preserve existing values
        self.assertEqual(result['strata_abbr'].tolist(), ['UBA3', 'CBB1', 'UWBI2'])

    def test_format_strata_abbreviations_missing_strata_column(self):
        """Test strata abbreviation when strata column is missing"""
        # Use realistic subject IDs from fixture data
        base_data = get_realistic_session_data()
        subject_ids = base_data['subject_id'].unique()[:2].tolist()
        test_df = pd.DataFrame({'subject_id': subject_ids})
        
        result = format_strata_abbreviations(test_df)
        
        # Should add empty abbreviation column
        self.assertIn('strata_abbr', result.columns)
        self.assertEqual(result['strata_abbr'].tolist(), ['', ''])

    @patch('app_utils.ui_utils.print')
    def test_get_optimized_table_data_ui_cache_hit(self, mock_print):
        """Test get_optimized_table_data with UI cache hit"""
        # Mock app_utils with UI cache data using realistic structure
        mock_app_utils = Mock()
        realistic_subjects = self.sample_session_data['subject_id'].unique()[:2].tolist()
        mock_app_utils.get_table_display_data.return_value = [
            {'subject_id': realistic_subjects[0], 'session': 1},
            {'subject_id': realistic_subjects[1], 'session': 2}
        ]
        
        result = get_optimized_table_data(mock_app_utils, use_cache=True)
        
        # Verify UI cache was used
        mock_app_utils.get_table_display_data.assert_called_once_with(use_cache=True)
        self.assertEqual(len(result), 2)
        self.assertEqual(result['subject_id'].tolist(), realistic_subjects)

    @patch('app_utils.ui_utils.print')
    def test_get_optimized_table_data_session_cache_fallback(self, mock_print):
        """Test get_optimized_table_data with session cache fallback"""
        # Mock app_utils with no UI cache but session cache using realistic data
        mock_app_utils = Mock()
        mock_app_utils.get_table_display_data.return_value = None
        realistic_subject = self.sample_session_data['subject_id'].iloc[0]
        mock_app_utils._cache.get.return_value = pd.DataFrame({'subject_id': [realistic_subject]})
        mock_app_utils.get_most_recent_subject_sessions.return_value = pd.DataFrame({
            'subject_id': [realistic_subject], 'session': [1]
        })
        
        result = get_optimized_table_data(mock_app_utils, use_cache=True)
        
        # Verify session cache fallback was used
        mock_app_utils.get_most_recent_subject_sessions.assert_called_once_with(use_cache=True)
        self.assertEqual(len(result), 1)

    @patch('app_utils.ui_utils.print')
    def test_get_optimized_table_data_pipeline_fallback(self, mock_print):
        """Test get_optimized_table_data with pipeline fallback"""
        # Mock app_utils with no caches using realistic data
        mock_app_utils = Mock()
        mock_app_utils.get_table_display_data.side_effect = [None, None]  # First call fails, second succeeds
        mock_app_utils._cache.get.return_value = None
        realistic_subject = self.sample_session_data['subject_id'].iloc[0]
        mock_app_utils.get_most_recent_subject_sessions.return_value = pd.DataFrame({
            'subject_id': [realistic_subject], 'session': [1]
        })
        
        result = get_optimized_table_data(mock_app_utils, use_cache=True)
        
        # Verify pipeline was triggered
        mock_app_utils.process_data_pipeline.assert_called_once_with(use_cache=False)

    def test_process_unified_alerts_integration_basic(self):
        """Test basic unified alerts integration using realistic data"""
        # Mock app_utils with realistic subject IDs
        mock_app_utils = Mock()
        mock_app_utils.alert_service = Mock()  # Already initialized
        realistic_subjects = self.sample_session_data['subject_id'].unique()[:2].tolist()
        mock_app_utils.get_unified_alerts.return_value = {
            realistic_subjects[0]: {
                'alert_category': 'SB',
                'ns_reason': None,
                'threshold': {'threshold_alert': 'N'}
            },
            realistic_subjects[1]: {
                'alert_category': 'G',
                'ns_reason': None,
                'threshold': {'threshold_alert': 'T', 'specific_alerts': {
                    'total_sessions': {'alert': 'T', 'value': 45}
                }}
            }
        }
        
        test_df = self.sample_session_data.copy()[:2]  # Use first 2 rows
        result = process_unified_alerts_integration(test_df, mock_app_utils)
        
        # Verify alert columns are added
        expected_alert_columns = [
            'percentile_category', 'threshold_alert', 'combined_alert', 'ns_reason', 'strata_abbr',
            'total_sessions_alert', 'stage_sessions_alert', 'water_day_total_alert'
        ]
        for col in expected_alert_columns:
            self.assertIn(col, result.columns)
        
        # Verify alert integration
        self.assertEqual(result.loc[result['subject_id'] == realistic_subjects[0], 'percentile_category'].iloc[0], 'SB')
        self.assertEqual(result.loc[result['subject_id'] == realistic_subjects[1], 'percentile_category'].iloc[0], 'G')

    def test_process_unified_alerts_integration_with_threshold_config(self):
        """Test unified alerts integration with threshold configuration using realistic data"""
        # Mock app_utils with proper alert_coordinator structure
        mock_app_utils = Mock()
        mock_app_utils.alert_coordinator = Mock()
        mock_app_utils.alert_coordinator.alert_service = None
        mock_app_utils.reference_processor = Mock()
        mock_app_utils.get_unified_alerts.return_value = {}
        mock_app_utils.get_subject_sessions.return_value = pd.DataFrame({
            'session': [1, 2, 3, 4, 5]
        })
        
        threshold_config = {
            'session': {'condition': 'gt', 'value': 3}
        }
        stage_thresholds = {
            'STAGE_2': 4,
            'STAGE_4': 6
        }
        
        test_df = self.sample_session_data.copy()[:3]  # Use first 3 rows
        
        # Test that the function handles the alert processing gracefully
        result = process_unified_alerts_integration(
            test_df, mock_app_utils, 
            threshold_config=threshold_config, 
            stage_thresholds=stage_thresholds
        )
        
        # Verify that alert columns are present with default values
        expected_alert_columns = [
            'percentile_category', 'threshold_alert', 'combined_alert', 'ns_reason', 'strata_abbr',
            'total_sessions_alert', 'stage_sessions_alert', 'water_day_total_alert'
        ]
        for col in expected_alert_columns:
            self.assertIn(col, result.columns)
        


    def test_process_unified_alerts_integration_combined_alerts(self):
        """Test unified alerts integration with combined alert logic using realistic data"""
        # Mock app_utils with comprehensive alert data
        mock_app_utils = Mock()
        mock_app_utils.alert_service = Mock()
        realistic_subjects = self.sample_session_data['subject_id'].unique()[:3].tolist()
        mock_app_utils.get_unified_alerts.return_value = {
            realistic_subjects[0]: {
                'alert_category': 'B',
                'ns_reason': None,
                'threshold': {'threshold_alert': 'T', 'specific_alerts': {
                    'total_sessions': {'alert': 'T', 'value': 50},
                    'water_day_total': {'alert': 'T', 'value': 1.2}
                }}
            },
            realistic_subjects[1]: {
                'alert_category': 'G',
                'ns_reason': None,
                'threshold': {'threshold_alert': 'N'}
            },
            realistic_subjects[2]: {
                'alert_category': 'NS',
                'ns_reason': 'Insufficient data',
                'threshold': {'threshold_alert': 'N'}
            }
        }
        
        test_df = self.sample_session_data.copy()[:3]  # Use first 3 rows
        result = process_unified_alerts_integration(test_df, mock_app_utils)
        
        # Verify combined alert logic
        self.assertIn('combined_alert', result.columns)
        
        # Subject with threshold alerts should have combined alert
        subject_0_combined = result.loc[result['subject_id'] == realistic_subjects[0], 'combined_alert'].iloc[0]
        self.assertIn('B', subject_0_combined)  # Should include percentile category
        self.assertIn('T', subject_0_combined)  # Should include threshold alert


class TestDataFrameBusinessLogicIntegration(unittest.TestCase):
    """Integration tests for dataframe business logic using realistic fixtures"""

    def setUp(self):
        """Set up test fixtures using realistic sample data"""
        self.sample_data = get_realistic_session_data()
        # Create mock app_utils that returns realistic data
        self.mock_app_utils = Mock()
        self.mock_app_utils.get_table_display_data.return_value = self.sample_data.to_dict('records')

    def test_end_to_end_business_logic_chain(self):
        """Test complete business logic chain using realistic data"""
        # Start with realistic session data
        test_df = self.sample_data.copy()
        
        # Apply the business logic chain
        result = format_strata_abbreviations(test_df)
        result = process_unified_alerts_integration(result, self.mock_app_utils)
        
        # Verify the complete transformation
        self.assertIn('strata_abbr', result.columns)
        self.assertIn('percentile_category', result.columns)
        self.assertIn('combined_alert', result.columns)
        
        # Verify data integrity is maintained
        self.assertEqual(len(result), len(test_df))
        self.assertEqual(set(result['subject_id']), set(test_df['subject_id']))


if __name__ == '__main__':
    unittest.main() 