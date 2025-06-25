"""
Unit tests for filter utility functions

These tests verify the filtering logic used throughout the AIND Dashboard,
ensuring that data filtering operations work correctly with various inputs.

Updated to use realistic fixtures from sample_data.py for consistency.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from app_utils.filter_utils import (
    apply_time_window_filter,
    apply_multi_select_filters,
    apply_alert_category_filter,
    apply_sorting_logic,
    apply_all_filters
)

# Import realistic fixtures
from tests.fixtures.sample_data import get_realistic_session_data, get_simple_session_data


class TestTimeWindowFilter:
    """Tests for apply_time_window_filter function"""
    
    def test_empty_dataframe_handling(self):
        """Test that empty DataFrames are handled correctly"""
        empty_df = pd.DataFrame()
        result = apply_time_window_filter(empty_df, 30)
        assert result.empty
        assert isinstance(result, pd.DataFrame)
    
    def test_none_time_window_returns_original(self, sample_session_data):
        """Test that None time window returns original data unchanged"""
        result = apply_time_window_filter(sample_session_data, None)
        pd.testing.assert_frame_equal(result, sample_session_data)
    
    def test_time_window_filtering_logic(self, sample_session_data):
        """Test time window filtering with realistic session data"""
        # Use realistic data and test with a 30-day window
        result = apply_time_window_filter(sample_session_data, 30)
        
        # Should have some data (the fixture has recent dates)
        assert isinstance(result, pd.DataFrame)
        
        # All remaining data should be within the time window
        if len(result) > 0:
            cutoff_date = datetime.now() - timedelta(days=30)
            assert all(result['session_date'] >= cutoff_date)
    
    def test_time_window_filtering_with_no_matches(self, sample_session_data):
        """Test time window filtering when no data matches"""
        # Use a very small time window that should exclude all fixture data
        result = apply_time_window_filter(sample_session_data, 1)  # 1 day window
        
        # Should return empty DataFrame with correct structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == list(sample_session_data.columns)
    
    def test_large_time_window_returns_deduplicated_data(self, sample_session_data):
        """Test that very large time window returns deduplicated data (most recent session per subject)"""
        result = apply_time_window_filter(sample_session_data, 1000)  # Large window
        
        # Should return deduplicated data (one session per subject)
        # The fixture has 10 sessions for 5 unique subjects, so should return 5
        unique_subjects = len(sample_session_data['subject_id'].unique())
        assert len(result) == unique_subjects
    
    def test_duplicate_session_handling(self):
        """Test that only the most recent session per subject is kept"""
        # Create test data with multiple sessions per subject using realistic structure
        base_data = get_realistic_session_data()
        test_data = pd.DataFrame({
            'subject_id': ['690494', '690494', '690486', '690486', '702200'],
            'session_date': [
                datetime.now() - timedelta(days=1),  # Most recent for 690494
                datetime.now() - timedelta(days=3),  # Older for 690494
                datetime.now() - timedelta(days=2),  # Most recent for 690486  
                datetime.now() - timedelta(days=4),  # Older for 690486
                datetime.now() - timedelta(days=1)   # Only session for 702200
            ],
            'session': [10, 8, 5, 3, 1],
            'strata': [
                'Uncoupled Without Baiting_BEGINNER_v1',
                'Uncoupled Without Baiting_BEGINNER_v1', 
                'Uncoupled Without Baiting_INTERMEDIATE_v1',
                'Uncoupled Without Baiting_INTERMEDIATE_v1',
                'Coupled Baiting_INTERMEDIATE_v1'
            ]
        })
        
        result = apply_time_window_filter(test_data, 10)
        
        # Should have one row per subject
        assert len(result) == 3
        assert set(result['subject_id']) == {'690494', '690486', '702200'}
        
        # Should keep the most recent session for each subject
        subj_690494_row = result[result['subject_id'] == '690494'].iloc[0]
        subj_690486_row = result[result['subject_id'] == '690486'].iloc[0]
        
        assert subj_690494_row['session'] == 10  # Most recent session for 690494
        assert subj_690486_row['session'] == 5   # Most recent session for 690486

class TestMultiSelectFilters:
    """Tests for apply_multi_select_filters function"""
    
    def test_empty_dataframe_handling(self):
        """Test that empty DataFrames are handled correctly"""
        empty_df = pd.DataFrame()
        result = apply_multi_select_filters(empty_df, {})
        assert result.empty
        assert isinstance(result, pd.DataFrame)
    
    def test_empty_filter_configs_returns_original(self, sample_session_data):
        """Test that empty filter configs return original data"""
        result = apply_multi_select_filters(sample_session_data, {})
        pd.testing.assert_frame_equal(result, sample_session_data)
    
    def test_single_stage_filter(self, sample_session_data):
        """Test filtering by a single stage value"""
        test_data = sample_session_data.copy()
        
        filter_configs = {'stage': 'STAGE_3'}
        result = apply_multi_select_filters(test_data, filter_configs)
        
        # All results should have the specified stage
        if len(result) > 0:
            assert all(result['current_stage_actual'] == 'STAGE_3')
    
    def test_multi_stage_filter(self, sample_session_data):
        """Test filtering by multiple stage values"""
        test_data = sample_session_data.copy()
        
        filter_configs = {'stage': ['STAGE_1', 'STAGE_3']}
        result = apply_multi_select_filters(test_data, filter_configs)
        
        # All results should have one of the specified stages
        if len(result) > 0:
            assert all(result['current_stage_actual'].isin(['STAGE_1', 'STAGE_3']))
    
    def test_subject_id_filter_single(self, sample_session_data):
        """Test filtering by single subject ID"""
        test_data = sample_session_data.copy()
        target_subject = test_data.iloc[0]['subject_id']
        
        filter_configs = {'subject_id': target_subject}
        result = apply_multi_select_filters(test_data, filter_configs)
        
        # All results should have the specified subject ID
        assert all(result['subject_id'].astype(str) == str(target_subject))
    
    def test_subject_id_filter_multiple(self, sample_session_data):
        """Test filtering by multiple subject IDs"""
        test_data = sample_session_data.copy()
        target_subjects = test_data['subject_id'].unique()[:2].tolist()
        
        filter_configs = {'subject_id': target_subjects}
        result = apply_multi_select_filters(test_data, filter_configs)
        
        # All results should have one of the specified subject IDs
        target_subjects_str = [str(sid) for sid in target_subjects]
        assert all(result['subject_id'].astype(str).isin(target_subjects_str))
    
    def test_combined_filters(self, sample_session_data):
        """Test multiple filters applied together"""
        test_data = sample_session_data.copy()
        
        filter_configs = {
            'stage': 'STAGE_3',
            'curriculum': 'Coupled Baiting',
            'trainer': 'Ella Hilton'
        }
        result = apply_multi_select_filters(test_data, filter_configs)
        
        # Apply each filter condition and verify
        if len(result) > 0:
            assert all(result['current_stage_actual'] == 'STAGE_3')
            assert all(result['curriculum_name'] == 'Coupled Baiting')
            assert all(result['trainer'] == 'Ella Hilton')
    
    def test_no_matching_data(self, sample_session_data):
        """Test behavior when no data matches the filters"""
        test_data = sample_session_data.copy()
        
        filter_configs = {'stage': 'NONEXISTENT_STAGE'}
        result = apply_multi_select_filters(test_data, filter_configs)
        
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)


class TestAlertCategoryFilter:
    """Tests for apply_alert_category_filter function"""
    
    def test_empty_dataframe_handling(self):
        """Test that empty DataFrames are handled correctly"""
        empty_df = pd.DataFrame()
        result = apply_alert_category_filter(empty_df, 'T')
        assert result.empty
        assert isinstance(result, pd.DataFrame)
    
    def test_all_category_returns_original(self, sample_session_data):
        """Test that 'all' category returns original data"""
        result = apply_alert_category_filter(sample_session_data, 'all')
        pd.testing.assert_frame_equal(result, sample_session_data)
    
    def test_threshold_alert_filtering(self):
        """Test threshold alert pattern matching using realistic subject IDs"""
        # Create test data using realistic subject IDs
        base_data = get_realistic_session_data()
        subject_ids = base_data['subject_id'].unique()[:5].tolist()
        
        test_data = pd.DataFrame({
            'subject_id': subject_ids,
            'threshold_alert': ['', 'T', '', '', ''],
            'total_sessions_alert': ['', 'T | Alert', '', '', ''],
            'stage_sessions_alert': ['', '', 'T | Alert', '', ''],
            'water_day_total_alert': ['', '', '', 'T | Alert', ''],
            'percentile_category': ['B', 'G', 'SG', 'NS', 'B']
        })
        
        result = apply_alert_category_filter(test_data, 'T')
        
        # Should include subjects with any threshold alert pattern
        expected_subjects = {subject_ids[1], subject_ids[2], subject_ids[3]}
        actual_subjects = set(result['subject_id'])
        
        assert actual_subjects == expected_subjects
    
    def test_percentile_category_filtering(self):
        """Test filtering by specific percentile categories using realistic subject IDs"""
        base_data = get_realistic_session_data()
        subject_ids = base_data['subject_id'].unique()[:5].tolist()
        
        test_data = pd.DataFrame({
            'subject_id': subject_ids,
            'percentile_category': ['B', 'G', 'B', 'SG', 'NS']
        })
        
        # Test filtering for 'B' category
        result_b = apply_alert_category_filter(test_data, 'B')
        assert set(result_b['subject_id']) == {subject_ids[0], subject_ids[2]}
        
        # Test filtering for 'NS' category  
        result_ns = apply_alert_category_filter(test_data, 'NS')
        assert set(result_ns['subject_id']) == {subject_ids[4]}
    
    def test_not_scored_filtering(self):
        """Test filtering for Not Scored (NS) subjects using realistic subject IDs"""
        base_data = get_realistic_session_data()
        subject_ids = base_data['subject_id'].unique()[:3].tolist()
        
        test_data = pd.DataFrame({
            'subject_id': subject_ids,
            'percentile_category': ['B', 'NS', 'G']
        })
        
        result = apply_alert_category_filter(test_data, 'NS')
        
        assert len(result) == 1
        assert result.iloc[0]['subject_id'] == subject_ids[1]
        assert result.iloc[0]['percentile_category'] == 'NS'


class TestSortingLogic:
    """Tests for apply_sorting_logic function"""
    
    def test_empty_dataframe_handling(self):
        """Test that empty DataFrames are handled correctly"""
        empty_df = pd.DataFrame()
        result = apply_sorting_logic(empty_df, 'overall_percentile_asc')
        assert result.empty
        assert isinstance(result, pd.DataFrame)
    
    def test_none_sorting_returns_original(self, sample_session_data):
        """Test that 'none' sorting returns original data"""
        result = apply_sorting_logic(sample_session_data, 'none')
        pd.testing.assert_frame_equal(result, sample_session_data)
    
    def test_session_overall_percentile_ascending(self):
        """Test sorting by session overall percentile ascending using realistic data"""
        # Create test data using realistic structure
        test_data = pd.DataFrame({
            'subject_id': ['690494', '690486', '702200', '697929'],
            'session_overall_percentile': [75.5, 25.3, 85.2, 45.1]
        })
        
        result = apply_sorting_logic(test_data, 'overall_percentile_asc')
        
        # Should be sorted in ascending order by session_overall_percentile
        percentiles = result['session_overall_percentile'].values
        assert all(percentiles[i] <= percentiles[i+1] for i in range(len(percentiles)-1))
    
    def test_session_overall_percentile_descending(self):
        """Test sorting by session overall percentile descending using realistic data"""
        # Create test data using realistic structure
        test_data = pd.DataFrame({
            'subject_id': ['690494', '690486', '702200', '697929'],
            'session_overall_percentile': [75.5, 25.3, 85.2, 45.1]
        })
        
        result = apply_sorting_logic(test_data, 'overall_percentile_desc')
        
        # Should be sorted in descending order by session_overall_percentile
        percentiles = result['session_overall_percentile'].values
        assert all(percentiles[i] >= percentiles[i+1] for i in range(len(percentiles)-1))
    
    def test_fallback_to_overall_percentile(self):
        """Test fallback when session_overall_percentile column doesn't exist"""
        # Create test data without session_overall_percentile column
        test_data = pd.DataFrame({
            'subject_id': ['690494', '690486', '702200'],
            'overall_percentile': [75.5, 25.3, 85.2]
        })
        
        result = apply_sorting_logic(test_data, 'overall_percentile_asc')
        
        # Should fallback to overall_percentile and sort ascending
        percentiles = result['overall_percentile'].values
        assert all(percentiles[i] <= percentiles[i+1] for i in range(len(percentiles)-1))
    
    def test_no_percentile_columns_warning(self):
        """Test warning when neither percentile column exists"""
        # Create test data without any percentile columns
        test_data = pd.DataFrame({
            'subject_id': ['690494', '690486', '702200'],
            'session': [1, 2, 3]
        })
        
        result = apply_sorting_logic(test_data, 'overall_percentile_asc')
        
        # Should return original data unchanged
        pd.testing.assert_frame_equal(result, test_data)


class TestApplyAllFilters:
    """Tests for apply_all_filters function"""
    
    def test_complete_filtering_pipeline(self, sample_session_data):
        """Test the complete filtering pipeline with realistic data"""
        result = apply_all_filters(
            df=sample_session_data,
            time_window_value=365,  # Large window to include fixture data
            stage_value=None,
            curriculum_value='Coupled Baiting',
            rig_value=None,
            trainer_value=None,
            pi_value=None,
            sort_option='overall_percentile_desc',
            alert_category='all',
            subject_id_value=None
        )
        
        # Should return a valid DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # If any data remains, verify filtering worked
        if len(result) > 0:
            # All remaining data should have 'Coupled Baiting' curriculum
            assert all(result['curriculum_name'] == 'Coupled Baiting')
    
    def test_no_filters_returns_time_filtered_data(self, sample_session_data):
        """Test that minimal config applies only time filtering"""
        result = apply_all_filters(
            df=sample_session_data,
            time_window_value=365,  # Large window
            stage_value=None,
            curriculum_value=None,
            rig_value=None,
            trainer_value=None,
            pi_value=None,
            sort_option='none',
            alert_category='all',
            subject_id_value=None
        )
        
        # Should return data with only time filtering applied (deduplicated)
        assert isinstance(result, pd.DataFrame)
        # With a large time window and no other filters, should have some data
        assert len(result) > 0
        # Verify all returned data is within expected subjects
        fixture_subjects = set(sample_session_data['subject_id'].unique())
        result_subjects = set(result['subject_id'].unique())
        assert result_subjects.issubset(fixture_subjects)
    
    def test_filter_order_consistency(self, sample_session_data):
        """Test that filter order produces consistent results"""
        # Apply filters multiple times with same parameters
        result1 = apply_all_filters(
            df=sample_session_data,
            time_window_value=365,
            stage_value=['STAGE_3', 'STAGE_FINAL'],
            curriculum_value=None,
            rig_value=None,
            trainer_value=None,
            pi_value=None,
            sort_option='overall_percentile_asc',
            alert_category='all',
            subject_id_value=None
        )
        result2 = apply_all_filters(
            df=sample_session_data,
            time_window_value=365,
            stage_value=['STAGE_3', 'STAGE_FINAL'],
            curriculum_value=None,
            rig_value=None,
            trainer_value=None,
            pi_value=None,
            sort_option='overall_percentile_asc',
            alert_category='all',
            subject_id_value=None
        )
        result3 = apply_all_filters(
            df=sample_session_data,
            time_window_value=365,
            stage_value=['STAGE_3', 'STAGE_FINAL'],
            curriculum_value=None,
            rig_value=None,
            trainer_value=None,
            pi_value=None,
            sort_option='overall_percentile_asc',
            alert_category='all',
            subject_id_value=None
        )
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)
        pd.testing.assert_frame_equal(result2, result3)


class TestFilterUtilsIntegration:
    """Integration tests for filter utilities"""
    
    def test_callback_integration_compatibility(self, sample_session_data):
        """Test that filter utils work with callback-style inputs"""
        filter_inputs = [
            {
                'time_window_value': 30,
                'stage_value': 'STAGE_3',
                'curriculum_value': None,
                'rig_value': None,
                'trainer_value': None,
                'pi_value': None,
                'sort_option': 'overall_percentile_desc',
                'alert_category': 'all',
                'subject_id_value': None
            },
            {
                'time_window_value': None,
                'stage_value': None,
                'curriculum_value': None,
                'rig_value': None,
                'trainer_value': None,
                'pi_value': None,
                'sort_option': 'none',
                'alert_category': 'all',
                'subject_id_value': None
            }
        ]
        
        for filter_config in filter_inputs:
            result = apply_all_filters(sample_session_data, **filter_config)
            assert isinstance(result, pd.DataFrame)
            expected_columns = set(sample_session_data.columns)
            actual_columns = set(result.columns)
            assert actual_columns == expected_columns
    
    @pytest.mark.integration
    def test_filter_utils_import_in_callback_context(self):
        """Test that filter utils can be imported in callback context"""
        # This simulates importing in a callback module
        try:
            from app_utils.filter_utils import (
                apply_time_window_filter,
                apply_multi_select_filters,
                apply_alert_category_filter,
                apply_sorting_logic,
                apply_all_filters
            )
            
            # Verify all functions are callable
            assert callable(apply_time_window_filter)
            assert callable(apply_multi_select_filters)
            assert callable(apply_alert_category_filter)
            assert callable(apply_sorting_logic)
            assert callable(apply_all_filters)
            
        except ImportError as e:
            pytest.fail(f"Failed to import filter utils in callback context: {e}")
    
    def test_realistic_data_compatibility(self, sample_session_data):
        """Test that all filters work with realistic session data structure"""
        time_filtered = apply_time_window_filter(sample_session_data, 365)
        assert isinstance(time_filtered, pd.DataFrame)
        
        multi_filtered = apply_multi_select_filters(time_filtered, {'curriculum': 'Coupled Baiting'})
        assert isinstance(multi_filtered, pd.DataFrame)
        
        alert_filtered = apply_alert_category_filter(multi_filtered, 'all')
        assert isinstance(alert_filtered, pd.DataFrame)
        
        sorted_data = apply_sorting_logic(alert_filtered, 'overall_percentile_desc')
        assert isinstance(sorted_data, pd.DataFrame)
        
        # Verify the pipeline preserves data integrity
        if len(sorted_data) > 0:
            # Check that essential columns are preserved
            essential_columns = ['subject_id', 'session_date', 'session']
            for col in essential_columns:
                assert col in sorted_data.columns 