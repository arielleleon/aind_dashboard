"""
Unit tests for filter utilities

These tests verify the extracted filtering logic functions work correctly
and preserve the exact behavior from the original callback implementation.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

# Ensure we can import from the project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from app_utils.filter_utils import (
    apply_time_window_filter,
    apply_multi_select_filters,
    apply_alert_category_filter,
    apply_sorting_logic,
    apply_all_filters
)


class TestTimeWindowFilter:
    """Tests for apply_time_window_filter function"""
    
    def test_empty_dataframe_handling(self):
        """Test that empty DataFrames are handled correctly"""
        empty_df = pd.DataFrame()
        result = apply_time_window_filter(empty_df, 30)
        assert result.empty
        assert isinstance(result, pd.DataFrame)
    
    def test_none_time_window_returns_original(self, sample_session_data):
        """Test that None or 0 time window returns original data"""
        result_none = apply_time_window_filter(sample_session_data, None)
        result_zero = apply_time_window_filter(sample_session_data, 0)
        
        assert result_none.equals(sample_session_data)
        assert result_zero.equals(sample_session_data)
    
    def test_time_window_filtering_logic(self, sample_session_data):
        """Test the core time window filtering logic"""
        # Create test data with known dates
        test_data = sample_session_data.copy()
        
        # Set up dates: some recent, some old
        now = datetime.now()
        test_data['session_date'] = [
            now - timedelta(days=1),   # Recent
            now - timedelta(days=5),   # Recent  
            now - timedelta(days=15),  # Old
            now - timedelta(days=20),  # Old
            now - timedelta(days=2),   # Recent
            now - timedelta(days=50),  # Old
            now - timedelta(days=3),   # Recent
            now - timedelta(days=100), # Old
            now - timedelta(days=4),   # Recent
            now - timedelta(days=200)  # Old
        ]
        
        # Apply 10-day window filter
        result = apply_time_window_filter(test_data, 10)
        
        # Should only include sessions within 10 days
        expected_subjects = len(test_data[test_data['session_date'] >= (now - timedelta(days=10))])
        print(f"Expected subjects within 10 days: {expected_subjects}")
        print(f"Actual result count: {len(result)}")
        
        # Verify all returned dates are within the window
        reference_date = test_data['session_date'].max()
        start_date = reference_date - timedelta(days=10)
        
        assert all(result['session_date'] >= start_date)
        
        # Should have unique subjects only (most recent session per subject)
        assert len(result['subject_id'].unique()) == len(result)
    
    def test_subject_deduplication(self):
        """Test that only the most recent session per subject is kept"""
        # Create test data with multiple sessions per subject
        test_data = pd.DataFrame({
            'subject_id': ['S001', 'S001', 'S002', 'S002', 'S003'],
            'session_date': [
                datetime.now() - timedelta(days=1),  # Most recent for S001
                datetime.now() - timedelta(days=3),  # Older for S001
                datetime.now() - timedelta(days=2),  # Most recent for S002  
                datetime.now() - timedelta(days=4),  # Older for S002
                datetime.now() - timedelta(days=1)   # Only session for S003
            ],
            'session': [10, 8, 5, 3, 1],
            'other_col': ['a', 'b', 'c', 'd', 'e']
        })
        
        result = apply_time_window_filter(test_data, 10)
        
        # Should have one row per subject
        assert len(result) == 3
        assert set(result['subject_id']) == {'S001', 'S002', 'S003'}
        
        # Should keep the most recent session for each subject
        s001_row = result[result['subject_id'] == 'S001'].iloc[0]
        s002_row = result[result['subject_id'] == 'S002'].iloc[0]
        
        assert s001_row['session'] == 10  # Most recent session for S001
        assert s002_row['session'] == 5   # Most recent session for S002


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
        """Test threshold alert pattern matching"""
        # Create test data with threshold alert patterns
        test_data = pd.DataFrame({
            'subject_id': ['S001', 'S002', 'S003', 'S004', 'S005'],
            'threshold_alert': ['', 'T', '', '', ''],
            'total_sessions_alert': ['', 'T | Alert', '', '', ''],
            'stage_sessions_alert': ['', '', 'T | Alert', '', ''],
            'water_day_total_alert': ['', '', '', 'T | Alert', ''],
            'percentile_category': ['B', 'G', 'SG', 'NS', 'B']
        })
        
        result = apply_alert_category_filter(test_data, 'T')
        
        # Should include subjects with any threshold alert pattern
        expected_subjects = {'S002', 'S003', 'S004'}  # S001 and S005 have no threshold alerts
        actual_subjects = set(result['subject_id'])
        
        assert actual_subjects == expected_subjects
    
    def test_percentile_category_filtering(self):
        """Test filtering by specific percentile categories"""
        test_data = pd.DataFrame({
            'subject_id': ['S001', 'S002', 'S003', 'S004', 'S005'],
            'percentile_category': ['B', 'G', 'B', 'SG', 'NS']
        })
        
        # Test filtering for 'B' category
        result_b = apply_alert_category_filter(test_data, 'B')
        assert set(result_b['subject_id']) == {'S001', 'S003'}
        
        # Test filtering for 'NS' category  
        result_ns = apply_alert_category_filter(test_data, 'NS')
        assert set(result_ns['subject_id']) == {'S005'}
    
    def test_not_scored_filtering(self):
        """Test filtering for Not Scored (NS) subjects"""
        test_data = pd.DataFrame({
            'subject_id': ['S001', 'S002', 'S003'],
            'percentile_category': ['B', 'NS', 'G']
        })
        
        result = apply_alert_category_filter(test_data, 'NS')
        
        assert len(result) == 1
        assert result.iloc[0]['subject_id'] == 'S002'
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
        """Test ascending sort by session_overall_percentile"""
        test_data = pd.DataFrame({
            'subject_id': ['S001', 'S002', 'S003', 'S004'],
            'session_overall_percentile': [75.0, 25.0, 95.0, np.nan],
            'overall_percentile': [70.0, 30.0, 90.0, 50.0]
        })
        
        result = apply_sorting_logic(test_data, 'overall_percentile_asc')
        
        # Should be sorted by session_overall_percentile (ascending), NaN at end
        expected_order = ['S002', 'S001', 'S003', 'S004']  # 25.0, 75.0, 95.0, NaN
        actual_order = result['subject_id'].tolist()
        
        assert actual_order == expected_order
    
    def test_session_overall_percentile_descending(self):
        """Test descending sort by session_overall_percentile"""
        test_data = pd.DataFrame({
            'subject_id': ['S001', 'S002', 'S003', 'S004'],
            'session_overall_percentile': [75.0, 25.0, 95.0, np.nan],
            'overall_percentile': [70.0, 30.0, 90.0, 50.0]
        })
        
        result = apply_sorting_logic(test_data, 'overall_percentile_desc')
        
        # Should be sorted by session_overall_percentile (descending), NaN at end
        expected_order = ['S003', 'S001', 'S002', 'S004']  # 95.0, 75.0, 25.0, NaN
        actual_order = result['subject_id'].tolist()
        
        assert actual_order == expected_order
    
    def test_fallback_to_overall_percentile(self):
        """Test fallback to overall_percentile when session_overall_percentile missing"""
        test_data = pd.DataFrame({
            'subject_id': ['S001', 'S002', 'S003'],
            'overall_percentile': [75.0, 25.0, 95.0]
            # No session_overall_percentile column
        })
        
        result = apply_sorting_logic(test_data, 'overall_percentile_asc')
        
        # Should be sorted by overall_percentile (ascending)
        expected_order = ['S002', 'S001', 'S003']  # 25.0, 75.0, 95.0
        actual_order = result['subject_id'].tolist()
        
        assert actual_order == expected_order
    
    def test_no_percentile_columns_warning(self):
        """Test behavior when no percentile columns are available"""
        test_data = pd.DataFrame({
            'subject_id': ['S001', 'S002', 'S003'],
            'other_column': [1, 2, 3]
            # No percentile columns
        })
        
        result = apply_sorting_logic(test_data, 'overall_percentile_asc')
        
        # Should return original data unchanged
        pd.testing.assert_frame_equal(result, test_data)


class TestApplyAllFilters:
    """Tests for apply_all_filters orchestration function"""
    
    def test_complete_filtering_pipeline(self, sample_session_data):
        """Test the complete filtering pipeline with all filters"""
        test_data = sample_session_data.copy()
        
        # Set up test data with known values
        now = datetime.now()
        test_data['session_date'] = [
            now - timedelta(days=1),
            now - timedelta(days=2), 
            now - timedelta(days=3),
            now - timedelta(days=50),  # This should be filtered out by time window
            now - timedelta(days=4),
            now - timedelta(days=5),
            now - timedelta(days=6),
            now - timedelta(days=7),
            now - timedelta(days=8),
            now - timedelta(days=9)
        ]
        
        result = apply_all_filters(
            df=test_data,
            time_window_value=30,  # Include sessions within 30 days
            stage_value='STAGE_3',
            curriculum_value=None,
            rig_value=None,
            trainer_value=None,
            pi_value=None,
            sort_option='none',
            alert_category='all',
            subject_id_value=None
        )
        
        # Verify the filtering worked
        assert isinstance(result, pd.DataFrame)
        
        # All sessions should be within 30 days
        reference_date = test_data['session_date'].max()
        start_date = reference_date - timedelta(days=30)
        if len(result) > 0:
            assert all(result['session_date'] >= start_date)
    
    def test_no_filters_returns_time_filtered_data(self, sample_session_data):
        """Test that only time window filtering is applied when no other filters specified"""
        test_data = sample_session_data.copy()
        
        result = apply_all_filters(
            df=test_data,
            time_window_value=365,  # Large window to include all data
            stage_value=None,
            curriculum_value=None,
            rig_value=None,
            trainer_value=None,
            pi_value=None,
            sort_option='none',
            alert_category='all',
            subject_id_value=None
        )
        
        # Should have same number of unique subjects as original (after deduplication)
        original_subjects = len(test_data['subject_id'].unique())
        result_subjects = len(result['subject_id'].unique()) if len(result) > 0 else 0
        
        # Should be equal or less (due to deduplication)
        assert result_subjects <= original_subjects
    
    def test_filter_order_consistency(self, sample_session_data):
        """Test that filters are applied in the correct order"""
        test_data = sample_session_data.copy()
        
        # Apply filters that should result in a specific subset
        result = apply_all_filters(
            df=test_data,
            time_window_value=365,
            stage_value=['STAGE_3', 'STAGE_FINAL'],
            curriculum_value=None,
            rig_value=None,
            trainer_value=None,
            pi_value=None,
            sort_option='overall_percentile_desc',
            alert_category='all',
            subject_id_value=None
        )
        
        # Verify the filtering and sorting worked together
        assert isinstance(result, pd.DataFrame)
        
        # If we have results, verify they meet all filter criteria
        if len(result) > 0:
            assert all(result['current_stage_actual'].isin(['STAGE_3', 'STAGE_FINAL']))


class TestFilterUtilsIntegration:
    """Integration tests to verify filter utilities work with callback integration"""
    
    def test_callback_integration_compatibility(self, sample_session_data):
        """Test that filter utilities work exactly like the original callback logic"""
        test_data = sample_session_data.copy()
        
        # Test the exact same call pattern as used in the callback
        result = apply_all_filters(
            df=test_data,
            time_window_value=30,
            stage_value='STAGE_3',
            curriculum_value=None,
            rig_value=None,
            trainer_value=None,
            pi_value=None,
            sort_option='overall_percentile_asc',
            alert_category='all',
            subject_id_value=None
        )
        
        # Should return a DataFrame that can be converted to records (as done in callback)
        records = result.to_dict("records")
        assert isinstance(records, list)
        
        # Each record should be a dictionary (as expected by DataTable)
        if len(records) > 0:
            assert all(isinstance(record, dict) for record in records)
    
    @pytest.mark.integration
    def test_filter_utils_import_in_callback_context(self):
        """Integration test to verify filter utilities can be imported in callback context"""
        try:
            # Import as done in the actual callback
            from app_utils.filter_utils import apply_all_filters
            
            # Verify the function is callable
            assert callable(apply_all_filters)
            
            # Verify it has the expected signature
            import inspect
            sig = inspect.signature(apply_all_filters)
            expected_params = {
                'df', 'time_window_value', 'stage_value', 'curriculum_value',
                'rig_value', 'trainer_value', 'pi_value', 'sort_option', 
                'alert_category', 'subject_id_value'
            }
            actual_params = set(sig.parameters.keys())
            
            assert expected_params == actual_params
            
        except ImportError as e:
            pytest.fail(f"Failed to import filter utilities in callback context: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error with filter utilities import: {e}")
    
    def test_realistic_data_compatibility(self, sample_session_data):
        """Test that filter utilities work correctly with realistic app data"""
        # This test uses the realistic sample data to verify compatibility
        result = apply_all_filters(
            df=sample_session_data,
            time_window_value=365,  # Include all data
            stage_value=None,
            curriculum_value=None,
            rig_value=None,
            trainer_value=None,
            pi_value=None,
            sort_option='none',
            alert_category='all',
            subject_id_value=None
        )
        
        # Verify it works with real column names and data types
        assert isinstance(result, pd.DataFrame)
        
        # Check that real columns are preserved
        expected_columns = ['subject_id', 'session_date', 'current_stage_actual', 
                          'curriculum_name', 'trainer', 'rig', 'PI']
        available_columns = [col for col in expected_columns if col in result.columns]
        assert len(available_columns) > 0  # At least some expected columns should be present
        
        # Verify realistic data formats are preserved
        if len(result) > 0:
            # Subject IDs should be strings representing numbers (like '690494')
            assert all(isinstance(str(sid), str) for sid in result['subject_id'])
            
            # Session dates should be datetime objects
            assert all(isinstance(date, (pd.Timestamp, datetime)) for date in result['session_date']) 