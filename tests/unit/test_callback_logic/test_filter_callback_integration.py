"""
Integration tests for filter callback refactoring

These tests verify that the refactored filter callback using the extracted
filter utilities produces identical results to the original implementation.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Ensure we can import from the project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


class TestFilterCallbackIntegration:
    """Tests for filter callback integration with extracted utilities"""
    
    def test_update_table_data_callback_signature(self):
        """Test that update_table_data callback has the expected signature"""
        # Import the actual callback
        from callbacks.filter_callbacks import update_table_data
        
        # Verify it's callable
        assert callable(update_table_data)
        
        # Verify the function exists and can be inspected
        import inspect
        sig = inspect.signature(update_table_data)
        
        # Should have the expected parameters
        expected_params = {
            'time_window_value', 'stage_value', 'curriculum_value', 'rig_value',
            'trainer_value', 'pi_value', 'sort_option', 'alert_category', 
            'subject_id_value', 'clear_clicks'
        }
        actual_params = set(sig.parameters.keys())
        
        assert expected_params == actual_params
    
    def test_filter_utils_import_in_callback(self):
        """Test that filter utilities are properly imported in the callback module"""
        # Import the callback module
        import callbacks.filter_callbacks as filter_callbacks
        
        # Verify the apply_all_filters import exists
        assert hasattr(filter_callbacks, 'apply_all_filters')
        
        # Verify it's the correct function
        from app_utils.filter_utils import apply_all_filters
        assert filter_callbacks.apply_all_filters is apply_all_filters
    
    @patch('callbacks.filter_callbacks.app_utils')
    @patch('callbacks.filter_callbacks.app_dataframe')
    def test_callback_uses_filter_utilities(self, mock_app_dataframe, mock_app_utils, sample_session_data):
        """Test that the callback properly uses the extracted filter utilities"""
        # Import the callback function
        from callbacks.filter_callbacks import update_table_data
        
        # Setup mocks to return our test data
        mock_app_utils.get_table_display_data.return_value = sample_session_data.to_dict('records')
        
        # Mock pandas DataFrame creation
        with patch('callbacks.filter_callbacks.pd.DataFrame') as mock_dataframe:
            mock_dataframe.return_value = sample_session_data.copy()
            
            # Mock the apply_all_filters function to track its usage
            with patch('callbacks.filter_callbacks.apply_all_filters') as mock_apply_all_filters:
                mock_apply_all_filters.return_value = sample_session_data.copy()
                
                # Call the callback
                result = update_table_data(
                    time_window_value=30,
                    stage_value='STAGE_3',
                    curriculum_value=None,
                    rig_value=None,
                    trainer_value=None,
                    pi_value=None,
                    sort_option='none',
                    alert_category='all',
                    subject_id_value=None,
                    clear_clicks=None
                )
                
                # Verify apply_all_filters was called with correct parameters
                mock_apply_all_filters.assert_called_once()
                call_args = mock_apply_all_filters.call_args
                
                # Verify the parameters passed to apply_all_filters
                assert call_args.kwargs['time_window_value'] == 30
                assert call_args.kwargs['stage_value'] == 'STAGE_3'
                assert call_args.kwargs['curriculum_value'] is None
                assert call_args.kwargs['sort_option'] == 'none'
                assert call_args.kwargs['alert_category'] == 'all'
                
                # Verify the result is in the correct format (records for DataTable)
                assert isinstance(result, list)
    
    def test_callback_data_flow(self, sample_session_data):
        """Test the complete data flow through the refactored callback"""
        # Import required modules
        from callbacks.filter_callbacks import update_table_data
        from app_utils.filter_utils import apply_all_filters
        
        # Create realistic test data
        test_data = sample_session_data.copy()
        
        # Mock the app_utils to return our test data
        with patch('callbacks.filter_callbacks.app_utils') as mock_app_utils:
            mock_app_utils.get_table_display_data.return_value = test_data.to_dict('records')
            
            # Mock pandas DataFrame creation
            with patch('callbacks.filter_callbacks.pd.DataFrame') as mock_dataframe:
                mock_dataframe.return_value = test_data
                
                # Call the callback - this should exercise the full refactored logic
                result = update_table_data(
                    time_window_value=365,  # Large window to include all data
                    stage_value=None,
                    curriculum_value=None,
                    rig_value=None,
                    trainer_value=None,
                    pi_value=None,
                    sort_option='none',
                    alert_category='all',
                    subject_id_value=None,
                    clear_clicks=None
                )
                
                # Verify the result
                assert isinstance(result, list)
                
                # If we have data, verify the structure
                if len(result) > 0:
                    assert all(isinstance(record, dict) for record in result)
    
    def test_callback_error_handling(self):
        """Test that the callback handles errors gracefully"""
        from callbacks.filter_callbacks import update_table_data
        
        # Mock app_utils to simulate different error conditions
        with patch('callbacks.filter_callbacks.app_utils') as mock_app_utils:
            # Test empty data handling
            mock_app_utils.get_table_display_data.return_value = []
            mock_app_utils._cache = {'formatted_data': None}
            
            with patch('callbacks.filter_callbacks.pd.DataFrame') as mock_dataframe:
                # Create a mock DataFrame that returns empty list from to_dict
                empty_df_mock = Mock()
                empty_df_mock.to_dict.return_value = []
                mock_dataframe.return_value = empty_df_mock
                
                # Mock apply_all_filters to return the same empty DataFrame
                with patch('callbacks.filter_callbacks.apply_all_filters') as mock_apply_all_filters:
                    mock_apply_all_filters.return_value = empty_df_mock
                    
                    # Should not crash with empty data
                    result = update_table_data(
                        time_window_value=30,
                        stage_value=None,
                        curriculum_value=None,
                        rig_value=None,
                        trainer_value=None,
                        pi_value=None,
                        sort_option='none',
                        alert_category='all',
                        subject_id_value=None,
                        clear_clicks=None
                    )
                    
                    # Should return empty list
                    assert result == []
    
    @pytest.mark.integration  
    def test_callback_with_realistic_filters(self, sample_session_data):
        """Integration test with realistic filter combinations"""
        from callbacks.filter_callbacks import update_table_data
        
        # Mock the dependencies with realistic data
        with patch('callbacks.filter_callbacks.app_utils') as mock_app_utils:
            mock_app_utils.get_table_display_data.return_value = sample_session_data.to_dict('records')
            
            with patch('callbacks.filter_callbacks.pd.DataFrame') as mock_dataframe:
                mock_dataframe.return_value = sample_session_data.copy()
                
                # Test various realistic filter combinations
                test_cases = [
                    # Basic time window
                    {
                        'time_window_value': 30,
                        'stage_value': None,
                        'curriculum_value': None,
                        'rig_value': None,
                        'trainer_value': None,
                        'pi_value': None,
                        'sort_option': 'none',
                        'alert_category': 'all',
                        'subject_id_value': None,
                        'clear_clicks': None
                    },
                    # Stage filter
                    {
                        'time_window_value': 365,
                        'stage_value': 'STAGE_3',
                        'curriculum_value': None,
                        'rig_value': None,
                        'trainer_value': None,
                        'pi_value': None,
                        'sort_option': 'none',
                        'alert_category': 'all',
                        'subject_id_value': None,
                        'clear_clicks': None
                    },
                    # Multiple filters + sorting
                    {
                        'time_window_value': 90,
                        'stage_value': ['STAGE_3', 'STAGE_FINAL'],
                        'curriculum_value': 'Coupled Baiting',
                        'rig_value': None,
                        'trainer_value': None,
                        'pi_value': None,
                        'sort_option': 'overall_percentile_desc',
                        'alert_category': 'all',
                        'subject_id_value': None,
                        'clear_clicks': None
                    }
                ]
                
                # Test each case
                for i, test_case in enumerate(test_cases):
                    try:
                        result = update_table_data(**test_case)
                        
                        # Each should return a list of records
                        assert isinstance(result, list), f"Test case {i} failed: result is not a list"
                        
                        # If we have results, verify the structure
                        if len(result) > 0:
                            assert all(isinstance(record, dict) for record in result), \
                                f"Test case {i} failed: not all results are dictionaries"
                                
                    except Exception as e:
                        pytest.fail(f"Test case {i} failed with exception: {e}")


class TestOriginalVsRefactoredBehavior:
    """Tests to verify refactored code produces identical results to original"""
    
    def test_time_window_logic_preservation(self):
        """Test that time window logic produces identical results"""
        # Create test data that would exercise the original time window logic
        test_data = pd.DataFrame({
            'subject_id': ['S001', 'S001', 'S002', 'S002', 'S003'],
            'session_date': [
                datetime.now() - timedelta(days=1),
                datetime.now() - timedelta(days=5),
                datetime.now() - timedelta(days=2), 
                datetime.now() - timedelta(days=10),
                datetime.now() - timedelta(days=3)
            ]
        })
        
        # Apply our refactored logic
        from app_utils.filter_utils import apply_time_window_filter
        refactored_result = apply_time_window_filter(test_data, 7)
        
        # Apply the original logic manually (extracted from original callback)
        reference_date = test_data['session_date'].max()
        start_date = reference_date - timedelta(days=7)
        time_filtered = test_data[test_data['session_date'] >= start_date]
        time_filtered = time_filtered.sort_values('session_date', ascending=False)
        original_result = time_filtered.drop_duplicates(subset=['subject_id'], keep='first')
        
        # Results should be identical
        pd.testing.assert_frame_equal(refactored_result.reset_index(drop=True), 
                                    original_result.reset_index(drop=True))
    
    def test_multi_filter_logic_preservation(self):
        """Test that multi-filter logic produces identical results"""
        test_data = pd.DataFrame({
            'subject_id': ['S001', 'S002', 'S003', 'S004'],
            'current_stage_actual': ['STAGE_1', 'STAGE_2', 'STAGE_1', 'STAGE_3'],
            'curriculum_name': ['CurrA', 'CurrB', 'CurrA', 'CurrC'],
            'trainer': ['Alice', 'Bob', 'Alice', 'Charlie']
        })
        
        # Apply our refactored logic
        from app_utils.filter_utils import apply_multi_select_filters
        refactored_result = apply_multi_select_filters(test_data, {
            'stage': 'STAGE_1',
            'curriculum': 'CurrA'
        })
        
        # Apply the original logic manually
        original_filtered = test_data.copy()
        original_filtered = original_filtered[original_filtered['current_stage_actual'] == 'STAGE_1']
        original_filtered = original_filtered[original_filtered['curriculum_name'] == 'CurrA']
        
        # Results should be identical
        pd.testing.assert_frame_equal(refactored_result.reset_index(drop=True),
                                    original_filtered.reset_index(drop=True))
    
    def test_sorting_logic_preservation(self):
        """Test that sorting logic produces identical results"""
        test_data = pd.DataFrame({
            'subject_id': ['S001', 'S002', 'S003', 'S004'],
            'session_overall_percentile': [75.0, 25.0, 95.0, np.nan],
            'overall_percentile': [70.0, 30.0, 90.0, 50.0]
        })
        
        # Apply our refactored logic
        from app_utils.filter_utils import apply_sorting_logic
        refactored_result = apply_sorting_logic(test_data, 'overall_percentile_asc')
        
        # Apply the original logic manually (from original callback)
        original_filtered = test_data.copy()
        if 'session_overall_percentile' in original_filtered.columns:
            original_filtered = original_filtered.sort_values(
                "session_overall_percentile", ascending=True, na_position='last'
            )
        
        # Results should be identical
        pd.testing.assert_frame_equal(refactored_result.reset_index(drop=True),
                                    original_filtered.reset_index(drop=True))
    
    def test_complete_pipeline_preservation(self, sample_session_data):
        """Test that the complete filtering pipeline preserves behavior"""
        test_data = sample_session_data.copy()
        
        # Make dates recent so time window doesn't filter everything out
        now = datetime.now()
        test_data['session_date'] = [
            now - timedelta(days=i) for i in range(len(test_data))
        ]
        
        # Apply our refactored complete pipeline
        from app_utils.filter_utils import apply_all_filters
        refactored_result = apply_all_filters(
            df=test_data,
            time_window_value=365,
            stage_value='STAGE_3',
            curriculum_value=None,
            rig_value=None,
            trainer_value=None,
            pi_value=None,
            sort_option='overall_percentile_asc',
            alert_category='all',
            subject_id_value=None
        )
        
        # Verify the result has the expected characteristics
        assert isinstance(refactored_result, pd.DataFrame)
        
        # All subjects should have STAGE_3 (if any results)
        if len(refactored_result) > 0:
            assert all(refactored_result['current_stage_actual'] == 'STAGE_3')
            
        # Should be sortable by percentile (no errors)
        if 'session_overall_percentile' in refactored_result.columns and len(refactored_result) > 1:
            percentiles = refactored_result['session_overall_percentile'].dropna()
            if len(percentiles) > 1:
                # Should be in ascending order
                assert all(percentiles.iloc[i] <= percentiles.iloc[i+1] 
                          for i in range(len(percentiles)-1)) 