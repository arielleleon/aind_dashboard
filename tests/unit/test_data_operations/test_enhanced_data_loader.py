import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from app_utils.app_data_load import EnhancedDataLoader, AppLoadData

class TestEnhancedDataLoader:
    """Test suite for the EnhancedDataLoader class"""

    def test_enhanced_data_loader_initialization(self):
        """Test that EnhancedDataLoader initializes correctly"""
        with patch('app_utils.app_data_load.data_loader.get_session_table') as mock_get_session:
            # Mock the session table
            mock_df = pd.DataFrame({
                'subject_id': ['test_subject_1', 'test_subject_2'],
                'session': [1, 2],
                'session_date': [datetime(2024, 1, 1), datetime(2024, 1, 2)]
            })
            mock_get_session.return_value = mock_df
            
            # Initialize the loader
            loader = EnhancedDataLoader()
            
            # Verify initialization
            assert loader.session_table is not None
            assert loader.last_load_time is not None
            assert loader.load_parameters == {'load_bpod': False}
            assert len(loader.session_table) == 2

    def test_backward_compatibility_alias(self):
        """Test that AppLoadData is an alias for EnhancedDataLoader"""
        with patch('app_utils.app_data_load.data_loader.get_session_table') as mock_get_session:
            mock_df = pd.DataFrame({'subject_id': ['test'], 'session': [1]})
            mock_get_session.return_value = mock_df
            
            # Both should create compatible objects
            enhanced_loader = EnhancedDataLoader()
            app_load_data = AppLoadData()
            
            # AppLoadData should be a subclass of EnhancedDataLoader
            assert isinstance(app_load_data, EnhancedDataLoader)
            
            # Both should have the same interface
            assert hasattr(app_load_data, 'load')
            assert hasattr(app_load_data, 'get_data')
            assert hasattr(app_load_data, 'reload_data')
            assert hasattr(app_load_data, 'get_subject_sessions')
            assert hasattr(app_load_data, 'get_most_recent_subject_sessions')

    def test_load_with_bpod_parameter(self):
        """Test loading data with bpod parameter"""
        with patch('app_utils.app_data_load.data_loader.get_session_table') as mock_get_session:
            mock_df = pd.DataFrame({
                'subject_id': ['test_subject'],
                'session': [1],
                'session_date': [datetime(2024, 1, 1)]
            })
            mock_get_session.return_value = mock_df
            
            loader = EnhancedDataLoader()
            
            # Test loading with bpod=True
            result = loader.load(load_bpod=True)
            
            # Verify the call was made with correct parameters
            mock_get_session.assert_called_with(if_load_bpod=True)
            assert loader.load_parameters == {'load_bpod': True}
            assert isinstance(result, pd.DataFrame)

    def test_reload_data_functionality(self):
        """Test the reload_data method"""
        with patch('app_utils.app_data_load.data_loader.get_session_table') as mock_get_session:
            mock_df = pd.DataFrame({
                'subject_id': ['test_subject'],
                'session': [1]
            })
            mock_get_session.return_value = mock_df
            
            loader = EnhancedDataLoader()
            original_load_time = loader.last_load_time
            
            # Wait a moment and reload
            import time
            time.sleep(0.01)
            
            result = loader.reload_data(load_bpod=True)
            
            # Verify reload happened
            assert loader.last_load_time > original_load_time
            assert loader.load_parameters == {'load_bpod': True}
            assert isinstance(result, pd.DataFrame)

    def test_get_subject_sessions(self):
        """Test getting sessions for a specific subject"""
        with patch('app_utils.app_data_load.data_loader.get_session_table') as mock_get_session:
            mock_df = pd.DataFrame({
                'subject_id': ['subject_1', 'subject_1', 'subject_2'],
                'session': [1, 2, 1],
                'session_date': [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 1)]
            })
            mock_get_session.return_value = mock_df
            
            loader = EnhancedDataLoader()
            
            # Test getting sessions for subject_1
            result = loader.get_subject_sessions('subject_1')
            
            assert result is not None
            assert len(result) == 2
            assert all(result['subject_id'] == 'subject_1')
            # Should be sorted by session_date descending
            assert result.iloc[0]['session_date'] > result.iloc[1]['session_date']

    def test_get_subject_sessions_not_found(self):
        """Test getting sessions for non-existent subject"""
        with patch('app_utils.app_data_load.data_loader.get_session_table') as mock_get_session:
            mock_df = pd.DataFrame({
                'subject_id': ['subject_1'],
                'session': [1]
            })
            mock_get_session.return_value = mock_df
            
            loader = EnhancedDataLoader()
            
            # Test getting sessions for non-existent subject
            result = loader.get_subject_sessions('non_existent_subject')
            
            assert result is None

    def test_get_most_recent_subject_sessions(self):
        """Test getting most recent session for each subject"""
        with patch('app_utils.app_data_load.data_loader.get_session_table') as mock_get_session:
            mock_df = pd.DataFrame({
                'subject_id': ['subject_1', 'subject_1', 'subject_2'],
                'session': [1, 2, 1],
                'session_date': [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)]
            })
            mock_get_session.return_value = mock_df
            
            loader = EnhancedDataLoader()
            
            result = loader.get_most_recent_subject_sessions()
            
            assert len(result) == 2  # Two unique subjects
            # subject_1 should have session 2 (most recent)
            subject_1_row = result[result['subject_id'] == 'subject_1'].iloc[0]
            assert subject_1_row['session'] == 2
            
            # subject_2 should have session 1
            subject_2_row = result[result['subject_id'] == 'subject_2'].iloc[0]
            assert subject_2_row['session'] == 1

    def test_get_subjects_list(self):
        """Test getting list of unique subjects"""
        with patch('app_utils.app_data_load.data_loader.get_session_table') as mock_get_session:
            mock_df = pd.DataFrame({
                'subject_id': ['subject_b', 'subject_a', 'subject_b'],
                'session': [1, 1, 2]
            })
            mock_get_session.return_value = mock_df
            
            loader = EnhancedDataLoader()
            
            result = loader.get_subjects_list()
            
            assert len(result) == 2
            assert 'subject_a' in result
            assert 'subject_b' in result
            # Should be sorted alphabetically
            assert result == ['subject_a', 'subject_b']

    def test_get_sessions_count(self):
        """Test getting session counts by subject"""
        with patch('app_utils.app_data_load.data_loader.get_session_table') as mock_get_session:
            mock_df = pd.DataFrame({
                'subject_id': ['subject_1', 'subject_1', 'subject_2'],
                'session': [1, 2, 1]
            })
            mock_get_session.return_value = mock_df
            
            loader = EnhancedDataLoader()
            
            result = loader.get_sessions_count()
            
            assert result['subject_1'] == 2
            assert result['subject_2'] == 1

    def test_get_data_summary(self):
        """Test getting data summary"""
        with patch('app_utils.app_data_load.data_loader.get_session_table') as mock_get_session:
            mock_df = pd.DataFrame({
                'subject_id': ['subject_1', 'subject_2'],
                'session': [1, 1],
                'session_date': [datetime(2024, 1, 1), datetime(2024, 1, 2)]
            })
            mock_get_session.return_value = mock_df
            
            loader = EnhancedDataLoader()
            
            result = loader.get_data_summary()
            
            assert result['total_sessions'] == 2
            assert result['total_subjects'] == 2
            assert result['date_range']['earliest'] == datetime(2024, 1, 1)
            assert result['date_range']['latest'] == datetime(2024, 1, 2)
            assert result['last_load_time'] is not None
            assert result['load_parameters'] == {'load_bpod': False}
            assert 'subject_1' in result['subjects_list']
            assert 'subject_2' in result['subjects_list']

    def test_validate_data_success(self):
        """Test data validation with valid data"""
        with patch('app_utils.app_data_load.data_loader.get_session_table') as mock_get_session:
            mock_df = pd.DataFrame({
                'subject_id': ['subject_1', 'subject_2'],
                'session': [1, 1],
                'session_date': [datetime(2024, 1, 1), datetime(2024, 1, 2)]
            })
            mock_get_session.return_value = mock_df
            
            loader = EnhancedDataLoader()
            
            result = loader.validate_data()
            
            assert result['valid'] is True
            assert len(result['issues']) == 0
            assert result['total_sessions'] == 2
            assert result['total_subjects'] == 2

    def test_validate_data_with_issues(self):
        """Test data validation with data issues"""
        with patch('app_utils.app_data_load.data_loader.get_session_table') as mock_get_session:
            # Create data with issues
            mock_df = pd.DataFrame({
                'subject_id': ['subject_1', None],  # Null value in required column
                'session': [1, 1],
                'session_date': [datetime(2024, 1, 1), datetime(2024, 1, 2)]
            })
            mock_get_session.return_value = mock_df
            
            loader = EnhancedDataLoader()
            
            result = loader.validate_data()
            
            assert result['valid'] is False
            assert any('null values in subject_id' in issue for issue in result['issues'])

    def test_is_data_loaded(self):
        """Test checking if data is loaded"""
        with patch('app_utils.app_data_load.data_loader.get_session_table') as mock_get_session:
            mock_df = pd.DataFrame({'subject_id': ['test'], 'session': [1]})
            mock_get_session.return_value = mock_df
            
            loader = EnhancedDataLoader()
            
            assert loader.is_data_loaded() is True
            
            # Clear data and test
            loader.clear_data()
            assert loader.is_data_loaded() is False

    def test_clear_data(self):
        """Test clearing loaded data"""
        with patch('app_utils.app_data_load.data_loader.get_session_table') as mock_get_session:
            mock_df = pd.DataFrame({'subject_id': ['test'], 'session': [1]})
            mock_get_session.return_value = mock_df
            
            loader = EnhancedDataLoader()
            
            # Verify data is loaded
            assert loader.session_table is not None
            assert loader.last_load_time is not None
            
            # Clear data
            loader.clear_data()
            
            # Verify data is cleared
            assert loader.session_table is None
            assert loader.last_load_time is None
            assert loader.load_parameters is None

    def test_error_handling_in_load(self):
        """Test error handling during data loading"""
        with patch('app_utils.app_data_load.data_loader.get_session_table') as mock_get_session:
            # Make the session table loading fail
            mock_get_session.side_effect = Exception("Database connection failed")
            
            with pytest.raises(ValueError, match="Failed to load session table"):
                EnhancedDataLoader()

    def test_error_handling_in_subject_sessions(self):
        """Test error handling when getting subject sessions"""
        with patch('app_utils.app_data_load.data_loader.get_session_table') as mock_get_session:
            mock_df = pd.DataFrame({'subject_id': ['test']})
            mock_get_session.return_value = mock_df
            
            loader = EnhancedDataLoader()
            
            # Mock an error in the subject filtering
            with patch.object(loader, 'get_data', side_effect=Exception("Data access error")):
                result = loader.get_subject_sessions('test_subject')
                
                assert result is None  # Should return None on error


class TestEnhancedDataLoaderIntegration:
    """Integration tests for EnhancedDataLoader with AppUtils"""

    def test_app_utils_uses_enhanced_data_loader(self):
        """Test that AppUtils correctly uses the EnhancedDataLoader"""
        with patch('app_utils.app_data_load.data_loader.get_session_table') as mock_get_session:
            mock_df = pd.DataFrame({
                'subject_id': ['test_subject'],
                'session': [1],
                'session_date': [datetime(2024, 1, 1)]
            })
            mock_get_session.return_value = mock_df
            
            # Import and create AppUtils
            from app_utils.app_utils import AppUtils
            
            app_utils = AppUtils()
            
            # Verify that it uses EnhancedDataLoader
            assert isinstance(app_utils.data_loader, EnhancedDataLoader)
            
            # Test delegation methods
            session_data = app_utils.get_session_data(use_cache=False)
            assert isinstance(session_data, pd.DataFrame)
            
            subject_sessions = app_utils.get_subject_sessions('test_subject')
            assert isinstance(subject_sessions, pd.DataFrame)
            assert len(subject_sessions) == 1
            
            reload_data = app_utils.reload_data()
            assert isinstance(reload_data, pd.DataFrame)

    def test_enhanced_data_loader_caching_integration(self):
        """Test that enhanced data loader works correctly with caching"""
        with patch('app_utils.app_data_load.data_loader.get_session_table') as mock_get_session:
            mock_df = pd.DataFrame({
                'subject_id': ['test_subject'],
                'session': [1],
                'session_date': [datetime(2024, 1, 1)]
            })
            mock_get_session.return_value = mock_df
            
            from app_utils.app_utils import AppUtils
            
            app_utils = AppUtils()
            
            # First call should load data
            data1 = app_utils.get_session_data(use_cache=True)
            assert mock_get_session.call_count >= 1
            
            # Second call should use cache
            call_count_before = mock_get_session.call_count
            data2 = app_utils.get_session_data(use_cache=True)
            assert mock_get_session.call_count == call_count_before  # No additional calls
            
            # Data should be the same
            pd.testing.assert_frame_equal(data1, data2) 