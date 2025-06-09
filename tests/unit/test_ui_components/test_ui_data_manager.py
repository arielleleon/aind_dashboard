import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Import the UIDataManager class
from app_utils.ui_utils import UIDataManager
# Import the new realistic fixtures
from tests.fixtures import get_realistic_session_data, get_strata_test_cases


class TestUIDataManager:
    """Test suite for UIDataManager class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.ui_manager = UIDataManager()
        
        # Use realistic sample session data for testing
        self.sample_session_data = get_realistic_session_data()
        
    def test_initialization(self):
        """Test UIDataManager initialization"""
        assert self.ui_manager is not None
        assert self.ui_manager.features == ['finished_trials', 'ignore_rate', 'total_trials', 'foraging_performance', 'abs(bias_naive)']
    
    def test_map_percentile_to_category(self):
        """Test percentile to category mapping"""
        # Test all categories
        assert self.ui_manager.map_percentile_to_category(5.0) == 'SB'  # Severely Below
        assert self.ui_manager.map_percentile_to_category(20.0) == 'B'   # Below
        assert self.ui_manager.map_percentile_to_category(50.0) == 'N'   # Normal
        assert self.ui_manager.map_percentile_to_category(80.0) == 'G'   # Good
        assert self.ui_manager.map_percentile_to_category(95.0) == 'SG'  # Severely Good
        assert self.ui_manager.map_percentile_to_category(np.nan) == 'NS'  # Not Significant
        
        # Test boundary conditions
        assert self.ui_manager.map_percentile_to_category(6.5) == 'B'
        assert self.ui_manager.map_percentile_to_category(28.0) == 'N'
        assert self.ui_manager.map_percentile_to_category(72.0) == 'N'
        assert self.ui_manager.map_percentile_to_category(93.5) == 'G'
    
    def test_get_strata_abbreviation(self):
        """Test strata abbreviation generation - Updated with real app data"""
        # Test real strata formats from app
        test_cases = get_strata_test_cases()
        
        for case in test_cases:
            result = self.ui_manager.get_strata_abbreviation(case['input'])
            assert result == case['expected'], f"Failed for {case['description']}: {case['input']} should be {case['expected']}, got {result}"
    
    def test_optimize_session_data_storage(self):
        """Test session data storage optimization"""
        # Mock bootstrap manager for testing
        mock_bootstrap_manager = Mock()
        mock_bootstrap_manager.is_bootstrap_available.return_value = False
        
        # Mock cache manager for testing
        mock_cache_manager = Mock()
        mock_cache_manager.calculate_data_hash.return_value = 'test_hash_123'
        
        result = self.ui_manager.optimize_session_data_storage(
            self.sample_session_data,
            bootstrap_manager=mock_bootstrap_manager,
            cache_manager=mock_cache_manager
        )
        
        # Verify structure
        assert 'subjects' in result
        assert 'strata_reference' in result
        assert 'metadata' in result
        assert 'bootstrap_coverage' in result
        
        # Verify metadata with real data counts
        metadata = result['metadata']
        assert metadata['total_subjects'] == 5  # 5 unique subjects in real data
        assert metadata['total_sessions'] == 10  # 10 sessions in real data
        assert metadata['total_strata'] == 7   # 7 unique strata in real data  
        assert metadata['data_hash'] == 'test_hash_123'
        assert metadata['phase3_enhanced'] is True
        
        # Verify subjects data structure
        subjects = result['subjects']
        assert '690494' in subjects  # Real subject ID
        assert '690486' in subjects  # Real subject ID
        assert '702200' in subjects  # Real subject ID
        
        # Verify subject data structure with real data
        subject_690494_data = subjects['690494']
        assert 'sessions' in subject_690494_data
        assert 'current_strata' in subject_690494_data
        assert 'total_sessions' in subject_690494_data
        assert subject_690494_data['total_sessions'] == 2  # 690494 has 2 sessions in real data
        assert subject_690494_data['current_strata'] == 'Uncoupled Without Baiting_BEGINNER_v1'  # Real strata
    
    def test_create_ui_optimized_structures(self):
        """Test UI optimized structures creation"""
        # Mock bootstrap manager
        mock_bootstrap_manager = Mock()
        
        result = self.ui_manager.create_ui_optimized_structures(
            self.sample_session_data,
            bootstrap_manager=mock_bootstrap_manager
        )
        
        # Verify structure
        assert 'feature_rank_data' in result
        assert 'subject_lookup' in result
        assert 'strata_lookup' in result
        assert 'time_series_data' in result
        assert 'table_display_cache' in result
        
        # Verify feature rank data with real subjects
        feature_rank_data = result['feature_rank_data']
        assert len(feature_rank_data) == 5  # 5 unique subjects in real data
        assert '690494' in feature_rank_data  # Real subject ID
        
        subject_690494_rank = feature_rank_data['690494']
        assert 'features' in subject_690494_rank
        assert 'overall_percentile' in subject_690494_rank
        assert 'overall_category' in subject_690494_rank
        # This subject has most recent session with NaN percentile, so should be None
        assert pd.isna(subject_690494_rank['overall_percentile']) or subject_690494_rank['overall_percentile'] is None
        
        # Verify subject lookup with real data
        subject_lookup = result['subject_lookup']
        assert '690494' in subject_lookup
        subject_690494_lookup = subject_lookup['690494']
        assert 'latest' in subject_690494_lookup
        assert 'summary' in subject_690494_lookup
        assert subject_690494_lookup['latest']['strata'] == 'Uncoupled Without Baiting_BEGINNER_v1'  # Real strata
        
        # Verify strata lookup with real strata
        strata_lookup = result['strata_lookup']
        assert 'Uncoupled Without Baiting_BEGINNER_v1' in strata_lookup  # Real strata name
        strata_info = strata_lookup['Uncoupled Without Baiting_BEGINNER_v1']
        assert 'subject_count' in strata_info
        assert 'session_count' in strata_info
        assert strata_info['subject_count'] == 1  # Only 690494 in this strata
        assert strata_info['session_count'] == 2  # Two sessions for 690494 in this strata
        
        # Verify time series data with real subjects
        time_series_data = result['time_series_data']
        assert '690494' in time_series_data
        subject_690494_ts = time_series_data['690494']
        assert 'sessions' in subject_690494_ts
        assert 'dates' in subject_690494_ts
        assert 'overall_percentiles' in subject_690494_ts
        assert len(subject_690494_ts['sessions']) == 2  # 690494 has 2 sessions
        
        # Verify table display cache with real data
        table_display_cache = result['table_display_cache']
        assert len(table_display_cache) == 5  # 5 unique subjects
        
        # Check that real subject IDs are in table display
        subject_ids_in_table = [row['subject_id'] for row in table_display_cache]
        assert '690494' in subject_ids_in_table
        assert '690486' in subject_ids_in_table
        assert '702200' in subject_ids_in_table
        
        # Verify strata abbreviation works in table context with real data
        for row in table_display_cache:
            if row['subject_id'] == '690494':
                assert row['strata'] == 'Uncoupled Without Baiting_BEGINNER_v1'
                assert row['strata_abbr'] == 'UWBB1'  # Real abbreviation
                break
    
    def test_get_subject_display_data(self):
        """Test get subject display data"""
        # Create sample UI structures using real data
        ui_structures = self.ui_manager.create_ui_optimized_structures(
            self.sample_session_data,
            bootstrap_manager=Mock()
        )
        
        # Test with real subject ID
        result = self.ui_manager.get_subject_display_data('690494', ui_structures)
        
        assert 'latest' in result
        assert 'summary' in result
        assert result['latest']['strata'] == 'Uncoupled Without Baiting_BEGINNER_v1'  # Real strata
        assert result['summary']['total_sessions'] == 2  # Real session count
    
    def test_get_table_display_data(self):
        """Test get table display data"""
        # Create sample UI structures using real data
        ui_structures = self.ui_manager.create_ui_optimized_structures(
            self.sample_session_data,
            bootstrap_manager=Mock()
        )
        
        result = self.ui_manager.get_table_display_data(ui_structures)
        
        assert isinstance(result, list)
        assert len(result) == 5  # 5 unique subjects in real data
        
        # Verify real subject data is present
        subject_ids = [row['subject_id'] for row in result]
        assert '690494' in subject_ids
        assert '690486' in subject_ids
        
        # Check that real strata are present and correctly abbreviated
        for row in result:
            if row['subject_id'] == '690494':
                assert row['strata'] == 'Uncoupled Without Baiting_BEGINNER_v1'
                assert row['strata_abbr'] == 'UWBB1'
                break
    
    def test_get_time_series_data(self):
        """Test get time series data"""
        # Create sample UI structures using real data
        ui_structures = self.ui_manager.create_ui_optimized_structures(
            self.sample_session_data,
            bootstrap_manager=Mock()
        )
        
        # Test with real subject ID
        result = self.ui_manager.get_time_series_data('690494', ui_structures)
        
        assert 'sessions' in result
        assert 'dates' in result
        assert 'overall_percentiles' in result
        assert len(result['sessions']) == 2  # 690494 has 2 sessions
        
        # Verify real session numbers
        assert 9.0 in result['sessions']  # Real session numbers from data
        assert 10.0 in result['sessions']
    
    def test_create_table_display_cache_with_threshold_analyzer(self):
        """Test table display cache creation with threshold analyzer integration"""
        result = self.ui_manager._create_table_display_cache(self.sample_session_data)
        
        assert isinstance(result, list)
        assert len(result) == 5  # 5 unique subjects
        
        # Verify threshold alert columns are present
        for row in result:
            assert 'threshold_alert' in row
            assert 'total_sessions_alert' in row
            assert 'stage_sessions_alert' in row
            assert 'water_day_total_alert' in row
            
            # Verify real subject data structure
            assert 'subject_id' in row
            assert 'strata' in row
            assert 'strata_abbr' in row
            
            # Check real strata abbreviations
            if row['subject_id'] == '690494':
                assert row['strata'] == 'Uncoupled Without Baiting_BEGINNER_v1'
                assert row['strata_abbr'] == 'UWBB1'
    
    def test_create_time_series_data(self):
        """Test time series data creation"""
        result = self.ui_manager._create_time_series_data(self.sample_session_data)
        
        assert isinstance(result, dict)
        assert len(result) == 5  # 5 unique subjects
        
        # Test with real subject ID
        assert '690494' in result
        subject_690494_ts = result['690494']
        
        # Verify time series structure
        assert 'sessions' in subject_690494_ts
        assert 'dates' in subject_690494_ts
        assert 'overall_percentiles' in subject_690494_ts
        assert 'strata' in subject_690494_ts
        
        # Verify real data values
        assert len(subject_690494_ts['sessions']) == 2
        assert 9.0 in subject_690494_ts['sessions']
        assert 10.0 in subject_690494_ts['sessions']
        
        # Verify real strata names in time series
        assert 'Uncoupled Without Baiting_BEGINNER_v1' in subject_690494_ts['strata']
    
    def test_bootstrap_manager_integration(self):
        """Test bootstrap manager integration with real data"""
        # Mock bootstrap manager with bootstrap availability
        mock_bootstrap_manager = Mock()
        mock_bootstrap_manager.is_bootstrap_available.return_value = True
        mock_bootstrap_manager.statistical_utils = Mock()
        
        # Mock cache manager
        mock_cache_manager = Mock()
        mock_cache_manager.calculate_data_hash.return_value = 'test_hash_bootstrap'
        
        result = self.ui_manager.optimize_session_data_storage(
            self.sample_session_data,
            bootstrap_manager=mock_bootstrap_manager,
            cache_manager=mock_cache_manager
        )
        
        # Verify bootstrap coverage is tracked
        assert 'bootstrap_coverage' in result
        bootstrap_coverage = result['bootstrap_coverage']
        
        # Should have coverage for all real strata
        real_strata_names = self.sample_session_data['strata'].unique()
        for strata in real_strata_names:
            assert strata in bootstrap_coverage
            strata_coverage = bootstrap_coverage[strata]
            assert 'bootstrap_enabled' in strata_coverage
            assert 'feature_coverage' in strata_coverage
    
    def test_error_handling(self):
        """Test error handling with edge cases"""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        result = self.ui_manager.optimize_session_data_storage(empty_df)
        
        assert result['metadata']['total_subjects'] == 0
        assert result['metadata']['total_sessions'] == 0
        assert result['subjects'] == {}
        
        # Test with invalid strata abbreviation
        result = self.ui_manager.get_strata_abbreviation(None)
        assert result == ''
        
        # Test with missing columns (but include required ones to avoid KeyError)
        incomplete_df = pd.DataFrame({
            'subject_id': ['test'],
            'session_date': [datetime.now()],  # Add required column to avoid KeyError
            'session': [1],
            'strata': ['Test_Strata_v1']
        })
        result = self.ui_manager.optimize_session_data_storage(incomplete_df)
        assert result['metadata']['total_subjects'] == 1  # Should handle gracefully
        assert result['metadata']['total_sessions'] == 1
        
        # Test with None input for strata abbreviation
        result = self.ui_manager.get_strata_abbreviation(None)
        assert result == ''


class TestUIDataManagerIntegration:
    """Integration tests for UIDataManager with other components"""
    
    @patch('app_utils.app_utils.UIDataManager')
    def test_app_utils_uses_ui_data_manager(self, mock_ui_data_manager):
        """Test that AppUtils properly uses UIDataManager"""
        from app_utils.app_utils import AppUtils
        
        # Mock the UIDataManager methods
        mock_instance = Mock()
        mock_ui_data_manager.return_value = mock_instance
        
        app_utils = AppUtils()
        
        # Verify UIDataManager was instantiated
        mock_ui_data_manager.assert_called_once()
        assert app_utils.ui_data_manager == mock_instance
    
    def test_ui_data_manager_delegation(self):
        """Test UI data manager delegation in AppUtils"""
        from app_utils.app_utils import AppUtils
        
        # Create real AppUtils instance
        app_utils = AppUtils()
        
        # Test strata abbreviation delegation with real data
        real_strata = 'Uncoupled Without Baiting_BEGINNER_v1'
        result = app_utils._get_strata_abbreviation(real_strata)
        assert result == 'UWBB1'  # Real expected abbreviation
        
        # Test with different real strata
        another_strata = 'Coupled Baiting_ADVANCED_v2'
        result = app_utils._get_strata_abbreviation(another_strata)
        assert result == 'CBA2'  # Real expected abbreviation


@pytest.fixture
def sample_ui_structures():
    """Fixture providing sample UI structures for testing"""
    return {
        'feature_rank_data': {
            '690494': {  # Real subject ID
                'features': {
                    'finished_trials': {'percentile': 75.0, 'category': 'G'},
                    'ignore_rate': {'percentile': 25.0, 'category': 'B'}
                },
                'overall_percentile': 50.0,
                'overall_category': 'N',
                'strata': 'Uncoupled Without Baiting_BEGINNER_v1'  # Real strata
            }
        },
        'subject_lookup': {
            '690494': {  # Real subject ID
                'latest': {
                    'strata': 'Uncoupled Without Baiting_BEGINNER_v1',  # Real strata
                    'overall_percentile': 50.0
                },
                'summary': {
                    'total_sessions': 2
                }
            }
        },
        'time_series_data': {
            '690494': {  # Real subject ID
                'sessions': [9.0, 10.0],  # Real session numbers
                'dates': ['2023-01-01', '2023-01-02'],
                'overall_percentiles': [45.0, 55.0]
            }
        },
        'table_display_cache': [
            {
                'subject_id': '690494',  # Real subject ID
                'strata': 'Uncoupled Without Baiting_BEGINNER_v1',  # Real strata
                'strata_abbr': 'UWBB1',  # Real abbreviation
                'overall_percentile': 50.0
            }
        ]
    }


@pytest.fixture
def mock_bootstrap_manager():
    """Fixture providing a mock bootstrap manager"""
    mock = Mock()
    mock.is_bootstrap_available.return_value = False
    mock.statistical_utils = Mock()
    return mock


@pytest.fixture
def mock_cache_manager():
    """Fixture providing a mock cache manager"""
    mock = Mock()
    mock.calculate_data_hash.return_value = 'test_hash_12345'
    return mock 