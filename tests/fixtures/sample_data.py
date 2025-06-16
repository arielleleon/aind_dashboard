"""
Shared Sample Data for AIND Dashboard Tests

This module provides realistic sample data extracted from the actual app
for use across all test modules. The data is based on real strata formats
and column structures to ensure tests accurately reflect production behavior.

Generated from real app data on 2025-06-06
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List


class SampleDataProvider:
    """
    Centralized provider for all test sample data
    
    This class provides various types of sample data needed across different
    test modules, all based on real app data structures.
    """
    
    def __init__(self):
        """Initialize with real app data structures"""
        self.real_strata = [
            'Uncoupled Without Baiting_BEGINNER_v1',
            'Uncoupled Without Baiting_INTERMEDIATE_v1', 
            'Uncoupled Without Baiting_ADVANCED_v1',
            'Coupled Baiting_INTERMEDIATE_v1',
            'Coupled Baiting_ADVANCED_v1',
            'Coupled Baiting_INTERMEDIATE_v2',
            'Coupled Baiting_ADVANCED_v2'
        ]
        
        self.real_strata_abbreviations = {
            'Uncoupled Without Baiting_BEGINNER_v1': 'UWBB1',
            'Uncoupled Without Baiting_INTERMEDIATE_v1': 'UWBI1',
            'Uncoupled Without Baiting_ADVANCED_v1': 'UWBA1',
            'Coupled Baiting_INTERMEDIATE_v1': 'CBI1',
            'Coupled Baiting_ADVANCED_v1': 'CBA1',
            'Coupled Baiting_INTERMEDIATE_v2': 'CBI2',
            'Coupled Baiting_ADVANCED_v2': 'CBA2'
        }
        
        # Standard feature columns from real app
        self.features = [
            'finished_trials', 'ignore_rate', 'total_trials', 
            'foraging_performance', 'abs(bias_naive)'
        ]
    
    def create_realistic_session_data(self) -> pd.DataFrame:
        """
        Create realistic session data based on real app data structure
        
        Returns:
            pd.DataFrame: Session data with all columns needed for testing
        """
        data = {
            'subject_id': ['690494', '690486', '690486', '690494', '702200', '702200', '697929', '697929', '700708', '700708'],
            'session_date': [
                datetime.now() - timedelta(days=457), 
                datetime.now() - timedelta(days=457), 
                datetime.now() - timedelta(days=456), 
                datetime.now() - timedelta(days=456), 
                datetime.now() - timedelta(days=449), 
                datetime.now() - timedelta(days=448), 
                datetime.now() - timedelta(days=442), 
                datetime.now() - timedelta(days=441), 
                datetime.now() - timedelta(days=330), 
                datetime.now() - timedelta(days=329)
            ],
            'session': [9.0, 8.0, 9.0, 10.0, 11.0, 12.0, 14.0, 15.0, 61.0, 62.0],
            'session_index': [2, 1, 1, 1, 1, 1, 2, 1, 1, 1],
            'strata': [
                'Uncoupled Without Baiting_BEGINNER_v1', 
                'Uncoupled Without Baiting_INTERMEDIATE_v1', 
                'Uncoupled Without Baiting_ADVANCED_v1', 
                'Uncoupled Without Baiting_BEGINNER_v1', 
                'Coupled Baiting_INTERMEDIATE_v1', 
                'Coupled Baiting_ADVANCED_v1', 
                'Coupled Baiting_ADVANCED_v1', 
                'Coupled Baiting_ADVANCED_v1', 
                'Coupled Baiting_INTERMEDIATE_v2', 
                'Coupled Baiting_ADVANCED_v2'
            ],
            'session_overall_percentile': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 33.6, 20.9],
            'overall_percentile_category': ['NS', 'NS', 'NS', 'NS', 'NS', 'NS', 'NS', 'NS', 'N', 'B'],
            'session_overall_rolling_avg': [0.065, 0.424, 0.113, 0.216, -0.205, -0.262, 0.726, 0.763, -0.347, -1.354],
            # Core features
            'finished_trials': [382, 388, 131, 346, 392, 317, 555, 548, 467, 206],
            'finished_trials_session_percentile': [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 40.9, 13.6],
            'finished_trials_processed_rolling_avg': [0.298, 0.567, -0.912, 0.425, -0.255, -0.522, 1.180, 1.185, -0.443, -1.671],
            'ignore_rate': [0.147, 0.008, 0.022, 0.170, 0.190, 0.208, 0.009, 0.023, 0.237, 0.520],
            'ignore_rate_session_percentile': [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 31.8, 13.6],
            'ignore_rate_processed_rolling_avg': [0.441, 0.727, 1.208, 0.446, -1.466, -0.855, 0.449, 0.454, -0.490, -2.115],
            'total_trials': [448, 391, 134, 417, 484, 400, 560, 561, 612, 429],
            'total_trials_session_percentile': [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 40.9, 22.7],
            'total_trials_processed_rolling_avg': [0.081, 0.377, -1.253, 0.236, 0.666, -0.115, 1.141, 1.143, -0.124, -1.207],
            'foraging_performance': [0.619, 0.762, 0.836, 0.641, 0.663, 0.671, 0.804, 0.833, 0.536, 0.443],
            'foraging_performance_session_percentile': [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 22.7, 13.6],
            'foraging_performance_processed_rolling_avg': [-0.148, 0.353, 1.372, 0.013, -0.622, -0.332, 0.532, 0.594, -0.427, -1.729],
            'abs(bias_naive)': [0.136, 0.119, 0.588, 0.133, 0.173, 0.211, 0.247, 0.080, 0.062, 0.816],
            'abs(bias_naive)_session_percentile': [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 31.8, 40.9],
            'abs(bias_naive)_processed_rolling_avg': [-0.346, 0.095, 0.150, -0.040, 0.654, 0.515, 0.327, 0.437, -0.252, -0.047],
            # Metadata columns
            'PI': ['Jackie Swapp', 'Jackie Swapp', 'Jackie Swapp', 'Jackie Swapp', 'Kenta Hagihara', 'Kenta Hagihara', 'Kenta Hagihara', 'Kenta Hagihara', 'Kenta Hagihara', 'Kenta Hagihara'],
            'trainer': ['Henry Loeffler', 'Henry Loeffler', 'Henry Loeffler', 'Henry Loeffler', 'Ella Hilton', 'Ella Hilton', 'Ella Hilton', 'Ella Hilton', 'Huy Nguyen', 'Huy Nguyen'],
            'rig': ['447-1-B', '447-1-C', '447-1-C', '447-1-B', '447-2-D', '447-2-D', '447-3-D', '447-3-D', '447-1-D', '447-1-D'],
            'water_day_total': [2.288, 1.084, 1.087, 2.053, 1.051, 1.0, 1.541, 2.756, 1.813, 1.47],
            'current_stage_actual': ['STAGE_1', 'STAGE_3', 'STAGE_FINAL', 'STAGE_2', 'STAGE_3', 'STAGE_FINAL', 'GRADUATED', 'GRADUATED', 'STAGE_3', 'STAGE_FINAL'],
            'is_outlier': [False, False, False, False, False, False, False, False, False, True],
            'outlier_weight': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
            'is_current_strata': [True, False, True, True, False, True, True, True, False, True],
            'is_last_session': [False, False, False, False, False, True, False, False, False, False],
            'curriculum_name': ['Uncoupled Without Baiting', 'Uncoupled Without Baiting', 'Uncoupled Without Baiting', 'Uncoupled Without Baiting', 'Coupled Baiting', 'Coupled Baiting', 'Coupled Baiting', 'Coupled Baiting', 'Coupled Baiting', 'Coupled Baiting'],
            'finished_rate': [0.853, 0.992, 0.978, 0.830, 0.810, 0.793, 0.991, 0.977, 0.763, 0.480],
            'base_weight': [36.57, 27.77, 27.77, 36.57, 26.5, 26.5, 28.8, 28.8, 25.07, 25.07],
            'target_weight': [31.084, 23.604, 23.604, 31.084, 22.525, 22.525, 24.48, 24.48, 21.31, 21.31],
            'weight_after': [29.6, 23.0, 22.7, 29.7, 21.9, 22.2, 23.5, 22.3, 19.8, 20.0],
        }

        return pd.DataFrame(data)
    
    def create_simple_session_data(self) -> pd.DataFrame:
        """
        Create simpler session data for basic tests
        
        Returns:
            pd.DataFrame: Minimal session data for lightweight tests
        """
        return pd.DataFrame({
            'subject_id': ['690494', '690486', '702200'],
            'session_date': [
                datetime.now() - timedelta(days=2),
                datetime.now() - timedelta(days=1), 
                datetime.now()
            ],
            'session': [1, 1, 1],
            'strata': [
                'Uncoupled Without Baiting_BEGINNER_v1',
                'Uncoupled Without Baiting_INTERMEDIATE_v1',
                'Coupled Baiting_ADVANCED_v1'
            ],
            'session_overall_percentile': [75.0, 25.0, 95.0],
            'overall_percentile_category': ['G', 'B', 'SG']
        })
    
    def create_table_display_data(self) -> List[Dict[str, Any]]:
        """
        Create realistic table display data for UI tests
        
        Returns:
            List[Dict[str, Any]]: Table display data matching real app structure
        """
        return [
            {
                'subject_id': '690494',
                'session_date': datetime.now() - timedelta(days=1),
                'session': 10,
                'strata': 'Uncoupled Without Baiting_BEGINNER_v1',
                'strata_abbr': 'UWBB1',
                'overall_percentile': 75.0,
                'overall_category': 'G',
                'PI': 'Jackie Swapp',
                'trainer': 'Henry Loeffler',
                'rig': '447-1-B'
            },
            {
                'subject_id': '690486', 
                'session_date': datetime.now() - timedelta(days=1),
                'session': 9,
                'strata': 'Uncoupled Without Baiting_ADVANCED_v1',
                'strata_abbr': 'UWBA1',
                'overall_percentile': 25.0,
                'overall_category': 'B',
                'PI': 'Jackie Swapp',
                'trainer': 'Henry Loeffler', 
                'rig': '447-1-C'
            }
        ]
    
    def create_statistical_data(self) -> Dict[str, Any]:
        """
        Create realistic statistical data for analysis tests
        
        Returns:
            Dict[str, Any]: Statistical data with real parameters
        """
        np.random.seed(42)  # For reproducible tests
        return {
            'values': np.random.normal(100, 15, 1000),
            'percentiles': [25, 50, 75, 90, 95],
            'confidence_level': 0.95,
            'features': self.features,
            'strata_reference': {
                strata: np.random.normal(50, 20, 100) 
                for strata in self.real_strata[:3]  # Use first 3 strata
            }
        }
    
    def get_strata_test_cases(self) -> List[Dict[str, str]]:
        """
        Get test cases for strata abbreviation based on real app data
        
        Returns:
            List[Dict[str, str]]: Test cases with strata and expected abbreviations
        """
        return [
            {
                'input': 'Uncoupled Without Baiting_BEGINNER_v1',
                'expected': 'UWBB1',
                'description': 'Real app strata format'
            },
            {
                'input': 'Uncoupled Without Baiting_INTERMEDIATE_v1', 
                'expected': 'UWBI1',
                'description': 'Real app strata format'
            },
            {
                'input': 'Coupled Baiting_ADVANCED_v2',
                'expected': 'CBA2',
                'description': 'Real app strata format'
            },
            {
                'input': '',
                'expected': '',
                'description': 'Empty string edge case'
            },
            {
                'input': 'Simple_Format',
                'expected': 'Simple_Format',
                'description': 'Simple format fallback'
            }
        ]
    
    def create_mock_cache_manager(self) -> Any:
        """Create a mock cache manager for testing"""
        from unittest.mock import Mock
        mock_cache = Mock()
        mock_cache.calculate_data_hash.return_value = 'test_hash_abc123'
        mock_cache.has.return_value = False
        mock_cache.get.return_value = None
        mock_cache.set.return_value = None
        return mock_cache


# Global instance for easy access
sample_data_provider = SampleDataProvider()


def get_realistic_session_data() -> pd.DataFrame:
    """Convenience function to get realistic session data"""
    return sample_data_provider.create_realistic_session_data()


def get_simple_session_data() -> pd.DataFrame:
    """Convenience function to get simple session data"""
    return sample_data_provider.create_simple_session_data()


def get_strata_test_cases() -> List[Dict[str, str]]:
    """Convenience function to get strata test cases"""
    return sample_data_provider.get_strata_test_cases()


def get_table_display_data() -> List[Dict[str, Any]]:
    """Convenience function to get table display data"""
    return sample_data_provider.create_table_display_data()


def get_statistical_data() -> Dict[str, Any]:
    """Convenience function to get statistical data"""
    return sample_data_provider.create_statistical_data() 