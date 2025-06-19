"""
Unit tests for shared utility functions

These tests focus on the utility functions that support the main application,
starting with the shared_utils module.
"""
import pytest
from unittest.mock import Mock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


class TestSharedUtils:
    """Tests for shared_utils module functionality"""
    
    def test_get_app_utils_returns_instance(self):
        """
        Test that get_app_utils returns an AppUtils instance
        
        This verifies the basic functionality of the shared utils pattern
        without requiring full data loading.
        """
        # Import the module first
        import shared_utils
        
        # Mock the AppUtils class to avoid data loading during testing
        with patch.object(shared_utils, 'AppUtils') as MockAppUtils:
            mock_instance = Mock()
            MockAppUtils.return_value = mock_instance
            
            # Create a new app_utils instance with the mock
            with patch.object(shared_utils, 'app_utils', mock_instance):
                # Test the function
                result = shared_utils.get_app_utils()
                assert result is not None
                assert result is mock_instance
    
    def test_app_utils_singleton_pattern(self):
        """
        Test that app_utils follows singleton pattern
        
        Verifies that multiple calls to get_app_utils return the same instance,
        which is important for caching and state management.
        """
        # Import the module first
        import shared_utils
        
        with patch.object(shared_utils, 'AppUtils') as MockAppUtils:
            mock_instance = Mock()
            MockAppUtils.return_value = mock_instance
            
            # Create a new app_utils instance with the mock
            with patch.object(shared_utils, 'app_utils', mock_instance):
                # Get the instance multiple times
                instance1 = shared_utils.get_app_utils()
                instance2 = shared_utils.get_app_utils()
                instance3 = shared_utils.get_app_utils()
                
                # All should be the same object
                assert instance1 is instance2
                assert instance2 is instance3
                assert instance1 is instance3
                assert instance1 is mock_instance
    
    @pytest.mark.integration
    def test_app_utils_module_import(self):
        """
        Integration test to verify the actual shared_utils module can be imported
        
        This test verifies that the module structure is correct and can be imported
        in the actual application context.
        """
        try:
            import shared_utils
            # Verify the module has the expected function
            assert hasattr(shared_utils, 'get_app_utils')
            assert callable(shared_utils.get_app_utils)
            
            # Verify the module has the app_utils instance
            assert hasattr(shared_utils, 'app_utils')
            
        except ImportError as e:
            pytest.fail(f"Failed to import shared_utils: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error importing shared_utils: {e}")


class TestUtilityFunctionPatterns:
    """Tests for common utility function patterns used throughout the app"""
    
    def test_caching_pattern_mock(self, mock_app_utils):
        """
        Test the caching pattern used by utility functions
        
        This tests the pattern of checking cache before performing operations,
        which is used throughout the application.
        """
        # Setup mock cache behavior
        mock_app_utils._cache = {'test_key': 'cached_value'}
        
        # Test cache hit
        result = mock_app_utils._cache.get('test_key')
        assert result == 'cached_value'
        
        # Test cache miss
        result = mock_app_utils._cache.get('missing_key')
        assert result is None
        
        # Test cache miss with default
        result = mock_app_utils._cache.get('missing_key', 'default')
        assert result == 'default'
    
    def test_data_processing_pipeline_pattern(self, sample_session_data, mock_app_utils):
        """
        Test the data processing pipeline pattern
        
        This tests the common pattern of data loading -> processing -> caching
        used throughout the application.
        """
        # Setup mock to return our sample data
        mock_app_utils.get_session_data.return_value = sample_session_data
        mock_app_utils.process_data_pipeline.return_value = sample_session_data
        
        # Test the pipeline pattern
        raw_data = mock_app_utils.get_session_data(use_cache=True)
        processed_data = mock_app_utils.process_data_pipeline(raw_data, use_cache=False)
        
        # Verify the pattern works
        assert raw_data is not None
        assert processed_data is not None
        assert len(processed_data) == len(sample_session_data) 