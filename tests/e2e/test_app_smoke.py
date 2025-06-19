"""
Smoke tests for AIND Dashboard

These tests verify that the application can start and basic functionality works.
They serve as a baseline to ensure the app continues working during refactoring.
"""
import pytest
import time
import sys
import os
import subprocess
import signal
import requests
from pathlib import Path


@pytest.mark.smoke
class TestAppSmoke:
    """Basic smoke tests to verify core functionality"""
    
    def test_app_imports_successfully(self):
        """
        Test that the app can be imported without errors
        
        This verifies that all imports and initial setup work correctly
        without actually running the server.
        """
        try:
            # Test importing main components
            import shared_utils
            import app_utils
            from app_elements import AppMain
            
            # Test that we can get the shared app_utils instance
            utils = shared_utils.get_app_utils()
            assert utils is not None
            
            # Test that AppMain can be instantiated
            app_main = AppMain()
            assert app_main is not None
            
        except Exception as e:
            pytest.fail(f"Failed to import app components: {e}")
    
    def test_app_utils_initialization(self):
        """
        Test that AppUtils can be initialized and basic methods exist
        
        This verifies the core data handling infrastructure without
        requiring full data loading.
        """
        try:
            from shared_utils import get_app_utils
            
            app_utils = get_app_utils()
            
            # Check that key methods exist
            assert hasattr(app_utils, 'get_session_data')
            assert hasattr(app_utils, 'process_data_pipeline')
            assert hasattr(app_utils, 'get_table_display_data')
            assert hasattr(app_utils, 'cache_manager')
            
            # Verify cache manager has expected methods
            assert hasattr(app_utils.cache_manager, 'get')
            assert hasattr(app_utils.cache_manager, 'set')
            assert hasattr(app_utils.cache_manager, 'has')
            
        except Exception as e:
            pytest.fail(f"Failed to initialize AppUtils: {e}")
    
    @pytest.mark.slow
    def test_app_server_starts(self):
        """
        Test that the app server can start successfully
        
        This is a more comprehensive test that actually starts the server.
        It allows the full 5+ minute startup time as mentioned by the user.
        """
        import threading
        import time
        from app import app
        
        server_started = threading.Event()
        server_error = None
        
        def run_server():
            nonlocal server_error
            try:
                # Start the server in a separate thread
                app.run_server(debug=False, port=8888, host='127.0.0.1')
            except Exception as e:
                server_error = e
            finally:
                server_started.set()
        
        # Start server in background thread
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait for server to start (give it up to 6 minutes as requested)
        max_wait_time = 360  # 6 minutes
        start_time = time.time()
        
        print(f"Starting app server (allowing up to {max_wait_time//60} minutes for startup)...")
        
        while time.time() - start_time < max_wait_time:
            try:
                # Try to connect to the server
                response = requests.get('http://127.0.0.1:8888', timeout=10)
                if response.status_code == 200:
                    print("✓ App server started successfully!")
                    return  # Success!
            except requests.exceptions.RequestException:
                # Server not ready yet, continue waiting
                pass
            
            # Check if server thread failed
            if server_error:
                pytest.fail(f"Server failed to start: {server_error}")
            
            print(f"Waiting for server... ({int(time.time() - start_time)}s elapsed)")
            time.sleep(10)  # Check every 10 seconds
        
        # If we get here, the server didn't start in time
        pytest.fail(f"App server did not start within {max_wait_time} seconds")
    
    def test_app_creation_without_running(self):
        """
        Test that the Dash app object can be created without running
        
        This is a lighter test that creates the app object but doesn't
        start the server, which should be much faster.
        """
        try:
            from app import app
            
            # Verify it's a Dash app
            assert hasattr(app, 'layout')
            assert hasattr(app, 'run_server')
            assert hasattr(app, 'callback')
            
            # Try to get the layout (this might trigger some initialization)
            layout = app.layout
            assert layout is not None
            
        except Exception as e:
            pytest.fail(f"Failed to create app without running: {e}")
    
    @pytest.mark.integration
    def test_data_pipeline_smoke_test(self):
        """
        Test that the data pipeline can be called without errors
        
        This tests the data loading and processing logic that's mentioned
        in app.py as taking time during startup.
        """
        try:
            from shared_utils import get_app_utils
            
            app_utils = get_app_utils()

            session_data = app_utils.get_session_data(use_cache=True)
            assert session_data is not None
            
            table_data = app_utils.get_table_display_data(use_cache=True)
            assert table_data is not None
            
            print(f"✓ Data pipeline working: {len(session_data)} sessions, {len(table_data)} table rows")
            
        except Exception as e:
            pytest.fail(f"Data pipeline smoke test failed: {e}")


@pytest.mark.smoke
@pytest.mark.integration  
class TestAppConfiguration:
    """Test app configuration and setup"""
    
    def test_required_files_exist(self):
        """
        Test that all required files exist for the app to run
        
        This verifies the basic file structure is intact.
        """
        required_files = [
            'app.py',
            'shared_utils.py',
            'requirements.txt',
            'app_elements/__init__.py',
            'app_utils/__init__.py',
            'callbacks/__init__.py'
        ]
        
        project_root = Path(__file__).parent.parent.parent
        
        for file_path in required_files:
            full_path = project_root / file_path
            assert full_path.exists(), f"Required file missing: {file_path}"
    
    def test_environment_setup(self):
        """
        Test that the environment has required packages
        
        This verifies the basic dependencies are available.
        """
        required_packages = [
            'dash',
            'pandas', 
            'plotly',
            'dash_bootstrap_components'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                pytest.fail(f"Required package not installed: {package}") 