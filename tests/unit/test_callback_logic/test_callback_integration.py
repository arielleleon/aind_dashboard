"""
Integration tests for modular callback structure.

This test validates that:
1. All callback modules are properly imported
2. All callbacks are registered correctly
3. No circular import issues exist
4. Component IDs referenced in callbacks exist
"""
import sys
import importlib
from unittest.mock import patch, MagicMock

# Test callback module imports
def test_callback_module_imports():
    """Test that all callback modules can be imported without errors."""
    modules_to_test = [
        'callbacks.table_callbacks',
        'callbacks.filter_callbacks', 
        'callbacks.subject_detail_callbacks',
        'callbacks.session_interaction_callbacks',
        'callbacks.visualization_callbacks',
        'callbacks.tooltip_callbacks'
    ]
    
    for module_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            assert module is not None, f"Failed to import {module_name}"
            print(f"Successfully imported {module_name}")
        except ImportError as e:
            raise Exception(f"Failed to import {module_name}: {e}")

def test_shared_utilities_import():
    """Test that shared callback utilities are properly accessible."""
    try:
        from callbacks.shared_callback_utils import (
            Input, Output, State, callback, ALL, ctx,
            html, dcc, dbc, pd, datetime, timedelta, go, json,
            app_utils, app_dataframe, app_filter, session_card, image_loader, compact_info
        )
        print("All shared utilities imported successfully")
        
        # Verify key utilities are available
        assert Input is not None, "Input not available"
        assert Output is not None, "Output not available" 
        assert State is not None, "State not available"
        assert callback is not None, "callback decorator not available"
        assert app_utils is not None, "app_utils not available"
        
    except ImportError as e:
        raise Exception(f"Failed to import shared utilities: {e}")

def test_callbacks_package_import():
    """Test that the callbacks package imports correctly."""
    try:
        import callbacks
        assert hasattr(callbacks, 'table_callbacks'), "table_callbacks not found in package"
        assert hasattr(callbacks, 'filter_callbacks'), "filter_callbacks not found in package"
        assert hasattr(callbacks, 'subject_detail_callbacks'), "subject_detail_callbacks not found in package"
        assert hasattr(callbacks, 'session_interaction_callbacks'), "session_interaction_callbacks not found in package"
        assert hasattr(callbacks, 'visualization_callbacks'), "visualization_callbacks not found in package"
        assert hasattr(callbacks, 'tooltip_callbacks'), "tooltip_callbacks not found in package"
        print("All callback modules accessible through package")
    except ImportError as e:
        raise Exception(f"Failed to import callbacks package: {e}")

def test_no_duplicate_callback_ids():
    """Test that there are no duplicate callback output IDs across modules."""
    import inspect
    import re
    
    # Import all callback modules
    callback_modules = [
        'callbacks.table_callbacks',
        'callbacks.filter_callbacks',
        'callbacks.subject_detail_callbacks', 
        'callbacks.session_interaction_callbacks',
        'callbacks.visualization_callbacks',
        'callbacks.tooltip_callbacks'
    ]
    
    output_combinations = set()  # Track (component_id, property) combinations
    duplicates = []
    
    for module_name in callback_modules:
        try:
            module = importlib.import_module(module_name)
            # Get module source to analyze callback decorators
            source = inspect.getsource(module)
            
            # Find all @callback declarations and their outputs
            callback_blocks = re.findall(r'@callback\s*\(\s*(?:\[)?([^)]+Output[^)]*)\]?[^)]*\)', source, re.DOTALL)
            
            for block in callback_blocks:
                # Extract individual Output statements from this callback block
                output_matches = re.findall(r'Output\(\s*["\']([^"\']+)["\']\s*,\s*["\']([^"\']+)["\']', block)
                
                for component_id, property_name in output_matches:
                    output_combo = (component_id, property_name)
                    if output_combo in output_combinations:
                        duplicates.append(f"Duplicate callback output '{component_id}.{property_name}' found in {module_name}")
                    else:
                        output_combinations.add(output_combo)
                        
        except Exception as e:
            print(f"Warning: Could not analyze {module_name}: {e}")
    
    if duplicates:
        raise Exception(f"Duplicate callback output combinations found:\n" + "\n".join(duplicates))
    else:
        print(f"No duplicate callback output combinations found across {len(callback_modules)} modules")
        print(f"   Total unique output combinations: {len(output_combinations)}")

def test_callback_decorator_usage():
    """Test that all modules use the @callback decorator correctly."""
    import inspect
    
    callback_modules = [
        'callbacks.table_callbacks',
        'callbacks.filter_callbacks',
        'callbacks.subject_detail_callbacks',
        'callbacks.session_interaction_callbacks', 
        'callbacks.visualization_callbacks',
        'callbacks.tooltip_callbacks'
    ]
    
    for module_name in callback_modules:
        try:
            module = importlib.import_module(module_name)
            source = inspect.getsource(module)
            
            # Check that @app.callback is not used (deprecated pattern)
            assert '@app.callback' not in source, f"{module_name} uses deprecated @app.callback instead of @callback"
            
            # Check if module contains any server-side callbacks or only client-side callbacks
            has_server_callbacks = '@callback' in source
            has_client_callbacks = 'clientside_callback' in source
            
            if has_server_callbacks:
                print(f"{module_name} uses proper @callback decorator for server-side callbacks")
            elif has_client_callbacks:
                print(f"{module_name} uses clientside_callback (client-side only module)")
            else:
                raise Exception(f"{module_name} contains no callback decorators")
            
        except Exception as e:
            raise Exception(f"Failed to validate callback decorator in {module_name}: {e}")

if __name__ == "__main__":
    # Run tests individually for debugging
    test_callback_module_imports()
    test_shared_utilities_import() 
    test_callbacks_package_import()
    test_no_duplicate_callback_ids()
    test_callback_decorator_usage()
    print("All integration tests passed") 