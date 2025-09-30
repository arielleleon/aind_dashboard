import warnings
import os
import threading
import time

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
import dash.dependencies

from app_utils.simple_logger import get_logger
from shared_utils import app_utils
from callbacks import (  # noqa: F401
    filter_callbacks,
    session_interaction_callbacks,
    subject_detail_callbacks,
    table_callbacks,
    tooltip_callbacks,
    visualization_callbacks,
)

warnings.filterwarnings("ignore", category=FutureWarning, message=".*fillna.*")
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*Downcasting object dtype arrays.*"
)

# Initialize logger for app startup
logger = get_logger("startup")

# Global flag to track data loading status
data_loading_status = {
    "raw_data_loaded": False,
    "processed_data_loaded": False,
    "ui_data_optimized": False,
    "loading_error": None
}

def load_data_background():
    """Load data in background thread after app starts"""
    try:
        logger.info("Starting background data loading...")
        start_time = time.time()
        
        # Pre-load and cache the raw data using the unified pipeline
        logger.info("Loading raw session data...")
        raw_data_start = time.time()
        raw_data = app_utils.get_session_data(use_cache=True)
        raw_data_time = time.time() - raw_data_start
        logger.info(f"Raw data loaded in {raw_data_time:.2f} seconds - {len(raw_data)} sessions")
        data_loading_status["raw_data_loaded"] = True
        
        # Process session-level data (including heavy percentile calculations)
        logger.info("Processing session-level data pipeline...")
        pipeline_start = time.time()
        session_level_data = app_utils.process_data_pipeline(raw_data, use_cache=False)
        pipeline_time = time.time() - pipeline_start
        subject_count = session_level_data["subject_id"].nunique()
        logger.info(f"Data pipeline processed in {pipeline_time:.2f} seconds - {len(session_level_data)} sessions across {subject_count} subjects")
        data_loading_status["processed_data_loaded"] = True

        # Ensure all cache structures are properly initialized
        logger.info("Optimizing UI data structures...")
        ui_start = time.time()
        app_utils.get_table_display_data(use_cache=True)
        app_utils.cache_manager.get("optimized_storage")  # Trigger cache creation
        app_utils.cache_manager.get("ui_structures")  # Trigger cache creation
        ui_time = time.time() - ui_start
        logger.info(f"UI structures optimized in {ui_time:.2f} seconds")
        data_loading_status["ui_data_optimized"] = True

        total_time = time.time() - start_time
        logger.info(f"Background data loading completed in {total_time:.2f} seconds")

    except Exception as e:
        error_msg = f"Background data loading failed: {str(e)}"
        logger.error(error_msg)
        data_loading_status["loading_error"] = error_msg


# Start data loading in background thread
logger.info("Starting Dash application with background data loading...")
data_thread = threading.Thread(target=load_data_background, daemon=True)
data_thread.start()

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

# Suppress callback exceptions for dynamic layout loading
app.config.suppress_callback_exceptions = True

# Make server accessible for gunicorn
application = app.server

# Use a simple loading layout initially to avoid blocking on data access
app.layout = html.Div([
    html.Div(id="main-content", children=[
        html.Div([
            html.H2("AIND Dashboard Loading...", className="text-center"),
            html.P("Please wait while data is being loaded.", className="text-center text-muted"),
            dbc.Spinner(color="primary", size="md")
        ], className="d-flex flex-column align-items-center justify-content-center", 
           style={"minHeight": "50vh"})
    ]),
    dcc.Interval(
        id='loading-interval',
        interval=1000,  # Check every second
        n_intervals=0,
        max_intervals=300  # Stop after 5 minutes
    ),
    html.Div(id="loading-status", style={"display": "none"})
])

logger.info("Dash application layout initialized - server starting...")

# Add callback to load the main content once data is ready
@app.callback(
    [dash.Output("main-content", "children"),
     dash.Output("loading-interval", "disabled")],
    [dash.Input("loading-interval", "n_intervals")],
    prevent_initial_call=False
)
def update_main_content(n_intervals):
    """Replace loading screen with main content once data is ready"""
    # Check if data loading is complete or if there's an error
    if data_loading_status["loading_error"]:
        # Show error message
        return [
            html.Div([
                html.H2("Error Loading Data", className="text-center text-danger"),
                html.P(f"Error: {data_loading_status['loading_error']}", className="text-center"),
                html.Button("Reload Page", id="reload-btn", className="btn btn-primary")
            ], className="d-flex flex-column align-items-center justify-content-center", 
               style={"minHeight": "50vh"})
        ], True
    
    elif data_loading_status["ui_data_optimized"]:
        # Data is ready, load the main application
        logger.info("Loading main application content")
        try:
            from app_elements import AppMain
            main_app = AppMain()
            return main_app.build(), True  # Disable interval once loaded
        except Exception as e:
            logger.error(f"Error building main content: {e}")
            return [
                html.Div([
                    html.H2("Error Building Interface", className="text-center text-danger"),
                    html.P(f"Error: {str(e)}", className="text-center")
                ], className="d-flex flex-column align-items-center justify-content-center")
            ], True
    
    else:
        # Still loading, show progress
        status_msg = "Loading data..."
        if data_loading_status["raw_data_loaded"]:
            status_msg = "Processing data pipeline..."
        if data_loading_status["processed_data_loaded"]:
            status_msg = "Optimizing interface..."
            
        return [
            html.Div([
                html.H2("AIND Dashboard Loading...", className="text-center"),
                html.P(status_msg, className="text-center text-muted"),
                dbc.Spinner(color="primary", size="md"),
                html.P(f"Checking... ({n_intervals})", className="text-center text-secondary mt-2", style={"fontSize": "12px"})
            ], className="d-flex flex-column align-items-center justify-content-center", 
               style={"minHeight": "50vh"})
        ], False  # Keep interval running

if __name__ == "__main__":
    # For local development
    app.run_server(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
