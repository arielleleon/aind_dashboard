import warnings

import dash
import dash_bootstrap_components as dbc
import pandas as pd

from app_elements import *
from app_utils.simple_logger import get_logger
from shared_utils import app_utils

# Suppress pandas FutureWarnings about fillna downcasting to reduce terminal spam
warnings.filterwarnings("ignore", category=FutureWarning, message=".*fillna.*")
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*Downcasting object dtype arrays.*"
)

# Initialize logger for app startup
logger = get_logger("startup")

# Pre-load and cache the raw data using the unified pipeline
logger.info("Initializing app with unified data pipeline...")
raw_data = app_utils.get_session_data(use_cache=True)
session_level_data = app_utils.process_data_pipeline(raw_data, use_cache=False)

# Ensure all cache structures are properly initialized
logger.info("Ensuring all cache structures are initialized...")
table_data = app_utils.get_table_display_data(use_cache=True)
optimized_storage = app_utils.cache_manager.get("optimized_storage")
ui_structures = app_utils.cache_manager.get("ui_structures")

# Cache validation (simplified, removed verbose details)
session_count = len(session_level_data)
subject_count = session_level_data["subject_id"].nunique()
logger.info(
    f"App initialized: {session_count} sessions across {subject_count} subjects"
)

# Import callbacks AFTER shared_utils is loaded
from callbacks import *

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)
app.layout = AppMain().build()

if __name__ == "__main__":
    app.run_server(debug=False)
