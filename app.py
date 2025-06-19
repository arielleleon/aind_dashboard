import warnings

import dash
import dash_bootstrap_components as dbc

from app_elements import AppMain
from app_utils.simple_logger import get_logger
from shared_utils import app_utils

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

# Cache validation
session_count = len(session_level_data)
subject_count = session_level_data["subject_id"].nunique()
logger.info(
    f"App initialized: {session_count} sessions across {subject_count} subjects"
)

from callbacks import (  # noqa: F401
    filter_callbacks,
    session_interaction_callbacks,
    subject_detail_callbacks,
    table_callbacks,
    tooltip_callbacks,
    visualization_callbacks,
)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)
app.layout = AppMain().build()

if __name__ == "__main__":
    app.run_server(debug=False)
