import dash
import dash_bootstrap_components as dbc
from app_elements import *
from shared_utils import app_utils

# Pre-load and cache the raw data using the unified pipeline
print("🚀 Initializing app with unified data pipeline...")
raw_data = app_utils.get_session_data(use_cache=True)
session_level_data = app_utils.process_data_pipeline(raw_data, use_cache=False)

print(f"✅ App initialized with {len(session_level_data)} sessions across {session_level_data['subject_id'].nunique()} subjects")

# Import callbacks AFTER shared_utils is loaded
from callbacks import *

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],    
)
app.layout = AppMain().build()

if __name__ == "__main__":
    app.run_server(debug=False)