import dash
import dash_bootstrap_components as dbc
from app_elements import *
from callbacks import *
from app_utils import AppUtils

# Initialize app utilities and cache data on startup
app_utils = AppUtils()

# Pre-load and cache the raw data
raw_data = app_utils.get_session_data(use_cache=True)

# Format the data once at startup (full dataset, no time window)
from app_elements.app_content.app_dataframe.app_dataframe import AppDataFrame
formatter = AppDataFrame()
formatted_data = formatter.format_dataframe(raw_data)

# Store in cache for reuse
app_utils._cache['formatted_data'] = formatted_data
print(f"Initialized app with {len(formatted_data)} subjects' data")

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],    
)
app.layout = AppMain().build()

if __name__ == "__main__":
    app.run_server(debug=False)