import dash
import dash_bootstrap_components as dbc
from app_elements import *
from callbacks import *
from app_utils import *

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

if __name__ == "__main__":
    app.run_server(debug=False)