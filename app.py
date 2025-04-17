import dash
import dash_bootstrap_components as dbc
from app_elements import *
from callbacks import *
from app_utils import AppUtils
from app_elements.app_content.app_dataframe.tooltips import TooltipController
tooltip_controller = TooltipController()

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],    
)
tooltip_controller.register_callbacks(app)
app.layout = AppMain().build()

if __name__ == "__main__":
    app.run_server(debug=False)