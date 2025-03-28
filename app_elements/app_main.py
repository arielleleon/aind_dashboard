from dash import html
import dash_bootstrap_components as dbc
from .app_content import *


class AppMain:
    def build(self):
        """
        Build app main with content
        """
        return html.Div([
            # Top app bar
            dbc.Row([
                html.Div('AIND TEAM', className = 'top-bar')
            ], className = 'g-0'),

            # Main content
            dbc.Row([
                # Content column
                dbc.Col([
                    AppContent().build()
                ], width=12, className="content-col"),
            ], className="g-0 main-content-row"),
        ], className="app-main")