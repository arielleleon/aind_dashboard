from dash import html
import dash_bootstrap_components as dbc
from .app_content import *
from .app_sidebar import *


class AppMain:
    def build(self):
        """
        Build app main with sidebar and content
        """
        return html.Div([
            # Top app bar
            dbc.Row([
                html.Div('AIND TEAM', className = 'top-bar')
            ], className = 'g-0'),

            # Main content
            dbc.Row([
                # Sidebar column
                dbc.Col([
                    AppSidebar().build()
                ], width=0.5, className="sidebar-col"),
                
                # Content column
                dbc.Col([
                    AppContent().build()
                ], width=11.5, className="content-col"),
            ], className="g-0"),  # g-0 removes gutters
        ], className="app-main")