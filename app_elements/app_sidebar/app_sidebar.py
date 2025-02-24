import dash_bootstrap_components as dbc
from dash import html

class AppSidebar:
    def build(self):
        """
        Build app sidebar with navigation elements
        """
        return html.Div([], className="sidebar")
