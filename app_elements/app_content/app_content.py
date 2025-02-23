from dash import html, Input, Output, callback
import dash_bootstrap_components as dbc

class AppContent:

    def build(self):
        """
        Build app content
        """
        return html.Div(
            [
                html.H1([
                         'Test'
                         ], className = 'app-content-header')   
            ]
        )
