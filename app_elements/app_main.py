import dash_bootstrap_components as dbc
from dash import html

from .app_content import AppContent


class AppMain:
    def __init__(self):
        """Initialize main app components"""
        self.app_content = AppContent()

    def build(self):
        """
        Build app main with content
        """
        return html.Div(
            [
                # Top app bar
                dbc.Row([html.Div("AIND TEAM", className="top-bar")], className="g-0"),
                # Main content
                dbc.Row(
                    [
                        # Content column
                        dbc.Col(
                            [self.app_content.build()],
                            width=12,
                            className="content-col",
                        ),
                    ],
                    className="g-0 main-content-row",
                ),
            ],
            className="app-main",
        )
