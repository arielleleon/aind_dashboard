from dash import html, dash_table
import dash_bootstrap_components as dbc
from .app_plot_content import AppPlotContent
from .app_dataframe import AppDataFrame
from ..app_filter import AppFilter

class AppContent:
    def __init__(self):
        """Initialize app content components"""
        self.app_filter = AppFilter()
        self.app_dataframe = AppDataFrame()
        self.app_plot_content = AppPlotContent()
        
    def build(self):
        """
        Build app content with data table and plot side by side
        """
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        self.app_filter.build(),
                        html.Div([
                            self.app_dataframe.build()
                        ], style={"flex": "1", "overflow": "auto"})
                    ], style={"display": "flex", "flexDirection": "column", "height": "100vh"})
                ], width=6, className="content-column"),
                dbc.Col([
                    self.app_plot_content.build()
                ], width=6, className="content-column")
            ], className="g-0")
        ], className="content-wrapper")