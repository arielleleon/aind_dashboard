from dash import html, Input, Output, callback, dash_table
import dash_bootstrap_components as dbc
from .app_plot_content import AppPlotContent
from .app_dataframe import AppDataFrame

class AppContent:
    def build(self):
        """
        Build app content with data table and plot side by side
        """
        return html.Div([
            html.H1('Test', className='app-content-header'),
            dbc.Row([
                # Left column for data table
                dbc.Col([
                    AppDataFrame().build()
                ], width=6),
                
                # Right column for plot content
                dbc.Col([
                    AppPlotContent().build()
                ], width=6)
            ])
        ])