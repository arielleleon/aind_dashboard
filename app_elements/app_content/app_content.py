from dash import html, Input, Output, callback, dash_table
import dash_bootstrap_components as dbc
from .app_plot_content import AppPlotContent
from .app_dataframe import AppDataFrame
from ..app_filter import AppFilter

class AppContent:
    def build(self):
        """
        Build app content with data table and plot side by side
        """
        return html.Div([
            # Layout with filter in left column only
            dbc.Row([
                # Left column for filter and data table
                dbc.Col([
                    # Add filter component at top of left column
                    AppFilter().build(),
                    
                    # Data table below the filter
                    AppDataFrame().build()
                ], width=6, className="content-column"),
                
                # Right column for plot content
                dbc.Col([
                    AppPlotContent().build()
                ], width=6, className="content-column")
            ], className="h-100")
        ], className="content-wrapper")