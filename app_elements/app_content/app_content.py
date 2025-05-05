from dash import html, dash_table
import dash_bootstrap_components as dbc
from .app_plot_content import AppPlotContent
from .app_dataframe import AppDataFrame
from ..app_filter import AppFilter
from ..app_subject_detail import AppSubjectDetail

class AppContent:
    def __init__(self):
        """Initialize app content components"""
        self.app_filter = AppFilter()
        self.app_dataframe = AppDataFrame()
        self.app_plot_content = AppPlotContent()
        self.app_subject_detail = AppSubjectDetail()

    def build(self):
        """
        Build app content with data table and plot side by side
        """
        return html.Div([
            dbc.Row([
                dbc.Col([
                    self.app_filter.build()
                ], width=6, className="filter-column"),

                # Rank change plot column
                dbc.Col([
                    self.app_plot_content.build()
                ], width=6, className="plot-column")
            ], className="g-0 top-row"),

            dbc.Row([
                dbc.Col([
                    self.app_dataframe.build()
                ], width=12, className="data-table-column")
            ], className="g-0 bottom-row"),

            # Subject details column - no margins
            dbc.Row([
                dbc.Col([
                    self.app_subject_detail.build()
                ], width=12, className="subject-detail-column p-0")  # Remove padding
            ], className="g-0 detail-row mt-0")  # Remove top margin
        ], className="content-wrapper")