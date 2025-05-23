from dash import html, dash_table
import dash_bootstrap_components as dbc
from .app_dataframe import AppDataFrame
from ..app_filter import AppFilter
from ..app_subject_detail import AppSubjectDetail
from shared_utils import app_utils

class AppContent:
    def __init__(self):
        """Initialize app content components"""
        self.app_filter = AppFilter()
        self.app_dataframe = AppDataFrame(app_utils=app_utils)
        self.app_subject_detail = AppSubjectDetail()

    def build(self):
        """
        Build app content with filter at top, data table in middle, and subject details at bottom
        """
        return html.Div([
            # Filter row
            dbc.Row([
                dbc.Col([
                    self.app_filter.build()
                ], width=12, className="filter-column")
            ], className="g-0 filter-row"),

            # Data table row
            dbc.Row([
                dbc.Col([
                    self.app_dataframe.build()
                ], width=12, className="data-table-column")
            ], className="g-0 data-row"),
            
            # Subject details row
            dbc.Row([
                dbc.Col([
                    self.app_subject_detail.build()
                ], width=12, className="subject-detail-column p-0")
            ], className="g-0 detail-row mt-0")
        ], className="content-wrapper")