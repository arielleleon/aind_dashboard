from dash import html, dash_table
from app_utils import AppLoadData

class AppDataFrame:
    def __init__(self):
        # Initialize data loader
        self.data_loader = AppLoadData()

    def build(self):
        """
        Build data table component
        """
        return html.Div([
            dash_table.DataTable(
                id='session-table',
                data = self.data_loader.get_data().to_dict('records'),
                page_size = 15,
                style_table = {
                    'height': '800px',
                    'overflowY': 'auto',
                    'backgroundColor': 'white'
                },
                style_cell = {
                    'textAlign': 'left',
                    'padding': '16px',
                    'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
                    'fontSize': '14px',
                    'height': '56px',
                    'minWidth': '100px',
                    'backgroundColor': 'white',
                    'border': 'none'
                },
                style_header = {
                    'backgroundColor': 'white',
                    'fontWeight': '600',
                    'border': 'none',
                    'borderBottom': '1px solid #e0e0e0',
                },
                style_data_conditional = [
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#f9f9f9'
                    }
                ]
            )
        ])