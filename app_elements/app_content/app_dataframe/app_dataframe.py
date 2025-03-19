from dash import html, dash_table
import pandas as pd
from datetime import datetime, timedelta
from app_utils import AppUtils
from app_utils.app_analysis.reference_processor import ReferenceProcessor

class AppDataFrame:
    def __init__(self):
        # Initialize data loader through AppUtils
        app_utils = AppUtils()
        self.data_loader = app_utils.data_loader
        
        # Create reference processor with minimal default settings
        self.reference_processor = ReferenceProcessor(
            features_config={},  # Empty dict since we're not using features
            window_days=30,     # Default window of 30 days
            min_sessions=1,     # Minimal requirements since we just want the window
            min_days=1
        )

    def format_dataframe(self, df: pd.DataFrame, window_days: int = 30, reference_date: datetime = None) -> pd.DataFrame:
        """
        Format the dataframe for display in the table
        """
        df = df.copy()
        
        # Update window days in the reference processor
        self.reference_processor.window_days = window_days
        
        # Apply sliding window filter
        window_df = self.reference_processor.apply_sliding_window(df, reference_date)
        
        # Get the most current session for each subject
        window_df = window_df.sort_values('session_date', ascending=False)
        window_df = window_df.drop_duplicates(subset=['subject_id'], keep='first')

        # Define column order
        column_order = [
            'subject_id',
            'session_date',
            'session',
            'rig',
            'trainer',
            'PI',
            'current_stage_actual',
            'task',
            'session_run_time',
            'total_trials',
            'finished_trials',
            'finished_rate',
            'ignore_rate',
            'water_in_session_foraging',
            'water_in_session_manual',
            'water_in_session_total',
            'water_after_session',
            'water_day_total',
            'base_weight',
            'target_weight',
            'target_weight_ratio',
            'weight_after',
            'weight_after_ratio',
            'total_trials_with_autowater',
            'finished_trials_with_autowater',
            'finished_rate_with_autowater',
            'ignore_rate_with_autowater',
            'autowater_collected',
            'autowater_ignored',
            'water_day_total_last_session',
            'water_after_session_last_session'
        ]

        # Filter columns to include only those in the defined order
        available_columns = [col for col in column_order if col in window_df.columns]

        # Add any remaining columns at the end
        remaining_columns = [col for col in window_df.columns if col not in column_order]
        ordered_columns = available_columns + remaining_columns

        # Reorder the columns
        window_df = window_df[ordered_columns]

        return window_df

    def build(self):
        """
        Build data table component
        """
        # Get the data and apply formatting
        raw_data = self.data_loader.get_data()
        formatted_data = self.format_dataframe(raw_data)
        columns = [{"name": col, "id": col} for col in formatted_data.columns]

        # Build the table with updated styling
        return html.Div([
            dash_table.DataTable(
                id='session-table',
                data = formatted_data.to_dict('records'),
                columns = columns,
                page_size = 25,
                fixed_rows={'headers': True},
                style_table = {
                    'overflowY': 'auto',
                    'backgroundColor': 'white',
                    'height': 'calc(100vh - 250px)',  # Slightly less height to ensure it fits
                    'minHeight': '400px'  # Add minimum height
                },
                style_cell = {
                    'textAlign': 'left',
                    'padding': '12px',  # Reduced padding for more compact rows
                    'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
                    'fontSize': '14px',
                    'height': 'auto',  # Auto-adjust height based on content
                    'minWidth': '100px',
                    'backgroundColor': 'white',
                    'border': 'none'
                },
                style_header = {
                    'backgroundColor': 'white',
                    'fontWeight': '600',
                    'border': 'none',
                    'borderBottom': '1px solid #e0e0e0',
                    'position': 'sticky',  # Make header sticky
                    'top': 0,
                    'zIndex': 999
                },
                style_data_conditional = [
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#f9f9f9'
                    }
                ]
            )
        ], className="data-table-container")