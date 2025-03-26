from dash import html, dash_table
import pandas as pd
from datetime import datetime, timedelta
from app_utils import AppUtils
from app_utils.app_analysis.reference_processor import ReferenceProcessor
from dash import callback_context, Input, Output, clientside_callback

class AppDataFrame:
    def __init__(self):
        # Initialize data loader through AppUtils
        app_utils = AppUtils()

        # Feature configuration for alerts
        features_config = {
            'finished_trials': False,  # Higher is better
            'ignore_rate': True,     # Lower is better
            'total_trials': False,   # Higher is better
            'foraging_performance': False,   # Higher is better
            'abs(bias_naive)': True  # Lower is better 
        }
        app_utils.initialize_reference_processor(features_config=features_config)

        threshold_config = {
            'water_day_total': {
                'upper': 3.5
            }
        }
        app_utils.initialize_threshold_analyzer(feature_thresholds=threshold_config)


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

        # Add alert columns with default values
        window_df['has_threshold_alert'] = False
        window_df['percentile_category'] = 'N'

        # Get alert information
        try:
            app_utils = AppUtils()
            if hasattr(app_utils, 'alert_service') and app_utils.alert_service is not None:
                if not hasattr(app_utils.alert_service, 'app_utils') or app_utils.alert_service.app_utils is None:
                    app_utils.alert_service.set_app_utils(app_utils)

                # Get subject IDs from the window dataframe
                subject_ids = window_df['subject_id'].tolist()

                # Get alerts for the subjects
                alerts = app_utils.get_alerts(subject_ids)

                # Add alert columns to dataframe
                for i, row in window_df.iterrows():
                    sid = row['subject_id']
                    if sid in alerts:
                        # Check for threshold alerts
                        window_df.at[i, 'has_threshold_alert'] = bool(alerts.get(sid, {}).get('threshold', {}))

                        # Get percentile category
                        worst_category = self._get_worst_percentile_category(alerts.get(sid, {}).get('percentile', {}))
                        window_df.at[i, 'percentile_category'] = worst_category
        except Exception as e:
            print(f"Error adding alerts to dataframe: {str(e)}")

        # Define column order 
        column_order = [
            'subject_id',
            'session_date',
            'session',
            'rig',
            'trainer',
            'PI',
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
            'water_after_session_last_session',
            'current_stage_actual',
            'task',
            'session_run_time',
            'total_trials',
            'finished_trials',
            'finished_rate',
            'ignore_rate',
        ]

        # Filter columns to include only those in the defined order
        available_columns = [col for col in column_order if col in window_df.columns]

        # Add any remaining columns at the end
        remaining_columns = [col for col in window_df.columns if col not in column_order]
        ordered_columns = available_columns + remaining_columns

        # Reorder the columns
        window_df = window_df[ordered_columns]

        return window_df
    
    def _get_worst_percentile_category(self, quantile_alerts):
        """ Helper function to get the worst percentile category from quantile alerts"""
        if not quantile_alerts or 'current' not in quantile_alerts:
            return 'N' # Default to normal

        # Define priority order
        priority_order = ['SB', 'B', 'N', 'G', 'SG']

        # Check all current strata features for worst category
        for strata_data in quantile_alerts.get('current', {}).values():
            for feature_data in strata_data.values():
                category = feature_data.get('category', 'N')
                # Return if this is the worst category
                if category == 'SB':
                    return 'SB'
                
        # If no SB found, go up in priority order
        for priority in priority_order:
            for strata_data in quantile_alerts.get('current', {}).values():
                for feature_data in strata_data.values():
                    if feature_data.get('category', 'N') == priority:
                        return priority
                    
        return 'N' # Default to normal if no category found
        
    def build(self):
        """
        Build data table component
        """
        # Get the data and apply formatting
        raw_data = self.data_loader.get_data()
        formatted_data = self.format_dataframe(raw_data)
        
        # Identify float columns for formatting
        float_columns = [col for col in formatted_data.columns if formatted_data[col].dtype == 'float64']
        
        # Improve column header display
        formatted_column_names = {
            'subject_id': 'Subject ID',
            'session_date': 'Date',
            'session': 'Session',
            'rig': 'Rig',
            'trainer': 'Trainer',
            'PI': 'PI',
            'current_stage_actual': 'Stage',
            'task': 'Task',
            'session_run_time': 'Run Time',
            'total_trials': 'Total Trials',
            'finished_trials': 'Finished Trials',
            'finished_rate': 'Finish Rate',
            'ignore_rate': 'Ignore Rate',
            'water_in_session_foraging': 'Water In-Session\n(Foraging)',
            'water_in_session_manual': 'Water In-Session\n(Manual)',
            'water_in_session_total': 'Water In-Session\n(Total)',
            'water_after_session': 'Water After\nSession',
            'water_day_total': 'Water Day\nTotal',
            'base_weight': 'Base Weight',
            'target_weight': 'Target Weight',
            'target_weight_ratio': 'Target Weight\nRatio',
            'weight_after': 'Weight After',
            'weight_after_ratio': 'Weight After\nRatio',
            'total_trials_with_autowater': 'Total Trials\n(Autowater)',
            'finished_trials_with_autowater': 'Finished Trials\n(Autowater)',
            'finished_rate_with_autowater': 'Finish Rate\n(Autowater)',
            'ignore_rate_with_autowater': 'Ignore Rate\n(Autowater)',
            'autowater_collected': 'Autowater\nCollected',
            'autowater_ignored': 'Autowater\nIgnored',
            'water_day_total_last_session': 'Water Day Total\n(Last Session)',
            'water_after_session_last_session': 'Water After\n(Last Session)'
        }

        # Create columns with formatted names and custom numeric formatting
        columns = []
        for col in formatted_data.columns:
            column_def = {
                "name": formatted_column_names.get(col, col.replace('_', ' ').title()),
                "id": col
            }
            
            # Add specific formatting for float columns
            if col in float_columns:
                column_def['type'] = 'numeric'
                column_def['format'] = {
                    "specifier": ".5~g"
                }

            # Hide alert columns
            #if col in ['has_threshold_alert', 'percentile_category']:
            #    column_def['hidden'] = True
            
            columns.append(column_def)

        # Define conditional styles for alerts
        conditional_styles = [
            # Default row styling
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#f9f9f9'
            },
            {
                'if': {'column_id': 'subject_id'},
                'fontWeight': 'bold'
            },
            # Percentile category styling
            {
                'if': {'filter_query': '{percentile_category} eq "SB"'},
                'backgroundColor': 'rgba(251, 133, 0, 0.3)' # Orange 30% opacity
            },
            {
                'if': {'filter_query': '{percentile_category} eq "B"'},
                'backgroundColor': 'rgba(255, 165, 0, 0.15)' # Orange 15% opacity
            },
            {
                'if': {'filter_query': '{percentile_category} eq "G"'},
                'backgroundColor': 'rgba(0, 48, 87, 0.15)' # Blue 15% opacity
            },
            {
                'if': {'filter_query': '{percentile_category} eq "SG"'},
                'backgroundColor': 'rgba(0, 48, 87, 0.3)' # Blue 30% opacity
            },

            # Threshold alert styling
            {
                'if': {'filter_query': '{has_threshold_alert} eq True'},
                'border-left': '4px solid #fb8500' # Orange dot left side
            }
                
        ]

        # Build the table with updated styling
        return html.Div([
            dash_table.DataTable(
                id='session-table',
                data = formatted_data.to_dict('records'),
                columns = columns,
                page_size = 16,
                fixed_rows={'headers': True},
                style_table = {
                    'overflowY': 'auto',
                    'overflowX': 'auto',
                    'backgroundColor': 'white',
                    'height': 'calc(100vh - 300px)',
                    'minHeight': '500px'
                },
                style_cell = {
                    'textAlign': 'left',
                    'padding': '12px',
                    'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
                    'fontSize': '14px',
                    'height': 'auto',
                    'minWidth': '100px',
                    'backgroundColor': 'white',
                    'border': 'none'
                },
                style_header = {
                    'backgroundColor': 'white',
                    'fontWeight': '600',
                    'border': 'none',
                    'borderBottom': '1px solid #e0e0e0',
                    'position': 'sticky',
                    'top': 0,
                    'zIndex': 999,
                    'height': 'auto',
                    'whiteSpace': 'normal',
                    'textAlign': 'center',  
                    'padding': '10px 5px',  
                    'lineHeight': '15px'     
                },
                style_data_conditional = conditional_styles
            )
        ], className="data-table-container")