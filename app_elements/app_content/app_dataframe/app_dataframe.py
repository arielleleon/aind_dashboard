from dash import html, dash_table
import pandas as pd
from datetime import datetime, timedelta
import traceback
from app_utils import AppUtils
from app_utils.app_analysis.reference_processor import ReferenceProcessor
from dash import callback_context, Input, Output, clientside_callback

class AppDataFrame:
    def __init__(self):
        # Initialize app utilities
        self.app_utils = AppUtils()

        # Feature configuration for quantile analysis
        self.features_config = {
            'finished_trials': False,  # Higher is better
            'ignore_rate': True,     # Lower is better
            'total_trials': False,   # Higher is better
            'foraging_performance': False,   # Higher is better
            'abs(bias_naive)': True  # Lower is better 
        }

        self.data_loader = self.app_utils.data_loader
        
        # Create reference processor with minimal default settings for filtering purposes
        self.reference_processor = ReferenceProcessor(
            features_config={},  # Empty dict since we're not using features
            window_days=30,     # Default window of 30 days
            min_sessions=1,     # Minimal requirements since we just want the window
            min_days=1
        )

    def format_dataframe(self, df: pd.DataFrame, window_days: int = 30, reference_date: datetime = None) -> pd.DataFrame:
        """
        Format the dataframe for display in the table

        1. Filter data to sliding window
        2. Get most recent sessions for each subject
        3. Run quantile analysis pipeline on same window
        4. Match subjects and add alert categories
        """
        df = df.copy()
        print(f" Starting data formatting with {len(df)} sessions, {window_days} day window")
        
        # Step 1: Update window days in reference processor and apply sliding window filter
        self.reference_processor.window_days = window_days
        window_df = self.reference_processor.apply_sliding_window(df, reference_date)
        print(f"Applied sliding window: {len(window_df)} sessions")

        # Step 2: Get most current session for each subject in the window
        window_df = window_df.sort_values('session_date', ascending=False)
        current_sessions_df = window_df.drop_duplicates(subset=['subject_id'], keep='first')
        print(f"Got current sessions for {len(current_sessions_df)} subjects")

        # Add percentile category column with default value
        current_sessions_df['percentile_category'] = 'NS'  # Default to Not Scored
        # Add overall percentile column
        current_sessions_df['overall_percentile'] = float('nan')

        # Step 3: Run quantile analysis pipeline on same window data
        try: 
            print(f" Starting quantile analysis pipeline for {len(window_df)} sessions") # DEBUGGING

            # Reset app_utils
            app_utils = AppUtils()

            # Step 3.1: Initialize reference processor with feature congfiguration
            reference_processor = app_utils.initialize_reference_processor(
                features_config=self.features_config,
                window_days = window_days,
                min_sessions = 1,  # Minimum sessions requirement for eligibility
                min_days = 1       # Minimum days requirement for eligibility
            )
            print(f" Initialized reference processor")

            # Step 3.2: Process data through reference pipeline to create strata
            stratified_data = app_utils.process_reference_data(
                df = window_df, 
                reference_date = reference_date,
                remove_outliers = False
            )
            print(f" Created {len(stratified_data)} strata")

            # Step 3.3: Initialize quantile analyzer with strata
            quantile_analyzer = app_utils.initialize_quantile_analyzer(stratified_data)
            print(f" Initialized quantile analyzer")

            # Step 3.4: Initialize alert service
            alert_service = app_utils.initialize_alert_service()
            print(f" Initialized alert service")

            # Step 3.5: Get subject IDs from current sessions dataframe
            subject_ids = current_sessions_df['subject_id'].tolist()

            # Step 3.6: Calculate overall percentiles for subjects
            overall_percentiles = quantile_analyzer.calculate_overall_percentile(subject_ids)
            print(f"Calculated overall percentiles for {len(overall_percentiles)} subjects") # DEBUGGING

            # Step 4: Map overall percentiles to alert categories and update the dataframe
            alert_count = 0
            for i, row in current_sessions_df.iterrows():
                subject_id = row['subject_id']
                
                # Find this subject in overall percentiles
                subject_percentile = overall_percentiles[overall_percentiles['subject_id'] == subject_id]
                
                if not subject_percentile.empty:
                    # Get the overall percentile value
                    overall_percentile = subject_percentile['overall_percentile'].iloc[0]

                    # Store percentile value in dataframe
                    current_sessions_df.loc[i, 'overall_percentile'] = overall_percentile

                    # Get strata and add to the dataframe
                    if 'strata' in subject_percentile.columns:
                        strata = subject_percentile['strata'].iloc[0]
                        current_sessions_df.loc[i, 'strata'] = strata
                    
                    # Map to a category
                    category = alert_service.map_overall_percentile_to_category(overall_percentile)
                    
                    # Update dataframe
                    current_sessions_df.at[i, 'percentile_category'] = category
                    
                    # Count non-normal alerts
                    if category not in ['N', 'NS']:
                        alert_count += 1

            print(f"Total alerts found: {alert_count} out of {len(current_sessions_df)} subjects") # DEBUGGING 

        except Exception as e:
            print(f"Error in quantile analysis pipeline: {str(e)}")
            print(traceback.format_exc())

        # Define column order 
        column_order = [
            'subject_id',
            'percentile_category',
            'overall_percentile',
            'strata',
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
        available_columns = [col for col in column_order if col in current_sessions_df.columns]

        # Add any remaining columns at the end
        remaining_columns = [col for col in current_sessions_df.columns if col not in column_order]
        ordered_columns = available_columns + remaining_columns

        # Reorder the columns
        formatted_df = current_sessions_df[ordered_columns]

        print(f" Formatted dataframe with {len(formatted_df)} rows and {len(formatted_df.columns)} columns")
        return formatted_df
    
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
            'percentile_category': 'Alert',  # Add label for the alert column
            'overall_percentile': 'Percentile',
            'strata': 'Strata',
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
                column_def['format'] = {"specifier": ".5~g"}
            
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
            # Row-level highlighting with lighter colors
            {
                'if': {'filter_query': '{percentile_category} eq "SB"'},
                'backgroundColor': '#FFC380',  # Light orange
                'className': 'alert-sb-row'
            },
            {
                'if': {'filter_query': '{percentile_category} eq "B"'},
                'backgroundColor': '#FFC380',  # Very light orange
                'className': 'alert-b-row'
            },
            {
                'if': {'filter_query': '{percentile_category} eq "G"'},
                'backgroundColor': '#9FC5E8',  # Light blue
                'className': 'alert-g-row'
            },
            {
                'if': {'filter_query': '{percentile_category} eq "SG"'},
                'backgroundColor': '#9FC5E8',  # Slightly darker blue
                'className': 'alert-sg-row'
            }
        ]

        # Add highlight for columns used for overall percentile calculation
        percentile_columns = list(self.features_config.keys())

        for col in percentile_columns:
            if col in formatted_data.columns:
                # Add blue border to column
                conditional_styles.append({
                    'if': {'column_id': col},
                    'border': '2px solid #5D9FD3',
                    'borderRadius': '2px'
                })

        # Also highlight overall percentile column
        conditional_styles.append({
            'if': {'column_id': 'overall_percentile'},
            'border': '2px solid #5D9FD3',
            'borderRadius': '2px'
        })

        # Build the table with updated styling
        return html.Div([
            dash_table.DataTable(
                id='session-table',
                data=formatted_data.to_dict('records'),
                columns=columns,
                page_size=20,
                fixed_rows={'headers': True},
                style_table={
                    'overflowY': 'auto',
                    'overflowX': 'auto',
                    'backgroundColor': 'white',
                    'height': 'calc(100vh - 300px)',
                    'minHeight': '500px'
                },
                style_cell={
                    'textAlign': 'left',
                    'padding': '12px',
                    'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
                    'fontSize': '14px',
                    'height': 'auto',
                    'minWidth': '100px',
                    'backgroundColor': 'white',
                    'border': 'none'
                },
                style_header={
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
                style_data_conditional=conditional_styles
            )
        ], className="data-table-container")