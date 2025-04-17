from dash import html, dash_table
import pandas as pd
from datetime import datetime, timedelta
import traceback
from app_utils import AppUtils
from app_utils.app_analysis.reference_processor import ReferenceProcessor
from dash import callback_context, Input, Output, clientside_callback
from .tooltips import TooltipController

class AppDataFrame:
    def __init__(self):
        # Initialize app utilities
        self.app_utils = AppUtils()

        # Initialize tooltip controller
        self.tooltip_controller = TooltipController()

        # Feature configuration for quantile analysis
        self.features_config = {
            'finished_trials': False,  # Higher is better
            'ignore_rate': True,     # Lower is better
            'total_trials': False,   # Higher is better
            'foraging_performance': False,   # Higher is better
            'abs(bias_naive)': True  # Lower is better 
        }

        # Configure threshold alerts
        self.threshold_config = {
            'session': {
                'condition': 'gt',
                'value': 40  # Total sessions threshold
            },
            'water_day_total': {
                'condition': 'gt',
                'value': 3.5  # Water day total threshold (ml)
            }
        }

        # Stage-specific session thresholds
        self.stage_thresholds = {
            'STAGE_1': 5,
            'STAGE_2': 5,
            'STAGE_3': 6,
            'STAGE_4': 10,
            'STAGE_FINAL': 10,
            'GRADUATED': 20
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

        1. Pre-compute all-time percentiles once
        2. Filter data for display based on window_days
        3. Get most recent sessions for each subject in the display window
        4. Apply alerts based on pre-computed percentiles
        """
        df = df.copy()
        print(f" Starting data formatting with {len(df)} sessions, {window_days} day window")
        
        # Step 1: Compute all-time percentiles if not already done
        if not hasattr(self, 'overall_percentiles'):
            print("Computing all-time percentiles...")
            
            # Use a very large window for all-time computation (365 days * 10 years)
            all_time_window = 365 * 10
            
            # Initialize app_utils
            app_utils = AppUtils()
            
            # Initialize reference processor for all-time percentile calculation
            reference_processor = app_utils.initialize_reference_processor(
                features_config=self.features_config,
                window_days=all_time_window,
                min_sessions=1,  # Minimum sessions requirement for eligibility
                min_days=1       # Minimum days requirement for eligibility
            )
            
            # Apply sliding window for all-time data
            all_time_df = reference_processor.apply_sliding_window(df, reference_date)
            print(f"Applied all-time window: {len(all_time_df)} sessions")
            
            # Process data through reference pipeline to create strata
            stratified_data = app_utils.process_reference_data(
                df=all_time_df, 
                reference_date=reference_date,
                remove_outliers=False
            )
            print(f"Created {len(stratified_data)} strata")
            
            # Initialize quantile analyzer with strata
            quantile_analyzer = app_utils.initialize_quantile_analyzer(stratified_data)
            print(f"Initialized quantile analyzer")
            
            # Initialize alert service
            alert_service = app_utils.initialize_alert_service()
            print(f"Initialized alert service")
            
            # Initialize threshold analyzer
            threshold_analyzer = app_utils.initialize_threshold_analyzer()
            threshold_analyzer.set_threshold_config(self.threshold_config)
            print(f"Initialized threshold analyzer")
            
            # Calculate overall percentiles for all subjects (one-time calculation)
            self.overall_percentiles = quantile_analyzer.calculate_overall_percentile()
            self.alert_service = alert_service
            self.threshold_analyzer = threshold_analyzer
            print(f"Calculated overall percentiles for {len(self.overall_percentiles)} subjects")
        
        # Step 2: Apply sliding window filter for DISPLAY only
        # Make sure we have a reference processor for display filtering
        if not hasattr(self, 'display_processor'):
            self.display_processor = ReferenceProcessor(
                features_config={},  # Empty dict since we're only using it for filtering
                window_days=window_days,
                min_sessions=1,
                min_days=1
            )
        else:
            self.display_processor.window_days = window_days
        
        window_df = self.display_processor.apply_sliding_window(df, reference_date)
        print(f"Applied display window: {len(window_df)} sessions")

        # Step 3: Get most current session for each subject in the window
        window_df = window_df.sort_values('session_date', ascending=False)
        current_sessions_df = window_df.drop_duplicates(subset=['subject_id'], keep='first')
        print(f"Got current sessions for {len(current_sessions_df)} subjects")

        # Add percentile category column with default value
        current_sessions_df['percentile_category'] = 'NS'  # Default to Not Scored
        # Add overall percentile column
        current_sessions_df['overall_percentile'] = float('nan')
        # Add threshold alert column with default value
        current_sessions_df['threshold_alert'] = 'N'  # Default to Normal
        # Add combined alert column
        current_sessions_df['combined_alert'] = 'NS'  # Default to Not Scored

        # Step 4: Apply pre-computed percentiles and alerts to the display data
        alert_count = 0
        for i, row in current_sessions_df.iterrows():
            subject_id = row['subject_id']
            stage = row.get('current_stage_actual')
            
            # Find this subject in overall percentiles
            subject_percentile = self.overall_percentiles[self.overall_percentiles['subject_id'] == subject_id]
            
            if not subject_percentile.empty:
                # Get the overall percentile value
                overall_percentile = subject_percentile['overall_percentile'].iloc[0]

                # Store percentile value in dataframe
                current_sessions_df.loc[i, 'overall_percentile'] = overall_percentile

                # Get strata and add to the dataframe
                if 'strata' in subject_percentile.columns:
                    strata = subject_percentile['strata'].iloc[0]
                    current_sessions_df.loc[i, 'strata'] = strata
                
                # Map to a category (percentile-based)
                percentile_category = self.alert_service.map_overall_percentile_to_category(overall_percentile)
                
                # Update dataframe with percentile category
                current_sessions_df.at[i, 'percentile_category'] = percentile_category
            
            # Apply stage-specific threshold alerts
            if stage in self.stage_thresholds:
                threshold = self.stage_thresholds[stage]
                if 'session' in row and row['session'] > threshold:
                    current_sessions_df.at[i, 'threshold_alert'] = 'T'
            
            # Apply total sessions threshold
            if 'session' in row and row['session'] > 40:
                current_sessions_df.at[i, 'threshold_alert'] = 'T'
            
            # Combine alerts
            percentile_category = current_sessions_df.at[i, 'percentile_category']
            threshold_alert = current_sessions_df.at[i, 'threshold_alert']
            
            if threshold_alert == 'T':
                if percentile_category != 'NS':
                    # Combine both alerts
                    current_sessions_df.at[i, 'combined_alert'] = f"{percentile_category}, T"
                else:
                    # Only threshold alert
                    current_sessions_df.at[i, 'combined_alert'] = 'T'
            else:
                # Only percentile alert
                current_sessions_df.at[i, 'combined_alert'] = percentile_category
            
            # Count non-normal alerts
            if current_sessions_df.at[i, 'combined_alert'] not in ['N', 'NS']:
                alert_count += 1

        print(f"Total alerts found: {alert_count} out of {len(current_sessions_df)} subjects")

        # Define column order 
        column_order = [
            'subject_id',
            'combined_alert',  # Change from percentile_category to combined_alert
            'percentile_category',
            'threshold_alert',
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
            'combined_alert': 'Alert',  # Update label for combined alert column
            'percentile_category': 'Percentile Alert', 
            'threshold_alert': 'Threshold Alert',
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
            # Row-level highlighting with lighter colors - update to use combined_alert
            {
                'if': {'filter_query': '{combined_alert} contains "SB" || {combined_alert} eq "T"'},
                'backgroundColor': '#FFC380',  # Light orange
                'className': 'alert-sb-row'
            },
            {
                'if': {'filter_query': '{combined_alert} contains "B" && !({combined_alert} contains "SB")'},
                'backgroundColor': '#FFC380',  # Very light orange
                'className': 'alert-b-row'
            },
            {
                'if': {'filter_query': '{combined_alert} contains "G" && !({combined_alert} contains "SG")'},
                'backgroundColor': '#9FC5E8',  # Light blue
                'className': 'alert-g-row'
            },
            {
                'if': {'filter_query': '{combined_alert} contains "SG"'},
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
            self.tooltip_controller.get_tooltip_container(),

            dash_table.DataTable(
                id='session-table',
                data=formatted_data.to_dict('records'),
                columns=columns,
                page_size=20,
                fixed_rows={'headers': True},
                style_data_conditional=conditional_styles + [
                    {
                        'if': {'column_id': 'subject_id'},
                        'cursor': 'pointer'
                    }
                ],
                tooltip_delay=0,
                tooltip_duration=None,
                style_table={
                    'overflowY': 'auto',
                    'overflowX': 'auto',
                    'backgroundColor': 'white',
                    'height': 'calc(100vh - 300px)',  # Adjusted height to fill available space
                    'minHeight': '500px',
                    'width': '100%'
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
                }
            )
        ], className="data-table-container", style={'width': '100%'})