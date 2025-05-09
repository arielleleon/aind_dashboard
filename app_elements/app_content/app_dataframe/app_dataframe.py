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
            min_sessions=1,     # Minimal requirements since we just want the window
            min_days=1
        )

    def get_abbreviated_strata(self, strata_name):
        """
        Convert full strata name to abbreviated forms for datatable
        
        Parameters:
            strata_name (str): The full strata name (e.g., "Uncoupled Baiting_ADVANCED_v3")
            
        Returns:
            str: The abbreviated strata (e.g., "UBA3")
        """
        # Return empty string if no strata found
        if not strata_name:
            return ''
        
        # Hard coded mappings for common terms
        strata_mappings = {
            'Uncoupled Baiting': 'UB',
            'Coupled Baiting': 'CB',
            'Uncoupled Without Baiting': 'UWB',
            'Coupled Without Baiting': 'CWB',

            'BEGINNER': 'B',
            'INTERMEDIATE': 'I',
            'ADVANCED': 'A',

            'v1': '1',
            'v2': '2',
            'v3': '3'
        }
        
        # Split the strata name
        parts = strata_name.split('_')
        
        # Handle different strata formats
        if len(parts) >= 3:
            # Format: curriculum_Stage_Version (e.g., "Uncoupled Baiting_ADVANCED_v3")
            curriculum = '_'.join(parts[:-2])
            stage = parts[-2]
            version = parts[-1]

            # Get abbreviations from mappings
            curriculum_abbr = strata_mappings.get(curriculum, curriculum[:2].upper())
            stage_abbr = strata_mappings.get(stage, stage[0])
            version_abbr = strata_mappings.get(version, version[-1])
            
            # Combine abbreviations
            return f"{curriculum_abbr}{stage_abbr}{version_abbr}"
        else:
            return strata_name.replace(" ", "")

    def format_dataframe(self, df: pd.DataFrame, reference_date: datetime = None) -> pd.DataFrame:
        """
        Format the dataframe for display in the table with enhanced feature-specific data
        
        1. Pre-compute all-time percentiles once
        2. Get most recent sessions for each subject
        3. Apply unified alerts (percentile and threshold)
        4. Add feature-specific percentiles and alert categories
        """
        # Copy input data to avoid modifying the original
        original_df = df.copy() if df is not None else pd.DataFrame()
        
        # Add defensive check with detailed error message
        if original_df.empty or 'session_date' not in original_df.columns:
            print(f"ERROR: Invalid dataframe passed to format_dataframe.")
            print(f"DataFrame is {'empty' if original_df.empty else 'missing session_date column'}")
            if not original_df.empty:
                print(f"Available columns: {original_df.columns.tolist()}")
            
            # Return empty dataframe with required columns to avoid breaking the UI
            return pd.DataFrame(columns=['subject_id', 'combined_alert', 'percentile_category', 
                                        'overall_percentile', 'session_date', 'session'])
        
        print(f" Starting data formatting with {len(original_df)} sessions")
        
        # Create a boolean mask for off-curriculum sessions
        off_curriculum_mask = (
            original_df['curriculum_name'].isna() | 
            (original_df['curriculum_name'] == "None") |
            original_df['current_stage_actual'].isna() | 
            (original_df['current_stage_actual'] == "None") |
            original_df['curriculum_version'].isna() |
            (original_df['curriculum_version'] == "None")
        )
        
        # Count and track off-curriculum subjects for display and debugging
        off_curriculum_df = original_df[off_curriculum_mask]
        off_curriculum_count = len(off_curriculum_df)
        print(f"Display: Found {off_curriculum_count} off-curriculum sessions ({off_curriculum_count/len(original_df):.1%} of total)")
        
        # Track subjects with off-curriculum sessions
        off_curriculum_subjects = {}
        for subject_id in off_curriculum_df['subject_id'].unique():
            subject_sessions = off_curriculum_df[off_curriculum_df['subject_id'] == subject_id]
            off_curriculum_subjects[subject_id] = {
                'count': len(subject_sessions),
                'latest_date': subject_sessions['session_date'].max()
            }
        print(f"Display: Identified {len(off_curriculum_subjects)} subjects with off-curriculum sessions")
        
        # Step 1: Use cached percentiles if available from app_utils
        app_utils = self.app_utils
        
        # Check if app_utils has cached overall percentiles
        if hasattr(app_utils, '_cache') and app_utils._cache['overall_percentiles'] is not None:
            print("Using cached overall percentiles from app_utils")
            self.overall_percentiles = app_utils._cache['overall_percentiles']
            self.alert_service = app_utils.alert_service
            self.threshold_analyzer = app_utils.threshold_analyzer
        # If not cached in app_utils, compute them if not already done locally
        elif not hasattr(self, 'overall_percentiles'):
            print("Computing all-time percentiles...")
            
            # Initialize reference processor for all-time percentile calculation
            reference_processor = app_utils.initialize_reference_processor(
                features_config=self.features_config,
                min_sessions=1,  # Minimum sessions requirement for eligibility
                min_days=1       # Minimum days requirement for eligibility
            )
            
            # Process data through reference pipeline to create strata
            stratified_data = app_utils.process_reference_data(
                df=original_df, 
                reference_date=reference_date,
                remove_outliers=False
            )
            print(f"Created {len(stratified_data)} strata")
            
            # Initialize quantile analyzer with strata
            quantile_analyzer = app_utils.initialize_quantile_analyzer(stratified_data)
            print(f"Initialized quantile analyzer with {len(stratified_data)} strata")
            for strata, df in stratified_data.items():
                print(f"  Strata '{strata}': {len(df)} subjects (combined current + historical)")
            
            # Initialize alert service
            alert_service = app_utils.initialize_alert_service()
            print(f"Initialized alert service")
            
            # Initialize threshold analyzer
            threshold_analyzer = app_utils.initialize_threshold_analyzer()
            threshold_analyzer.set_threshold_config(self.threshold_config)
            print(f"Initialized threshold analyzer")
            
            # Calculate overall percentiles for all subjects (one-time calculation using simple average)
            self.overall_percentiles = quantile_analyzer.calculate_overall_percentile()
            self.alert_service = alert_service
            self.threshold_analyzer = threshold_analyzer
            self.app_utils = app_utils  # Store for later use
            print(f"Calculated overall percentiles for {len(self.overall_percentiles)} subjects")
            ns_count = sum(1 for _, row in self.overall_percentiles.iterrows() if pd.isna(row.get('overall_percentile')))
            print(f"  Not Scored subjects: {ns_count} ({ns_count/len(self.overall_percentiles)*100:.1f}%)")
        
        # Step 2: Get most current session for each subject
        original_df = original_df.sort_values('session_date', ascending=False)
        current_sessions_df = original_df.drop_duplicates(subset=['subject_id'], keep='first').copy()
        print(f"Got current sessions for {len(current_sessions_df)} subjects")

        # Before proceeding, mark subjects with off-curriculum sessions for special handling
        off_curriculum_subjects = set()
        if hasattr(self.app_utils, 'off_curriculum_subjects'):
            off_curriculum_subjects = set(self.app_utils.off_curriculum_subjects.keys())
            print(f"Format dataframe: Found {len(off_curriculum_subjects)} subjects with off-curriculum sessions")
            
            # Check some sample subjects
            off_subjects_in_display = set(current_sessions_df['subject_id']) & off_curriculum_subjects
            print(f"Off-curriculum subjects in current display: {len(off_subjects_in_display)}")
            
            if off_subjects_in_display:
                sample_size = min(5, len(off_subjects_in_display))
                sample_subjects = list(off_subjects_in_display)[:sample_size]
                print("Sample off-curriculum subjects:")
                for subject_id in sample_subjects:
                    info = self.app_utils.off_curriculum_subjects[subject_id]
                    print(f"  Subject {subject_id}: {info['count']} of {info['total_sessions']} sessions")

        # Initialize columns with default values
        feature_list = list(self.features_config.keys())

        # Base alert columns - using .loc to avoid SettingWithCopyWarning
        current_sessions_df.loc[:, 'percentile_category'] = 'NS'  # Default to Not Scored
        current_sessions_df.loc[:, 'overall_percentile'] = float('nan')
        current_sessions_df.loc[:, 'threshold_alert'] = 'N'  # Default to Normal
        current_sessions_df.loc[:, 'combined_alert'] = 'NS'  # Default to Not Scored
        current_sessions_df.loc[:, 'strata_abbr'] = ''  # Default to empty string
        current_sessions_df.loc[:, 'ns_reason'] = ''  # Initialize NS reason column
        
        # Set NS reason for off-curriculum subjects immediately
        for subject_id in off_curriculum_subjects:
            mask = current_sessions_df['subject_id'] == subject_id
            if mask.any():
                info = self.app_utils.off_curriculum_subjects[subject_id]
                percent = (info['count'] / info['total_sessions']) * 100
                current_sessions_df.loc[mask, 'ns_reason'] = f"Off-curriculum sessions ({info['count']}, {percent:.0f}% of total)"
        
        # Threshold alert columns
        current_sessions_df.loc[:, 'total_sessions_alert'] = 'N'
        current_sessions_df.loc[:, 'stage_sessions_alert'] = 'N'
        current_sessions_df.loc[:, 'water_day_total_alert'] = 'N'
        
        # Feature-specific columns
        for feature in feature_list:
            current_sessions_df.loc[:, f'{feature}_percentile'] = float('nan')
            current_sessions_df.loc[:, f'{feature}_category'] = 'NS'
        
        # Step 3: Add strata information from the most recent sessions
        print(f"Adding strata information to current sessions")

        # Create a reference processor just for strata assignment
        temp_processor = ReferenceProcessor(
            features_config={},  # Empty dict since we're just using it for strata
            min_sessions=1,
            min_days=1
        )

        # Assign strata to the current sessions
        if not current_sessions_df.empty:
            # Pre-process to get curriculum_version_group
            preprocessed_df = temp_processor.preprocess_data(current_sessions_df, remove_outliers=False)
            print(f"Reference processor preprocess: Removing {len(current_sessions_df) - len(preprocessed_df)} off-curriculum sessions")
            
            # Assign strata 
            with_strata = temp_processor.assign_subject_strata(preprocessed_df)
            
            # Add strata columns to current_sessions_df
            if 'strata' in with_strata.columns:
                subjects_with_strata = 0
                
                # Create a mapping from subject_id to strata
                strata_map = with_strata.set_index('subject_id')['strata'].to_dict()
                
                # Apply strata to current_sessions_df using loc
                for subject_id, strata in strata_map.items():
                    mask = current_sessions_df['subject_id'] == subject_id
                    if mask.any():
                        # Add the strata
                        current_sessions_df.loc[mask, 'strata'] = strata
                        
                        # Add abbreviated strata using our local method
                        strata_abbr = self.get_abbreviated_strata(strata)
                        current_sessions_df.loc[mask, 'strata_abbr'] = strata_abbr
                        
                        subjects_with_strata += 1
                
                # Add extra debugging to check if strata_abbr exists and has values
                abbr_count = sum(1 for x in current_sessions_df['strata_abbr'] if x and not pd.isna(x))
                print(f"Added strata information to {subjects_with_strata} subjects (non-empty abbreviations: {abbr_count})")
                
                # Check for a few random subjects
                sample_subjects = current_sessions_df.sample(min(5, len(current_sessions_df)))
                print("Sample of subjects with their strata info:")
                for idx, row in sample_subjects.iterrows():
                    print(f"  Subject {row['subject_id']}: Strata={row.get('strata', 'N/A')}, Abbr={row.get('strata_abbr', 'N/A')}")

        # Step 4: Get unified alerts for all subjects in the current display
        subject_ids = current_sessions_df['subject_id'].unique().tolist()
        unified_alerts = self.app_utils.get_unified_alerts(subject_ids)
        print(f"Got unified alerts for {len(unified_alerts)} subjects")
        
        # Step 5: Apply unified alerts to the display data
        alert_count = 0
        
        # Process alerts in batch where possible
        for subject_id, alerts in unified_alerts.items():
            # Skip if subject is an off-curriculum subject
            if subject_id in off_curriculum_subjects:
                continue
                
            # Create mask for this subject
            mask = current_sessions_df['subject_id'] == subject_id
            if not mask.any():
                continue
                
            # Get alerts for this subject
            
            # Add overall percentile directly from unified alerts
            overall_percentile = alerts.get('overall_percentile')
            if overall_percentile is not None:
                current_sessions_df.loc[mask, 'overall_percentile'] = overall_percentile
            
            # Add alert category directly from unified alerts
            alert_category = alerts.get('alert_category', 'NS')
            current_sessions_df.loc[mask, 'percentile_category'] = alert_category
            
            # Add NS reason if applicable
            if alert_category == 'NS' and 'ns_reason' in alerts:
                current_sessions_df.loc[mask, 'ns_reason'] = alerts['ns_reason']
            
            # Add strata information if available and not already set
            if 'strata' in alerts:
                strata_mask = mask & (current_sessions_df['strata'].isna() | (current_sessions_df['strata'] == ''))
                if strata_mask.any():
                    alert_strata = alerts['strata']
                    current_sessions_df.loc[strata_mask, 'strata'] = alert_strata
                    
                    # Add abbreviated strata using our local method
                    strata_abbr = self.get_abbreviated_strata(alert_strata)
                    current_sessions_df.loc[strata_mask, 'strata_abbr'] = strata_abbr
            
            # Apply threshold alerts
            threshold_data = alerts.get('threshold', {})
            specific_alerts = threshold_data.get('specific_alerts', {})
            
            # Total sessions threshold
            total_sessions_alert_info = specific_alerts.get('total_sessions', {})
            total_sessions_alert = total_sessions_alert_info.get('alert', 'N')
            if total_sessions_alert == 'T':
                total_sessions_value = total_sessions_alert_info.get('value', '')
                current_sessions_df.loc[mask, 'total_sessions_alert'] = f"T | {total_sessions_value}"
            
            # Stage sessions threshold
            stage_alert_info = specific_alerts.get('stage_sessions', {})
            if stage_alert_info.get('alert') == 'T':
                stage_name = stage_alert_info.get('stage', current_sessions_df.loc[mask, 'current_stage_actual'].iloc[0])
                stage_value = stage_alert_info.get('value', '')
                current_sessions_df.loc[mask, 'stage_sessions_alert'] = f"T | {stage_name} | {stage_value}"
            
            # Water day total threshold
            water_alert_info = specific_alerts.get('water_day_total', {})
            if not isinstance(water_alert_info, dict):
                water_alert_info = {'alert': 'N', 'value': water_alert_info}
            water_day_total_alert = water_alert_info.get('alert', 'N')
            if water_day_total_alert == 'T':
                water_value = water_alert_info.get('value', '')
                current_sessions_df.loc[mask, 'water_day_total_alert'] = f"T | {water_value:.1f}"
            
            # Process feature-specific percentiles and categories
            feature_percentiles = alerts.get('feature_percentiles', {})
            for feature, details in feature_percentiles.items():
                if feature in feature_list:
                    # Add percentile
                    current_sessions_df.loc[mask, f'{feature}_percentile'] = details.get('percentile')
                    # Add category
                    current_sessions_df.loc[mask, f'{feature}_category'] = details.get('category', 'NS')
            
            # Combine alerts - simplify logic
            if threshold_data.get('threshold_alert', 'N') == 'T':
                if alert_category != 'NS':
                    current_sessions_df.loc[mask, 'combined_alert'] = f"{alert_category}, T"
                else:
                    current_sessions_df.loc[mask, 'combined_alert'] = 'T'
            else:
                current_sessions_df.loc[mask, 'combined_alert'] = alert_category
            
            # Count non-normal alerts
            if current_sessions_df.loc[mask, 'combined_alert'].iloc[0] not in ['N', 'NS']:
                alert_count += 1

        print(f"Total alerts found: {alert_count} out of {len(current_sessions_df)} subjects")

        # Define column order with new feature-specific columns
        base_columns = [
            'subject_id',
            'combined_alert',
            'percentile_category',
            'ns_reason',
            'threshold_alert',
            'total_sessions_alert',
            'stage_sessions_alert',
            'water_day_total_alert',
            'overall_percentile',
            'strata',
            'strata_abbr',
            'current_stage_actual',
            'curriculum',
            'session_date',
            'session',
            'rig',
            'trainer',
            'PI',
        ]
        
        # Add feature-specific columns
        feature_columns = []
        for feature in feature_list:
            feature_columns.append(f'{feature}_percentile')
            feature_columns.append(f'{feature}_category')
        
        # Add remaining standard columns
        remaining_columns = [
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
            'session_run_time',
            'total_trials',
            'finished_trials',
            'finished_rate',
            'ignore_rate',
        ]
        
        # Combine all column groups
        column_order = base_columns + feature_columns + remaining_columns

        # Filter columns to include only those in the defined order
        available_columns = [col for col in column_order if col in current_sessions_df.columns]

        # Add any remaining columns at the end
        extra_columns = [col for col in current_sessions_df.columns if col not in column_order]
        ordered_columns = available_columns + extra_columns

        # Reorder the columns
        formatted_df = current_sessions_df[ordered_columns]

        print(f" Formatted dataframe with {len(formatted_df)} rows and {len(formatted_df.columns)} columns")

        return formatted_df
    
    def build(self):
        """
        Build data table component with enhanced feature-specific columns
        """
        # Get the data and apply formatting
        raw_data = self.data_loader.get_data()
        formatted_data = self.format_dataframe(raw_data)
        
        # Identify float columns for formatting
        float_columns = [col for col in formatted_data.columns if formatted_data[col].dtype == 'float64']
        
        # Improve column header display
        formatted_column_names = {
            'subject_id': 'Subject ID',
            'combined_alert': 'Alert',
            'percentile_category': 'Percentile Alert', 
            'ns_reason': 'Not Scored Reason',
            'threshold_alert': 'Threshold Alert',
            'total_sessions_alert': 'Total Sessions Alert',
            'stage_sessions_alert': 'Stage Sessions Alert',
            'water_day_total_alert': 'Water Day Total Alert',
            'overall_percentile': 'Percentile',
            'strata': 'Strata',
            'strata_abbr': 'Strata (Abbr)',
            'current_stage_actual': 'Stage',
            'curriculum': 'curriculum',
            'session_date': 'Date',
            'session': 'Session',
            'rig': 'Rig',
            'trainer': 'Trainer',
            'PI': 'PI',
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
        
        # Add feature-specific column names
        for feature in self.features_config.keys():
            feature_display = feature.replace('_', ' ').replace('abs(', '|').replace(')', '|').title()
            formatted_column_names[f'{feature}_percentile'] = f'{feature_display}\nPercentile'
            formatted_column_names[f'{feature}_category'] = f'{feature_display}\nAlert'

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

        # Build the table with updated styling
        return html.Div([

            dash_table.DataTable(
                id='session-table',
                data=formatted_data.to_dict('records'),
                columns=columns,
                page_size=20,
                fixed_rows={'headers': True},
                style_data_conditional=[
                    {
                        'if': {'column_id': 'subject_id'},
                        'cursor': 'pointer'
                    }
                ],
                style_table={
                    'overflowY': 'auto',
                    'overflowX': 'auto',
                    'backgroundColor': 'white',
                    'height': 'calc(100vh - 350px)',
                    'minHeight': '400px',
                    'width': '100%',
                    'marginBottom': '0px'
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
                cell_selectable=True,
                row_selectable=False
            )
        ], className="data-table-container", style={'width': '100%'})