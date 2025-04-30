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
            window_days=30,     # Default window of 30 days
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

    def format_dataframe(self, df: pd.DataFrame, window_days: int = 30, reference_date: datetime = None) -> pd.DataFrame:
        """
        Format the dataframe for display in the table with enhanced feature-specific data
        
        1. Pre-compute all-time percentiles once
        2. Filter data for display based on window_days
        3. Get most recent sessions for each subject in the display window
        4. Apply unified alerts (percentile and threshold)
        5. Add feature-specific percentiles and alert categories
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
        
        print(f" Starting data formatting with {len(original_df)} sessions, {window_days} day window")
        
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
            all_time_df = reference_processor.apply_sliding_window(original_df, reference_date)
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
            
            # Calculate overall percentiles for all subjects (one-time calculation)
            self.overall_percentiles = quantile_analyzer.calculate_overall_percentile()
            self.alert_service = alert_service
            self.threshold_analyzer = threshold_analyzer
            self.app_utils = app_utils  # Store for later use
            print(f"Calculated overall percentiles for {len(self.overall_percentiles)} subjects")
            ns_count = sum(1 for _, row in self.overall_percentiles.iterrows() if pd.isna(row.get('overall_percentile')))
            print(f"  Not Scored subjects: {ns_count} ({ns_count/len(self.overall_percentiles)*100:.1f}%)")
        
        # Step 2: Apply sliding window filter for DISPLAY only
        if not hasattr(self, 'display_processor'):
            self.display_processor = ReferenceProcessor(
                features_config={},  # Empty dict since we're only using it for filtering
                window_days=window_days,
                min_sessions=1,
                min_days=1
            )
        else:
            self.display_processor.window_days = window_days
        
        window_df = self.display_processor.apply_sliding_window(original_df, reference_date)
        print(f"Applied display window: {len(window_df)} sessions")

        # Step 3: Get most current session for each subject in the window
        window_df = window_df.sort_values('session_date', ascending=False)
        current_sessions_df = window_df.drop_duplicates(subset=['subject_id'], keep='first')
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

        # Base alert columns
        current_sessions_df['percentile_category'] = 'NS'  # Default to Not Scored
        current_sessions_df['overall_percentile'] = float('nan')
        current_sessions_df['threshold_alert'] = 'N'  # Default to Normal
        current_sessions_df['combined_alert'] = 'NS'  # Default to Not Scored
        current_sessions_df['strata_abbr'] = ''  # Default to empty string
        current_sessions_df['ns_reason'] = ''  # Initialize NS reason column
        
        # Set NS reason for off-curriculum subjects immediately
        for i, row in current_sessions_df.iterrows():
            subject_id = row['subject_id']
            if subject_id in off_curriculum_subjects:
                info = self.app_utils.off_curriculum_subjects[subject_id]
                percent = (info['count'] / info['total_sessions']) * 100
                current_sessions_df.at[i, 'ns_reason'] = f"Off-curriculum sessions ({info['count']}, {percent:.0f}% of total)"
        
        # Threshold alert columns
        current_sessions_df['total_sessions_alert'] = 'N'
        current_sessions_df['stage_sessions_alert'] = 'N'
        current_sessions_df['water_day_total_alert'] = 'N'
        
        # Feature-specific columns
        for feature in feature_list:
            current_sessions_df[f'{feature}_percentile'] = float('nan')
            current_sessions_df[f'{feature}_category'] = 'NS'
        
        # Step 3.5: Add strata information from the most recent sessions
        print(f"Adding strata information to current sessions")

        # Create a reference processor just for strata assignment
        temp_processor = ReferenceProcessor(
            features_config={},  # Empty dict since we're just using it for strata
            window_days=window_days,
            min_sessions=1,
            min_days=1
        )

        # Assign strata to the current sessions
        if not current_sessions_df.empty:
            # Pre-process to get curriculum_version_group
            preprocessed_df = temp_processor.preprocess_data(current_sessions_df, remove_outliers=False)
            
            # Assign strata 
            with_strata = temp_processor.assign_subject_strata(preprocessed_df)
            
            # Add strata columns to current_sessions_df
            if 'strata' in with_strata.columns:
                subjects_with_strata = 0
                
                for i, row in with_strata.iterrows():
                    subject_id = row['subject_id']
                    # Find matching row in current_sessions_df
                    idx = current_sessions_df[current_sessions_df['subject_id'] == subject_id].index
                    if not idx.empty:
                        # Add the strata
                        strata = row['strata']
                        current_sessions_df.at[idx[0], 'strata'] = strata
                        
                        # Generate and add the abbreviated strata directly
                        strata_abbr = self.get_abbreviated_strata(strata)
                        current_sessions_df.at[idx[0], 'strata_abbr'] = strata_abbr
                        
                        # Debug specific subjects
                        if subject_id == '779531':  # Add your problem subject ID here
                            print(f"DEBUG - Subject {subject_id}:")
                            print(f"  Strata: {strata}")
                            print(f"  Abbreviated strata: {strata_abbr}")
                        
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
        for i, row in current_sessions_df.iterrows():
            subject_id = row['subject_id']
            
            # Skip if subject has no alerts or is an off-curriculum subject
            if subject_id not in unified_alerts or subject_id in off_curriculum_subjects:
                # For off-curriculum subjects, ensure they are marked as "NS"
                if subject_id in off_curriculum_subjects:
                    current_sessions_df.at[i, 'percentile_category'] = 'NS'
                    current_sessions_df.at[i, 'combined_alert'] = 'NS'
                    # NS reason is already set above
                continue
            
            # Get alerts for this subject
            alerts = unified_alerts[subject_id]
            
            # Add overall percentile directly from unified alerts
            overall_percentile = alerts.get('overall_percentile')
            if overall_percentile is not None:
                current_sessions_df.loc[i, 'overall_percentile'] = overall_percentile
            
            # Add alert category directly from unified alerts
            alert_category = alerts.get('alert_category', 'NS')
            current_sessions_df.at[i, 'percentile_category'] = alert_category
            
            # Add NS reason if applicable
            if alert_category == 'NS' and 'ns_reason' in alerts:
                current_sessions_df.at[i, 'ns_reason'] = alerts['ns_reason']
            
            # Add strata information if available and not already set
            if 'strata' in alerts and ('strata' not in current_sessions_df.columns or pd.isna(current_sessions_df.at[i, 'strata'])):
                alert_strata = alerts['strata']
                current_sessions_df.at[i, 'strata'] = alert_strata
                
                # Add abbreviated strata using our local method
                strata_abbr = self.get_abbreviated_strata(alert_strata)
                current_sessions_df.at[i, 'strata_abbr'] = strata_abbr
                
                # Debug for specific subjects
                if subject_id == '779531':  # Add your problem subject ID here
                    print(f"DEBUG - Set from alerts - Subject {subject_id}:")
                    print(f"  Alert strata: {alert_strata}")
                    print(f"  Generated abbr: {strata_abbr}")
            
            # Apply threshold alerts
            threshold_data = alerts.get('threshold', {})
            
            # Overall threshold alert
            threshold_alert = threshold_data.get('threshold_alert', 'N')
            current_sessions_df.at[i, 'threshold_alert'] = threshold_alert
            
            # Get specific threshold alerts
            specific_alerts = threshold_data.get('specific_alerts', {})
            
            # Total sessions threshold
            total_sessions_alert = specific_alerts.get('total_sessions', {}).get('alert', 'N')
            current_sessions_df.at[i, 'total_sessions_alert'] = total_sessions_alert
            
            # Stage sessions threshold - include stage name when threshold is exceeded
            stage_alert_info = specific_alerts.get('stage_sessions', {})
            if stage_alert_info.get('alert') == 'T':
                # Include stage name with the alert
                stage_name = stage_alert_info.get('stage', row.get('current_stage_actual', ''))
                # Format as "T | STAGE_NAME"
                current_sessions_df.at[i, 'stage_sessions_alert'] = f"T | {stage_name}"
            else:
                current_sessions_df.at[i, 'stage_sessions_alert'] = 'N'
            
            # Water day total threshold
            water_day_total_alert = specific_alerts.get('water_day_total', {}).get('alert', 'N')
            current_sessions_df.at[i, 'water_day_total_alert'] = water_day_total_alert
            
            # Process feature-specific percentiles and categories
            feature_percentiles = alerts.get('feature_percentiles', {})
            for feature, details in feature_percentiles.items():
                if feature in feature_list:
                    # Add percentile
                    current_sessions_df.at[i, f'{feature}_percentile'] = details.get('percentile')
                    # Add category
                    current_sessions_df.at[i, f'{feature}_category'] = details.get('category', 'NS')
            
            # Combine alerts - simplify logic
            if threshold_alert == 'T':
                if alert_category != 'NS':
                    current_sessions_df.at[i, 'combined_alert'] = f"{alert_category}, T"
                else:
                    current_sessions_df.at[i, 'combined_alert'] = 'T'
            else:
                current_sessions_df.at[i, 'combined_alert'] = alert_category
            
            # Count non-normal alerts
            if current_sessions_df.at[i, 'combined_alert'] not in ['N', 'NS']:
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

        # Before returning the formatted dataframe, add this:
        # Check if strata_abbr has values
        if 'strata_abbr' in formatted_df.columns:
            non_empty = sum(1 for x in formatted_df['strata_abbr'] if x and not pd.isna(x))
            print(f"Final check: strata_abbr column has {non_empty} non-empty values out of {len(formatted_df)} rows")
            
            # Check a few specific rows
            print("Final strata values for first 5 rows:")
            for i, row in formatted_df.head().iterrows():
                print(f"  Subject {row['subject_id']}: Strata={row.get('strata', 'N/A')}, Abbr={row.get('strata_abbr', 'N/A')}")

        return formatted_df
    
    def build(self):
        """
        Build data table component with enhanced feature-specific columns
        """
        # Add debugging to check data loading
        print("Starting build() method in AppDataFrame")
        
        # Get the data and apply formatting
        raw_data = self.data_loader.get_data()
        
        # Debug raw data
        print(f"Raw data from data_loader has shape: {raw_data.shape}")
        print(f"Raw data columns include 'session_date'? {'session_date' in raw_data.columns}")
        
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

        # Add styling for specific threshold alert columns
        conditional_styles.extend([
            {
                'if': {'column_id': 'total_sessions_alert', 'filter_query': '{total_sessions_alert} eq "T"'},
                'backgroundColor': '#FF8C40',  # Orange for alert
                'color': 'white',
                'fontWeight': 'bold'
            },
            {
                'if': {'column_id': 'water_day_total_alert', 'filter_query': '{water_day_total_alert} eq "T"'},
                'backgroundColor': '#FF8C40',  # Orange for alert
                'color': 'white',
                'fontWeight': 'bold'
            },
            # Updated style for stage sessions alert that checks if it contains "T |"
            {
                'if': {'column_id': 'stage_sessions_alert', 'filter_query': '{stage_sessions_alert} contains "T |"'},
                'backgroundColor': '#FF8C40',  # Orange for alert
                'color': 'white',
                'fontWeight': 'bold'
            }
        ])

        # Add feature-specific cell styling for alert categories
        for feature in self.features_config.keys():
            cat_col = f'{feature}_category'
            
            # Add styling for different alert categories
            conditional_styles.extend([
                {
                    'if': {'column_id': cat_col, 'filter_query': f'{{{cat_col}}} eq "SB"'},
                    'backgroundColor': '#FF8C40',  # Darker orange for SB
                    'color': 'white'
                },
                {
                    'if': {'column_id': cat_col, 'filter_query': f'{{{cat_col}}} eq "B"'},
                    'backgroundColor': '#FFCCA0',  # Light orange for B
                    'color': 'black'
                },
                {
                    'if': {'column_id': cat_col, 'filter_query': f'{{{cat_col}}} eq "G"'},
                    'backgroundColor': '#A0C5E8',  # Light blue for G
                    'color': 'black'
                },
                {
                    'if': {'column_id': cat_col, 'filter_query': f'{{{cat_col}}} eq "SG"'},
                    'backgroundColor': '#4D94DA',  # Darker blue for SG
                    'color': 'white'
                }
            ])
            
            # Add highlight for percentile columns
            percentile_col = f'{feature}_percentile'
            conditional_styles.append({
                'if': {'column_id': percentile_col},
                'border': '1px solid #5D9FD3',
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
                style_data_conditional=conditional_styles + [
                    {
                        'if': {'column_id': 'subject_id'},
                        'cursor': 'pointer'
                    }
                ],
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