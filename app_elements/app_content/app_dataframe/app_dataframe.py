from dash import html, dash_table
import pandas as pd
from datetime import datetime, timedelta
import traceback
from app_utils import AppUtils
from app_utils.app_analysis.reference_processor import ReferenceProcessor
from app_utils.app_analysis.overall_percentile_calculator import OverallPercentileCalculator
from dash import callback_context, Input, Output, clientside_callback
from typing import Dict, Any
from app_utils.app_analysis.threshold_analyzer import ThresholdAnalyzer

class AppDataFrame:
    def __init__(self, app_utils=None):
        # CRITICAL FIX: Allow injection of shared app_utils instance
        if app_utils is not None:
            self.app_utils = app_utils
        else:
            # Initialize app utilities (fallback for backward compatibility)
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
        
        # Create percentile calculator
        self.percentile_calculator = OverallPercentileCalculator()

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
        Format the dataframe for display in the table with unified session-level metrics
        
        1. Use already processed UI-optimized data if available
        2. Apply unified alerts
        3. Add feature-specific percentiles and metadata
        """
        # Defensive copy of input data
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
        
        print(f"📊 Starting data formatting with {len(original_df)} sessions")
        
        # Check if we have a cached formatted result or can use UI-optimized data
        app_utils = self.app_utils
        
        # CRITICAL FIX: First try to use UI-optimized table display data (fastest path)
        table_data = app_utils.get_table_display_data(use_cache=True)
        if table_data:
            print("✅ Using UI-optimized table display data (no pipeline re-run needed)")
            recent_sessions = pd.DataFrame(table_data)
            print(f"Loaded {len(recent_sessions)} subjects from UI cache")
        elif (app_utils._cache.get('session_level_data') is not None):
            print("🔄 Using cached session-level data to get most recent sessions")
            recent_sessions = app_utils.get_most_recent_subject_sessions(use_cache=True)
            print(f"Selected most recent session for {len(recent_sessions)} subjects")
        else:
            print("⚠️  No cached data available - this should not happen after app initialization")
            print("🔄 Unified pipeline not run yet - processing now to ensure data availability...")
            app_utils.process_data_pipeline(original_df, use_cache=False)
            print("✅ Unified pipeline complete - UI structures now available")
            
            # Now get the UI-optimized data
            table_data = app_utils.get_table_display_data(use_cache=True)
            if table_data:
                recent_sessions = pd.DataFrame(table_data)
                print(f"Loaded {len(recent_sessions)} subjects from UI cache")
            else:
                # Final fallback
                recent_sessions = app_utils.get_most_recent_subject_sessions(use_cache=True)
                print(f"Selected most recent session for {len(recent_sessions)} subjects")
        
        # Step 3: Apply alerts and prepare output
        # Initialize alert service if needed
        if app_utils.alert_service is None:
            app_utils.initialize_alert_service()
        
        # Get all subject IDs
        subject_ids = recent_sessions['subject_id'].unique().tolist()
        
        # Get unified alerts for these subjects  
        unified_alerts = app_utils.get_unified_alerts(subject_ids)
        print(f"Got unified alerts for {len(unified_alerts)} subjects")
        
        # Step 4: Apply alerts and format output dataframe
        # Start with the most recent sessions and add alert columns
        output_df = recent_sessions.copy()
        
        # DEBUG: Show what columns are available from the unified pipeline
        print(f"\nDEBUG: Columns available from unified pipeline ({len(output_df.columns)} total):")
        session_percentile_cols = [col for col in output_df.columns if col.endswith('_session_percentile')]
        rolling_avg_cols = [col for col in output_df.columns if col.endswith('_rolling_avg')]
        category_cols = [col for col in output_df.columns if col.endswith('_category')]
        overall_cols = [col for col in output_df.columns if 'overall_percentile' in col]
        
        print(f"  Session percentile columns ({len(session_percentile_cols)}): {session_percentile_cols}")
        print(f"  Rolling average columns ({len(rolling_avg_cols)}): {rolling_avg_cols}")
        print(f"  Category columns ({len(category_cols)}): {category_cols}")
        print(f"  Overall percentile columns ({len(overall_cols)}): {overall_cols}")
        
        # Check if we have sample data
        if len(output_df) > 0:
            print(f"  Sample data for first subject ({output_df.iloc[0]['subject_id']}):")
            sample_cols = session_percentile_cols[:2] + ['overall_percentile'] if session_percentile_cols else ['overall_percentile']
            for col in sample_cols:
                if col in output_df.columns:
                    value = output_df.iloc[0][col]
                    print(f"    {col}: {value}")
        print("")  # Empty line for readability
        
        # Initialize alert columns if not already present (for UI cache compatibility)
        for col in ['percentile_category', 'threshold_alert', 'combined_alert', 'ns_reason', 'strata_abbr',
                   'total_sessions_alert', 'stage_sessions_alert', 'water_day_total_alert']:
            if col not in output_df.columns:
                default_val = 'NS' if col in ['percentile_category', 'combined_alert'] else ('N' if col.endswith('_alert') else '')
                output_df.loc[:, col] = default_val
        
        # THRESHOLD ALERTS: Initialize threshold analyzer and calculate alerts
        print("Calculating threshold alerts...")
        
        # Initialize threshold analyzer with configuration
        # Combine general thresholds with stage-specific thresholds
        combined_config = self.threshold_config.copy()
        for stage, threshold in self.stage_thresholds.items():
            combined_config[f"stage_{stage}_sessions"] = {
                'condition': 'gt',
                'value': threshold
            }
        
        threshold_analyzer = ThresholdAnalyzer(combined_config)
        
        # Calculate threshold alerts for each subject
        for idx, row in output_df.iterrows():
            subject_id = row['subject_id']
            
            # Get all sessions for this subject (needed for session count calculations)
            subject_sessions = app_utils.get_subject_sessions(subject_id)
            if subject_sessions is not None and not subject_sessions.empty:
                
                # 1. Check total sessions alert
                total_sessions_alert = threshold_analyzer.check_total_sessions(subject_sessions)
                output_df.loc[idx, 'total_sessions_alert'] = total_sessions_alert['display_format']
                
                # 2. Check stage-specific sessions alert
                current_stage = row.get('current_stage_actual')
                if current_stage and current_stage in self.stage_thresholds:
                    stage_sessions_alert = threshold_analyzer.check_stage_sessions(subject_sessions, current_stage)
                    output_df.loc[idx, 'stage_sessions_alert'] = stage_sessions_alert['display_format']
                
                # 3. Check water day total alert
                water_day_total = row.get('water_day_total')
                if not pd.isna(water_day_total):
                    water_alert = threshold_analyzer.check_water_day_total(water_day_total)
                    output_df.loc[idx, 'water_day_total_alert'] = water_alert['display_format']
        
        print(f"Threshold alerts calculated for {len(output_df)} subjects")
        
        # Apply abbreviations to strata names if not already present
        if 'strata_abbr' not in output_df.columns or output_df['strata_abbr'].isna().all():
            output_df.loc[:, 'strata_abbr'] = output_df['strata'].apply(self.get_abbreviated_strata)
        
        # Apply alerts from unified_alerts
        for subject_id, alerts in unified_alerts.items():
            # Create mask for this subject
            mask = output_df['subject_id'] == subject_id
            if not mask.any():
                continue
            
            # Add alert category
            alert_category = alerts.get('alert_category', 'NS')
            output_df.loc[mask, 'percentile_category'] = alert_category
            
            # Add NS reason if applicable
            if alert_category == 'NS' and 'ns_reason' in alerts:
                output_df.loc[mask, 'ns_reason'] = alerts['ns_reason']
            
            # Apply threshold alerts
            threshold_data = alerts.get('threshold', {})
            if threshold_data.get('threshold_alert', 'N') == 'T':
                output_df.loc[mask, 'threshold_alert'] = 'T'
                
                # Combine alerts
                if alert_category != 'NS':
                    output_df.loc[mask, 'combined_alert'] = f"{alert_category}, T"
                else:
                    output_df.loc[mask, 'combined_alert'] = 'T'
            else:
                output_df.loc[mask, 'combined_alert'] = alert_category
        
        print(f"Applied alerts to {len(output_df)} subjects")
        
        # Use a simple hash of the input data size and max date to check cache validity
        data_hash = f"{len(original_df)}_{original_df['session_date'].max()}"
        
        # Cache the formatted result
        app_utils._cache['formatted_data'] = output_df.copy()
        app_utils._cache['formatted_data_hash'] = data_hash
        
        # Return the formatted dataframe
        return output_df
    
    def build(self):
        """
        Build data table component with enhanced feature-specific columns
        and session-level metrics
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
            'overall_percentile': 'Overall\nPercentile',
            'session_overall_percentile': 'Session\nPercentile',
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
            # Strata metrics (legacy - may not be present in new pipeline)
            formatted_column_names[f'{feature}_percentile'] = f'{feature_display}\nStrata %ile'
            formatted_column_names[f'{feature}_category'] = f'{feature_display}\nAlert'
            formatted_column_names[f'{feature}_processed'] = f'{feature_display}\nProcessed'
            # NEW: Session-level metrics (from unified pipeline)
            formatted_column_names[f'{feature}_session_percentile'] = f'{feature_display}\nSession %ile'
            formatted_column_names[f'{feature}_processed_rolling_avg'] = f'{feature_display}\nRolling Avg'
            
        # Add overall percentile columns (both session and strata versions)
        formatted_column_names['session_overall_percentile'] = 'Session Overall\nPercentile'
        formatted_column_names['overall_percentile'] = 'Strata Overall\nPercentile'

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
        
        # DEBUG: Show column count and key session-level columns for verification
        session_cols = [col for col in formatted_data.columns if col.endswith('_session_percentile')]
        rolling_cols = [col for col in formatted_data.columns if col.endswith('_rolling_avg')]
        print(f"\nDataTable build - Total columns: {len(columns)}")
        print(f"  Session percentile columns: {len(session_cols)}")
        print(f"  Rolling average columns: {len(rolling_cols)}")
        if session_cols:
            print(f"  Example session columns: {session_cols[:3]}")
        if rolling_cols:
            print(f"  Example rolling columns: {rolling_cols[:2]}")
        print("")

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

    def validate_and_optimize_cache(self):
        """
        Validate the consistency of cached data and optimize memory usage
        """
        # Check if we have both session-level and strata-level data for validation
        if self._cache['session_level_data'] is not None and self._cache['stratified_data'] is not None:
            print("Validating session vs. strata consistency...")
            
            # Get comprehensive data from cached session-level data
            session_data = self._cache['session_level_data']
            
            # Get strata-level data from stratified_data
            strata_level_data = pd.DataFrame()
            for strata, df in self._cache['stratified_data'].items():
                strata_level_data = pd.concat([strata_level_data, df])
            
            # Perform validation
            validation_results = self.validate_session_vs_strata_percentiles(
                session_data=session_data,
                strata_data=strata_level_data
            )
            
            print(f"Validation results: {validation_results['match_percentage']:.2f}% match rate")
            print(f"Average difference: {validation_results.get('average_difference', 0):.4f}")
            
            # Optimize memory usage if validation passes
            if validation_results['match_percentage'] > 95:
                print("Validation successful, optimizing cache...")
                # We can safely discard strata-level data since session data is valid
                self._cache['stratified_data'] = None
                self._cache['strata_percentiles'] = None
                
                # Force garbage collection
                import gc
                gc.collect()
                print("Cache optimization complete")
        
        return True

    def get_subject_latest_metrics(self, subject_id: str) -> Dict[str, Any]:
        """
        Get the latest metrics for a specific subject using the unified session-level data
        
        Parameters:
            subject_id: str
                Subject ID to retrieve metrics for
                
        Returns:
            Dict[str, Any]: Dictionary with latest metrics
        """
        # Get all subject sessions with metrics
        subject_sessions = self.get_subject_session_level_percentiles(subject_id)
        
        if subject_sessions.empty:
            return {}
        
        # Sort by date (descending) and get most recent
        subject_sessions = subject_sessions.sort_values('session_date', ascending=False)
        latest_session = subject_sessions.iloc[0]
        
        # Extract key metrics
        metrics = {
            'subject_id': subject_id,
            'strata': latest_session.get('strata', 'Unknown'),
            'strata_abbr': self._get_abbreviated_strata(latest_session.get('strata', '')),
            'overall_percentile': latest_session.get('overall_percentile', float('nan')),
            'session_date': latest_session.get('session_date'),
            'session': latest_session.get('session'),
            'is_current_strata': latest_session.get('is_current_strata', True)
        }
        
        # Add feature-specific percentiles
        percentile_cols = [col for col in latest_session.index if col.endswith('_percentile')]
        for col in percentile_cols:
            feature_name = col.replace('_percentile', '')
            metrics[f'{feature_name}_percentile'] = latest_session[col]
        
        return metrics

    def _get_abbreviated_strata(self, strata: str) -> str:
        """
        Get abbreviated strata name
        
        Parameters:
            strata: str
                Full strata name
                
        Returns:
            str: Abbreviated strata name
        """
        # Split the strata name
        parts = strata.split('_')
        
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
        
        # Handle different strata formats
        if len(parts) >= 3:
            # Format: curriculum_Stage_Version
            curriculum = '_'.join(parts[:-2])
            stage = parts[-2]
            version = parts[-1]
            
            # Get abbreviations
            curriculum_abbr = strata_mappings.get(curriculum, curriculum[:2].upper())
            stage_abbr = strata_mappings.get(stage, stage[0])
            version_abbr = strata_mappings.get(version, version[-1])
            
            return f"{curriculum_abbr}{stage_abbr}{version_abbr}"
        
        return strata.replace(" ", "")

    def get_subject_session_history(self, subject_id: str) -> pd.DataFrame:
        """
        Get complete session history with metrics for a specific subject
        
        Parameters:
            subject_id: str
                Subject ID to retrieve history for
                
        Returns:
            pd.DataFrame: DataFrame with session history
        """
        # Load session-level data for this subject
        sessions = self.analyze_unified_percentiles(subject_ids=[subject_id], use_cache=True)
        
        if sessions.empty:
            return pd.DataFrame()
        
        # Add strata_transition_date for visualization
        sessions_with_transitions = sessions.copy()
        
        # Sort by date for sequential processing
        sessions_with_transitions = sessions_with_transitions.sort_values('session_date')
        
        # Add transition dates
        transition_dates = {}
        current_strata = None
        
        for _, row in sessions_with_transitions.iterrows():
            strata = row['strata']
            if current_strata is None or strata != current_strata:
                # This is a transition
                transition_dates[strata] = row['session_date']
                current_strata = strata
        
        # Add transition date to each row
        for strata, transition_date in transition_dates.items():
            mask = sessions_with_transitions['strata'] == strata
            sessions_with_transitions.loc[mask, 'strata_transition_date'] = transition_date
        
        return sessions_with_transitions