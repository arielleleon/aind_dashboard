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
from .column_groups_config import COLUMN_GROUPS, get_columns_for_groups, get_default_visible_columns
from app_utils.ui_utils import (
    get_optimized_table_data, 
    process_unified_alerts_integration, 
    format_strata_abbreviations,
    create_empty_dataframe_structure
)

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

    def get_filtered_columns(self, all_columns, visible_column_ids):
        """
        Filter the column definitions to only include visible columns
        
        Parameters:
            all_columns (list): List of all column definitions
            visible_column_ids (list): List of column IDs that should be visible
            
        Returns:
            list: Filtered list of column definitions
        """
        return [col for col in all_columns if col['id'] in visible_column_ids]

    def format_dataframe(self, df: pd.DataFrame, reference_date: datetime = None, visible_columns: list = None) -> pd.DataFrame:
        """
        Format the dataframe for display in the table with unified session-level metrics
        
        This method now uses extracted business logic functions to:
        1. Get optimized data with intelligent cache fallback
        2. Integrate unified alerts using business logic
        3. Apply strata abbreviation formatting
        4. Handle column filtering
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
            return create_empty_dataframe_structure()
        
        print(f"📊 Starting data formatting with {len(original_df)} sessions")
        
        # STEP 1: Get optimized data using business logic
        recent_sessions = get_optimized_table_data(self.app_utils, use_cache=True)
        
        # STEP 2: Process alerts using business logic
        recent_sessions = process_unified_alerts_integration(
            recent_sessions, 
            self.app_utils, 
            threshold_config=self.threshold_config,
            stage_thresholds=self.stage_thresholds
        )
        
        # STEP 3: Apply strata formatting using business logic
        recent_sessions = format_strata_abbreviations(recent_sessions)
        
        # STEP 4: Filter to visible columns if specified
        if visible_columns is not None:
            # Only keep columns that exist in the dataframe and are in visible_columns
            existing_visible_cols = [col for col in visible_columns if col in recent_sessions.columns]
            recent_sessions = recent_sessions[existing_visible_cols]
            print(f"Filtered to {len(existing_visible_cols)} visible columns")
        
        # Return the formatted dataframe
        return recent_sessions
    
    def _get_percentile_formatting_rules(self):
        """
        Generate conditional formatting rules for percentile-based values
        Uses computed alert categories from UI cache to match tooltip coloring
        """
        formatting_rules = []
        
        # Define color mapping matching tooltip system
        alert_colors = {
            'SB': '#FF6B35',  # Dark orange (Severely Below)
            'B': '#FFB366',   # Light orange (Below)
            'G': '#4A90E2',   # Light blue (Good)
            'SG': '#2E5A87',  # Dark blue (Severely Good)
            'T': '#795548'    # Brown (Threshold alerts)
            # 'N' and 'NS' get no coloring (default text color)
        }
        
        # Color rules for overall percentile columns using overall category
        overall_percentile_columns = ['overall_percentile', 'session_overall_percentile']
        
        for col in overall_percentile_columns:
            for category, color in alert_colors.items():
                if category in ['N', 'NS']:  # Skip normal and not scored
                    continue
                    
                formatting_rules.append({
                    'if': {
                        'filter_query': f'{{overall_percentile_category}} = {category}',
                        'column_id': col
                    },
                    'color': color,
                    'fontWeight': '600'
                })
        
        # Color rules for feature-specific percentile columns using their category columns
        for feature in self.features_config.keys():
            feature_category_col = f'{feature}_category'
            
            # Feature percentile columns that should use this category
            feature_percentile_columns = [
                f'{feature}_session_percentile',
                f'{feature}_percentile',  # Legacy strata percentiles
                f'{feature}_processed_rolling_avg',
                feature,  # Raw feature value
                f'{feature}_processed'  # Processed feature value
            ]
            
            for col in feature_percentile_columns:
                for category, color in alert_colors.items():
                    if category in ['N', 'NS']:  # Skip normal and not scored
                        continue
                        
                    formatting_rules.append({
                        'if': {
                            'filter_query': f'{{{feature_category_col}}} = {category}',
                            'column_id': col
                        },
                        'color': color,
                        'fontWeight': '600'
                    })
        
        return formatting_rules

    def build(self):
        """
        Build data table component with enhanced feature-specific columns,
        session-level metrics, and collapsible column groups
        """
        # Get the data and apply formatting (with default visible columns)
        raw_data = self.data_loader.get_data()
        default_visible_columns = get_default_visible_columns()
        formatted_data = self.format_dataframe(raw_data, visible_columns=default_visible_columns)
        
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
            'reward_volume_left_mean': 'Reward Volume\nLeft (Mean)',
            'reward_volume_right_mean': 'Reward Volume\nRight (Mean)',
            'reaction_time_median': 'Reaction Time\n(Median)',
            'reaction_time_mean': 'Reaction Time\n(Mean)',
            'early_lick_rate': 'Early Lick\nRate',
            'invalid_lick_ratio': 'Invalid Lick\nRatio',
            'double_dipping_rate_finished_trials': 'Double Dipping Rate\n(Finished Trials)',
            'double_dipping_rate_finished_reward_trials': 'Double Dipping Rate\n(Reward Trials)',
            'double_dipping_rate_finished_noreward_trials': 'Double Dipping Rate\n(No Reward Trials)',
            'lick_consistency_mean_finished_trials': 'Lick Consistency\n(Finished Trials)',
            'lick_consistency_mean_finished_reward_trials': 'Lick Consistency\n(Reward Trials)',
            'lick_consistency_mean_finished_noreward_trials': 'Lick Consistency\n(No Reward Trials)',
            'avg_trial_length_in_seconds': 'Avg Trial Length\n(Seconds)',
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
            # Wilson CIs for percentiles
            formatted_column_names[f'{feature}_session_percentile_ci_lower'] = f'{feature_display}\nWilson CI Lower'
            formatted_column_names[f'{feature}_session_percentile_ci_upper'] = f'{feature_display}\nWilson CI Upper'
            # PHASE 3: Bootstrap enhancement indicators
            formatted_column_names[f'{feature}_bootstrap_enhanced'] = f'{feature_display}\nBootstrap Enhanced'
            # NEW: Bootstrap CIs for raw rolling averages
            formatted_column_names[f'{feature}_bootstrap_ci_lower'] = f'{feature_display}\nBootstrap CI Lower'
            formatted_column_names[f'{feature}_bootstrap_ci_upper'] = f'{feature_display}\nBootstrap CI Upper'
            # NEW: Bootstrap CI width
            formatted_column_names[f'{feature}_bootstrap_ci_width'] = f'{feature_display}\nBootstrap CI Width'
            # NEW: Bootstrap CI certainty
            formatted_column_names[f'{feature}_bootstrap_ci_certainty'] = f'{feature_display}\nCI Certainty'
        
        # Add overall percentile columns (both session and strata versions)
        formatted_column_names['session_overall_percentile'] = 'Session Overall\nPercentile'
        formatted_column_names['overall_percentile'] = 'Strata Overall\nPercentile'
        
        # Wilson CIs for overall percentiles
        formatted_column_names['session_overall_percentile_ci_lower'] = 'Overall Percentile\nWilson CI Lower'
        formatted_column_names['session_overall_percentile_ci_upper'] = 'Overall Percentile\nWilson CI Upper'
        
        # PHASE 2: Add outlier detection column names
        formatted_column_names['outlier_weight'] = 'Outlier\nWeight'
        formatted_column_names['is_outlier'] = 'Is\nOutlier'

        # PHASE 3: Add bootstrap enhancement indicator column names
        formatted_column_names['session_overall_bootstrap_enhanced'] = 'Overall\nBootstrap Enhanced'
        
        # NEW: Bootstrap CIs for overall rolling averages
        formatted_column_names['session_overall_bootstrap_ci_lower'] = 'Overall Rolling Avg\nBootstrap CI Lower'
        formatted_column_names['session_overall_bootstrap_ci_upper'] = 'Overall Rolling Avg\nBootstrap CI Upper'
        # NEW: Bootstrap CI width for overall
        formatted_column_names['session_overall_bootstrap_ci_width'] = 'Overall Rolling Avg\nBootstrap CI Width'
        # NEW: Overall bootstrap CI certainty
        formatted_column_names['session_overall_bootstrap_ci_certainty'] = 'Overall Rolling Avg\nCI Certainty'

        # Create columns with formatted names and custom numeric formatting
        # Create ALL column definitions (for switching between)
        all_table_data = self.format_dataframe(raw_data)  # Get all columns
        all_columns = []
        for col in all_table_data.columns:
            column_def = {
                "name": formatted_column_names.get(col, col.replace('_', ' ').title()),
                "id": col
            }
            
            # Add specific formatting for float columns
            if col in all_table_data.columns and all_table_data[col].dtype == 'float64':
                column_def['type'] = 'numeric'
                column_def['format'] = {"specifier": ".5~g"}
            
            all_columns.append(column_def)
        
        # Filter to default visible columns
        visible_columns = self.get_filtered_columns(all_columns, default_visible_columns)
        
        # DEBUG: Show column count and key session-level columns for verification
        session_cols = [col for col in formatted_data.columns if col.endswith('_session_percentile')]
        rolling_cols = [col for col in formatted_data.columns if col.endswith('_rolling_avg')]
        print(f"\nDataTable build - Total columns available: {len(all_columns)}")
        print(f"  Default visible columns: {len(visible_columns)}")
        print(f"  Session percentile columns: {len(session_cols)}")
        print(f"  Rolling average columns: {len(rolling_cols)}")
        if session_cols:
            print(f"  Example session columns: {session_cols[:3]}")
        if rolling_cols:
            print(f"  Example rolling columns: {rolling_cols[:2]}")
        print("")

        # Build the complete component with toggle controls and table
        return html.Div([
            # Main data table
            dash_table.DataTable(
                id='session-table',
                data=formatted_data.to_dict('records'),
                columns=visible_columns,  # Start with default visible columns
                page_size=25,  # Will be dynamically updated by callback
                fixed_rows={'headers': True},
                style_data_conditional=[
                    # Cursor styling for subject_id column
                    {
                        'if': {'column_id': 'subject_id'},
                        'cursor': 'pointer'
                    },
                    # Row highlighting based on alert categories
                    # SB (Severely Below) - Dark Orange
                    {
                        'if': {
                            'filter_query': '{percentile_category} = SB',
                            'column_id': ['subject_id', 'combined_alert', 'percentile_category', 'overall_percentile', 'session_overall_percentile']
                        },
                        'backgroundColor': '#FF6B35',  # Dark orange
                        'color': '#1a1a1a',  # Dark text for better readability
                        'fontWeight': '700'
                    },
                    {
                        'if': {
                            'filter_query': '{percentile_category} = SB'
                        },
                        'backgroundColor': '#FFF2EE'  # Very light orange for other columns
                    },
                    # SB subject_id column gets the border
                    {
                        'if': {
                            'filter_query': '{percentile_category} = SB',
                            'column_id': 'subject_id'
                        },
                        'borderLeft': '4px solid #FF6B35'
                    },
                    # B (Below) - Light Orange  
                    {
                        'if': {
                            'filter_query': '{percentile_category} = B',
                            'column_id': ['subject_id', 'combined_alert', 'percentile_category', 'overall_percentile', 'session_overall_percentile']
                        },
                        'backgroundColor': '#FFB366',  # Light orange
                        'color': '#2d1810',  # Dark brown text for better readability
                        'fontWeight': '700'
                    },
                    {
                        'if': {
                            'filter_query': '{percentile_category} = B'
                        },
                        'backgroundColor': '#FFF7F0'  # Very light orange for other columns
                    },
                    # B subject_id column gets the border
                    {
                        'if': {
                            'filter_query': '{percentile_category} = B',
                            'column_id': 'subject_id'
                        },
                        'borderLeft': '4px solid #FFB366'
                    },
                    # G (Good) - Light Blue
                    {
                        'if': {
                            'filter_query': '{percentile_category} = G',
                            'column_id': ['subject_id', 'combined_alert', 'percentile_category', 'overall_percentile', 'session_overall_percentile']
                        },
                        'backgroundColor': '#4A90E2',  # Light blue
                        'color': '#1a1a1a',  # Dark text for better readability
                        'fontWeight': '700'
                    },
                    {
                        'if': {
                            'filter_query': '{percentile_category} = G'
                        },
                        'backgroundColor': '#F0F6FF'  # Very light blue for other columns
                    },
                    # G subject_id column gets the border
                    {
                        'if': {
                            'filter_query': '{percentile_category} = G',
                            'column_id': 'subject_id'
                        },
                        'borderLeft': '4px solid #4A90E2'
                    },
                    # SG (Severely Good) - Dark Blue
                    {
                        'if': {
                            'filter_query': '{percentile_category} = SG',
                            'column_id': ['subject_id', 'combined_alert', 'percentile_category', 'overall_percentile', 'session_overall_percentile']
                        },
                        'backgroundColor': '#2E5A87',  # Dark blue
                        'color': '#ffffff',  # White text for dark background
                        'fontWeight': '700'
                    },
                    {
                        'if': {
                            'filter_query': '{percentile_category} = SG'
                        },
                        'backgroundColor': '#EBF3FF'  # Very light blue for other columns
                    },
                    # SG subject_id column gets the border
                    {
                        'if': {
                            'filter_query': '{percentile_category} = SG',
                            'column_id': 'subject_id'
                        },
                        'borderLeft': '4px solid #2E5A87'
                    },
                    # Special styling for combined alerts (percentile + threshold)
                    # SB with threshold alert
                    {
                        'if': {
                            'filter_query': '{combined_alert} contains "SB, T"',
                            'column_id': ['subject_id', 'combined_alert', 'percentile_category', 'overall_percentile', 'session_overall_percentile']
                        },
                        'backgroundColor': '#E55100',  # Darker orange for combined alert
                        'color': '#ffffff',  # White text for dark background
                        'fontWeight': '700',
                        'border': '2px solid #D84315'
                    },
                    {
                        'if': {
                            'filter_query': '{combined_alert} contains "SB, T"'
                        },
                        'backgroundColor': '#FFF0E6'  # Light orange background
                    },
                    # SB+T subject_id column gets thicker border for combined alerts
                    {
                        'if': {
                            'filter_query': '{combined_alert} contains "SB, T"',
                            'column_id': 'subject_id'
                        },
                        'borderLeft': '6px solid #E55100'
                    },
                    # B with threshold alert
                    {
                        'if': {
                            'filter_query': '{combined_alert} contains "B, T"',
                            'column_id': ['subject_id', 'combined_alert', 'percentile_category', 'overall_percentile', 'session_overall_percentile']
                        },
                        'backgroundColor': '#F57C00',  # Darker orange for combined alert
                        'color': '#1a1a1a',  # Dark text for better readability
                        'fontWeight': '700',
                        'border': '2px solid #EF6C00'
                    },
                    {
                        'if': {
                            'filter_query': '{combined_alert} contains "B, T"'
                        },
                        'backgroundColor': '#FFF4E6'  # Light orange background
                    },
                    # B+T subject_id column gets thicker border
                    {
                        'if': {
                            'filter_query': '{combined_alert} contains "B, T"',
                            'column_id': 'subject_id'
                        },
                        'borderLeft': '6px solid #F57C00'
                    },
                    # G with threshold alert
                    {
                        'if': {
                            'filter_query': '{combined_alert} contains "G, T"',
                            'column_id': ['subject_id', 'combined_alert', 'percentile_category', 'overall_percentile', 'session_overall_percentile']
                        },
                        'backgroundColor': '#1976D2',  # Darker blue for combined alert
                        'color': '#ffffff',  # White text for dark background
                        'fontWeight': '700',
                        'border': '2px solid #1565C0'
                    },
                    {
                        'if': {
                            'filter_query': '{combined_alert} contains "G, T"'
                        },
                        'backgroundColor': '#E8F4FD'  # Light blue background
                    },
                    # G+T subject_id column gets thicker border
                    {
                        'if': {
                            'filter_query': '{combined_alert} contains "G, T"',
                            'column_id': 'subject_id'
                        },
                        'borderLeft': '6px solid #1976D2'
                    },
                    # SG with threshold alert
                    {
                        'if': {
                            'filter_query': '{combined_alert} contains "SG, T"',
                            'column_id': ['subject_id', 'combined_alert', 'percentile_category', 'overall_percentile', 'session_overall_percentile']
                        },
                        'backgroundColor': '#0D47A1',  # Darker blue for combined alert
                        'color': '#ffffff',  # White text for dark background
                        'fontWeight': '700',
                        'border': '2px solid #01579B'
                    },
                    {
                        'if': {
                            'filter_query': '{combined_alert} contains "SG, T"'
                        },
                        'backgroundColor': '#E3F2FD'  # Light blue background
                    },
                    # SG+T subject_id column gets thicker border
                    {
                        'if': {
                            'filter_query': '{combined_alert} contains "SG, T"',
                            'column_id': 'subject_id'
                        },
                        'borderLeft': '6px solid #0D47A1'
                    },
                    # Threshold-only alerts (when percentile category is NS but has threshold alert)
                    {
                        'if': {
                            'filter_query': '{combined_alert} = T',
                            'column_id': ['subject_id', 'combined_alert', 'threshold_alert', 'total_sessions_alert', 'stage_sessions_alert', 'water_day_total_alert']
                        },
                        'backgroundColor': '#795548',  # Brown for threshold-only alerts
                        'color': '#ffffff',  # White text for dark background
                        'fontWeight': '700'
                    },
                    {
                        'if': {
                            'filter_query': '{combined_alert} = T'
                        },
                        'backgroundColor': '#F3F0EE'  # Light brown background
                    },
                    # T-only subject_id column gets the border
                    {
                        'if': {
                            'filter_query': '{combined_alert} = T',
                            'column_id': 'subject_id'
                        },
                        'borderLeft': '4px solid #795548'
                    },
                    # Individual threshold alert column styling (match tooltip colors)
                    # Total sessions alert column - when contains "T |"
                    {
                        'if': {
                            'filter_query': '{total_sessions_alert} contains "T |"',
                            'column_id': 'total_sessions_alert'
                        },
                        'color': '#795548',  # Brown color matching tooltip
                        'fontWeight': '600'
                    },
                    # Stage sessions alert column - when contains "T |"
                    {
                        'if': {
                            'filter_query': '{stage_sessions_alert} contains "T |"',
                            'column_id': 'stage_sessions_alert'
                        },
                        'color': '#795548',  # Brown color matching tooltip
                        'fontWeight': '600'
                    },
                    # Water day total alert column - when contains "T |"
                    {
                        'if': {
                            'filter_query': '{water_day_total_alert} contains "T |"',
                            'column_id': 'water_day_total_alert'
                        },
                        'color': '#795548',  # Brown color matching tooltip
                        'fontWeight': '600'
                    },
                    # Threshold alert column - when equals "T"
                    {
                        'if': {
                            'filter_query': '{threshold_alert} = T',
                            'column_id': 'threshold_alert'
                        },
                        'color': '#795548',  # Brown color matching tooltip
                        'fontWeight': '600'
                    },
                    # Base column formatting for threshold violations
                    # Session column - when any session threshold alert is triggered
                    {
                        'if': {
                            'filter_query': '{total_sessions_alert} contains "T |" || {stage_sessions_alert} contains "T |"',
                            'column_id': 'session'
                        },
                        'color': '#795548',  # Brown color matching threshold alerts
                        'fontWeight': '600'
                    },
                    # Water day total column - when water threshold alert is triggered
                    {
                        'if': {
                            'filter_query': '{water_day_total_alert} contains "T |"',
                            'column_id': 'water_day_total'
                        },
                        'color': '#795548',  # Brown color matching threshold alerts
                        'fontWeight': '600'
                    },
                    # PHASE 2: Outlier detection styling
                    # Mark outlier sessions with distinct violet/purple color
                    {
                        'if': {
                            'filter_query': '{is_outlier} = true',
                            'column_id': ['outlier_weight', 'is_outlier']
                        },
                        'backgroundColor': '#9C27B0',  # Purple for outlier indicator columns
                        'color': '#ffffff',
                        'fontWeight': '600'
                    },
                    # Add subtle background tint for outlier sessions in data columns
                    {
                        'if': {
                            'filter_query': '{is_outlier} = true'
                        },
                        'backgroundColor': '#F3E5F5'  # Very light purple background for outlier rows
                    },
                    # Add border indicator for outlier sessions
                    {
                        'if': {
                            'filter_query': '{is_outlier} = true',
                            'column_id': 'subject_id'
                        },
                        'borderRight': '3px solid #9C27B0'  # Purple right border on subject_id for outliers
                    },
                    # PHASE 3: Bootstrap enhancement styling
                    # Highlight bootstrap enhanced overall percentile in green
                    {
                        'if': {
                            'filter_query': '{session_overall_bootstrap_enhanced} = true',
                            'column_id': 'session_overall_bootstrap_enhanced'
                        },
                        'backgroundColor': '#4CAF50',  # Green for bootstrap enhanced
                        'color': '#ffffff',
                        'fontWeight': '600'
                    },
                    # Subtle styling for non-bootstrap enhanced overall percentile
                    {
                        'if': {
                            'filter_query': '{session_overall_bootstrap_enhanced} = false',
                            'column_id': 'session_overall_bootstrap_enhanced'
                        },
                        'backgroundColor': '#F5F5F5',  # Light gray for non-bootstrap
                        'color': '#666666',
                        'fontStyle': 'italic'
                    },
                    # Feature-specific bootstrap enhancement styling
                    # Finished trials bootstrap enhanced
                    {
                        'if': {
                            'filter_query': '{finished_trials_bootstrap_enhanced} = true',
                            'column_id': 'finished_trials_bootstrap_enhanced'
                        },
                        'backgroundColor': '#4CAF50',
                        'color': '#ffffff',
                        'fontWeight': '600'
                    },
                    {
                        'if': {
                            'filter_query': '{finished_trials_bootstrap_enhanced} = false',
                            'column_id': 'finished_trials_bootstrap_enhanced'
                        },
                        'backgroundColor': '#F5F5F5',
                        'color': '#666666',
                        'fontStyle': 'italic'
                    },
                    # Ignore rate bootstrap enhanced
                    {
                        'if': {
                            'filter_query': '{ignore_rate_bootstrap_enhanced} = true',
                            'column_id': 'ignore_rate_bootstrap_enhanced'
                        },
                        'backgroundColor': '#4CAF50',
                        'color': '#ffffff',
                        'fontWeight': '600'
                    },
                    {
                        'if': {
                            'filter_query': '{ignore_rate_bootstrap_enhanced} = false',
                            'column_id': 'ignore_rate_bootstrap_enhanced'
                        },
                        'backgroundColor': '#F5F5F5',
                        'color': '#666666',
                        'fontStyle': 'italic'
                    },
                    # Total trials bootstrap enhanced
                    {
                        'if': {
                            'filter_query': '{total_trials_bootstrap_enhanced} = true',
                            'column_id': 'total_trials_bootstrap_enhanced'
                        },
                        'backgroundColor': '#4CAF50',
                        'color': '#ffffff',
                        'fontWeight': '600'
                    },
                    {
                        'if': {
                            'filter_query': '{total_trials_bootstrap_enhanced} = false',
                            'column_id': 'total_trials_bootstrap_enhanced'
                        },
                        'backgroundColor': '#F5F5F5',
                        'color': '#666666',
                        'fontStyle': 'italic'
                    },
                    # Foraging performance bootstrap enhanced
                    {
                        'if': {
                            'filter_query': '{foraging_performance_bootstrap_enhanced} = true',
                            'column_id': 'foraging_performance_bootstrap_enhanced'
                        },
                        'backgroundColor': '#4CAF50',
                        'color': '#ffffff',
                        'fontWeight': '600'
                    },
                    {
                        'if': {
                            'filter_query': '{foraging_performance_bootstrap_enhanced} = false',
                            'column_id': 'foraging_performance_bootstrap_enhanced'
                        },
                        'backgroundColor': '#F5F5F5',
                        'color': '#666666',
                        'fontStyle': 'italic'
                    },
                    # Bias naive bootstrap enhanced
                    {
                        'if': {
                            'filter_query': '{abs(bias_naive)_bootstrap_enhanced} = true',
                            'column_id': 'abs(bias_naive)_bootstrap_enhanced'
                        },
                        'backgroundColor': '#4CAF50',
                        'color': '#ffffff',
                        'fontWeight': '600'
                    },
                    {
                        'if': {
                            'filter_query': '{abs(bias_naive)_bootstrap_enhanced} = false',
                            'column_id': 'abs(bias_naive)_bootstrap_enhanced'
                        },
                        'backgroundColor': '#F5F5F5',
                        'color': '#666666',
                        'fontStyle': 'italic'
                    },
                    # NEW: CI Certainty border styling
                    # Overall percentile - certain (narrow CI)
                    {
                        'if': {
                            'filter_query': '{session_overall_bootstrap_ci_certainty} = certain',
                            'column_id': ['session_overall_rolling_avg', 'session_overall_percentile']
                        },
                        'borderLeft': '4px solid #4CAF50'  # Green for certain
                    },
                    # Overall percentile - uncertain (wide CI) 
                    {
                        'if': {
                            'filter_query': '{session_overall_bootstrap_ci_certainty} = uncertain',
                            'column_id': ['session_overall_rolling_avg', 'session_overall_percentile']
                        },
                        'borderLeft': '4px solid #FF5722'  # Red-orange for uncertain
                    },
                    # Finished trials - certain
                    {
                        'if': {
                            'filter_query': '{finished_trials_bootstrap_ci_certainty} = certain',
                            'column_id': ['finished_trials', 'finished_trials_processed_rolling_avg']
                        },
                        'borderLeft': '4px solid #4CAF50'
                    },
                    # Finished trials - uncertain
                    {
                        'if': {
                            'filter_query': '{finished_trials_bootstrap_ci_certainty} = uncertain',
                            'column_id': ['finished_trials', 'finished_trials_processed_rolling_avg']
                        },
                        'borderLeft': '4px solid #FF5722'
                    },
                    # Ignore rate - certain
                    {
                        'if': {
                            'filter_query': '{ignore_rate_bootstrap_ci_certainty} = certain',
                            'column_id': ['ignore_rate', 'ignore_rate_processed_rolling_avg']
                        },
                        'borderLeft': '4px solid #4CAF50'
                    },
                    # Ignore rate - uncertain
                    {
                        'if': {
                            'filter_query': '{ignore_rate_bootstrap_ci_certainty} = uncertain',
                            'column_id': ['ignore_rate', 'ignore_rate_processed_rolling_avg']
                        },
                        'borderLeft': '4px solid #FF5722'
                    },
                    # Total trials - certain
                    {
                        'if': {
                            'filter_query': '{total_trials_bootstrap_ci_certainty} = certain',
                            'column_id': ['total_trials', 'total_trials_processed_rolling_avg']
                        },
                        'borderLeft': '4px solid #4CAF50'
                    },
                    # Total trials - uncertain
                    {
                        'if': {
                            'filter_query': '{total_trials_bootstrap_ci_certainty} = uncertain',
                            'column_id': ['total_trials', 'total_trials_processed_rolling_avg']
                        },
                        'borderLeft': '4px solid #FF5722'
                    },
                    # Foraging performance - certain
                    {
                        'if': {
                            'filter_query': '{foraging_performance_bootstrap_ci_certainty} = certain',
                            'column_id': ['foraging_performance', 'foraging_performance_processed_rolling_avg']
                        },
                        'borderLeft': '4px solid #4CAF50'
                    },
                    # Foraging performance - uncertain
                    {
                        'if': {
                            'filter_query': '{foraging_performance_bootstrap_ci_certainty} = uncertain',
                            'column_id': ['foraging_performance', 'foraging_performance_processed_rolling_avg']
                        },
                        'borderLeft': '4px solid #FF5722'
                    },
                    # Bias naive - certain
                    {
                        'if': {
                            'filter_query': '{abs(bias_naive)_bootstrap_ci_certainty} = certain',
                            'column_id': ['abs(bias_naive)', 'abs(bias_naive)_processed_rolling_avg']
                        },
                        'borderLeft': '4px solid #4CAF50'
                    },
                    # Bias naive - uncertain
                    {
                        'if': {
                            'filter_query': '{abs(bias_naive)_bootstrap_ci_certainty} = uncertain',
                            'column_id': ['abs(bias_naive)', 'abs(bias_naive)_processed_rolling_avg']
                        },
                        'borderLeft': '4px solid #FF5722'
                    }
                ] + self._get_percentile_formatting_rules(),
                style_table={
                    'overflowX': 'auto',  # Keep horizontal scroll for wide tables
                    'backgroundColor': 'white',
                    'width': '100%',
                    'marginBottom': '0px',
                    'height': 'auto'  # Let height be determined by content
                },
                style_cell={
                    'textAlign': 'left',
                    'padding': '10px 12px',  # Consistent cell padding
                    'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
                    'fontSize': '14px',
                    'height': '48px',  # Fixed row height for consistency
                    'minWidth': '100px',
                    'backgroundColor': 'white',
                    'border': 'none',
                    'lineHeight': '1.2'
                },
                style_header={
                    'backgroundColor': 'white',
                    'fontWeight': '600',
                    'border': 'none',
                    'borderBottom': '1px solid #e0e0e0',
                    'position': 'sticky',
                    'top': 0,
                    'zIndex': 999,
                    'height': '60px',  # Fixed header height
                    'whiteSpace': 'normal',
                    'textAlign': 'center',  
                    'padding': '10px 5px',  
                    'lineHeight': '15px'     
                },
                cell_selectable=True,
                row_selectable=False
            )
        ], className="data-table-container", style={'width': '100%', 'overflow': 'visible'})