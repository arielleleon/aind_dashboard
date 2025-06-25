import pandas as pd
from dash import dash_table, html

from app_utils import AppUtils
from app_utils.app_analysis.overall_percentile_calculator import (
    OverallPercentileCalculator,
)
from app_utils.app_analysis.reference_processor import ReferenceProcessor
from app_utils.ui_utils import (
    create_empty_dataframe_structure,
    format_strata_abbreviations,
    get_optimized_table_data,
    process_unified_alerts_integration,
)

from .column_groups_config import get_default_visible_columns


class AppDataFrame:
    def __init__(self, app_utils=None):
        if app_utils is not None:
            self.app_utils = app_utils
        else:
            self.app_utils = AppUtils()

        # Feature configuration for quantile analysis
        self.features_config = {
            "finished_trials": False,  # Higher is better
            "ignore_rate": True,  # Lower is better
            "total_trials": False,  # Higher is better
            "foraging_performance": False,  # Higher is better
            "abs(bias_naive)": True,  # Lower is better
        }

        # Configure threshold alerts
        self.threshold_config = {
            "session": {"condition": "gt", "value": 40},  # Total sessions threshold
            "water_day_total": {
                "condition": "gt",
                "value": 3.5,  # Water day total threshold (ml)
            },
        }

        # Stage-specific session thresholds
        self.stage_thresholds = {
            "STAGE_1": 5,
            "STAGE_2": 5,
            "STAGE_3": 6,
            "STAGE_4": 10,
            "STAGE_FINAL": 10,
            "GRADUATED": 20,
        }

        self.data_loader = self.app_utils.data_loader

        # Create reference processor with minimal default settings for filtering purposes
        self.reference_processor = ReferenceProcessor(
            features_config={},
            min_sessions=1,
            min_days=1,
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
        return [col for col in all_columns if col["id"] in visible_column_ids]

    def format_dataframe(
        self, df: pd.DataFrame, visible_columns: list = None
    ) -> pd.DataFrame:
        """
        Format the dataframe for display in the table with unified session-level metrics
        """
        original_df = df.copy() if df is not None else pd.DataFrame()

        if original_df.empty or "session_date" not in original_df.columns:
            print("ERROR: Invalid dataframe passed to format_dataframe.")
            print(
                f"DataFrame is {'empty' if original_df.empty else 'missing session_date column'}"
            )
            if not original_df.empty:
                print(f"Available columns: {original_df.columns.tolist()}")

            # Return empty dataframe with required columns to avoid breaking the UI
            return create_empty_dataframe_structure()

        # STEP 1: Get optimized data using business logic
        recent_sessions = get_optimized_table_data(self.app_utils, use_cache=True)

        # STEP 2: Process alerts using business logic
        recent_sessions = process_unified_alerts_integration(
            recent_sessions,
            self.app_utils,
            threshold_config=self.threshold_config,
            stage_thresholds=self.stage_thresholds,
        )

        # STEP 3: Apply strata formatting using business logic
        recent_sessions = format_strata_abbreviations(recent_sessions)

        # STEP 4: Filter to visible columns if specified
        if visible_columns is not None:
            existing_visible_cols = [
                col for col in visible_columns if col in recent_sessions.columns
            ]
            recent_sessions = recent_sessions[existing_visible_cols]

        return recent_sessions

    def _get_percentile_formatting_rules(self):
        """
        Generate conditional formatting rules for percentile-based values
        Uses computed alert categories from UI cache to match tooltip coloring
        """
        formatting_rules = []

        # Define color mapping matching tooltip system
        alert_colors = {
            "SB": "#FF6B35",  # Dark orange (Severely Below)
            "B": "#FFB366",  # Light orange (Below)
            "G": "#4A90E2",  # Light blue (Good)
            "SG": "#2E5A87",  # Dark blue (Significantly Good)
            "T": "#795548",  # Brown (Threshold alerts)
            # 'N' and 'NS' get no coloring (default text color)
        }

        overall_percentile_columns = [
            "overall_percentile",
            "session_overall_percentile",
        ]

        for col in overall_percentile_columns:
            for category, color in alert_colors.items():
                if category in ["N", "NS"]:  # Skip normal and not scored
                    continue

                formatting_rules.append(
                    {
                        "if": {
                            "filter_query": f"{{overall_percentile_category}} = {category}",
                            "column_id": col,
                        },
                        "color": color,
                        "fontWeight": "600",
                    }
                )

        # Color rules for feature-specific percentile columns using their category columns
        for feature in self.features_config.keys():
            feature_category_col = f"{feature}_category"

            # Feature percentile columns that should use this category
            feature_percentile_columns = [
                f"{feature}_session_percentile",
                f"{feature}_percentile",
                f"{feature}_processed_rolling_avg",
                feature,
                f"{feature}_processed",
            ]

            for col in feature_percentile_columns:
                for category, color in alert_colors.items():
                    if category in ["N", "NS"]:  # Skip normal and not scored
                        continue

                    formatting_rules.append(
                        {
                            "if": {
                                "filter_query": f"{{{feature_category_col}}} = {category}",
                                "column_id": col,
                            },
                            "color": color,
                            "fontWeight": "600",
                        }
                    )

        return formatting_rules

    def build(self):
        """
        Build data table component with enhanced feature-specific columns,
        session-level metrics, and collapsible column groups
        """
        # Get the data and apply formatting (with default visible columns)
        raw_data = self.data_loader.get_data()
        default_visible_columns = get_default_visible_columns()
        formatted_data = self.format_dataframe(
            raw_data, visible_columns=default_visible_columns
        )

        formatted_column_names = {
            "subject_id": "Subject ID",
            "combined_alert": "Alert",
            "percentile_category": "Percentile Alert",
            "ns_reason": "Not Scored Reason",
            "threshold_alert": "Threshold Alert",
            "total_sessions_alert": "Total Sessions Alert",
            "stage_sessions_alert": "Stage Sessions Alert",
            "water_day_total_alert": "Water Day Total Alert",
            "overall_percentile": "Overall\nPercentile",
            "session_overall_percentile": "Session\nPercentile",
            "strata": "Strata",
            "strata_abbr": "Strata (Abbr)",
            "current_stage_actual": "Stage",
            "curriculum": "curriculum",
            "session_date": "Date",
            "session": "Session",
            "rig": "Rig",
            "trainer": "Trainer",
            "PI": "PI",
            "session_run_time": "Run Time",
            "total_trials": "Total Trials",
            "finished_trials": "Finished Trials",
            "finished_rate": "Finish Rate",
            "ignore_rate": "Ignore Rate",
            "water_in_session_foraging": "Water In-Session\n(Foraging)",
            "water_in_session_manual": "Water In-Session\n(Manual)",
            "water_in_session_total": "Water In-Session\n(Total)",
            "water_after_session": "Water After\nSession",
            "water_day_total": "Water Day\nTotal",
            "base_weight": "Base Weight",
            "target_weight": "Target Weight",
            "target_weight_ratio": "Target Weight\nRatio",
            "weight_after": "Weight After",
            "weight_after_ratio": "Weight After\nRatio",
            "reward_volume_left_mean": "Reward Volume\nLeft (Mean)",
            "reward_volume_right_mean": "Reward Volume\nRight (Mean)",
            "reaction_time_median": "Reaction Time\n(Median)",
            "reaction_time_mean": "Reaction Time\n(Mean)",
            "early_lick_rate": "Early Lick\nRate",
            "invalid_lick_ratio": "Invalid Lick\nRatio",
            "double_dipping_rate_finished_trials": "Double Dipping Rate\n(Finished Trials)",
            "double_dipping_rate_finished_reward_trials": "Double Dipping Rate\n(Reward Trials)",
            "double_dipping_rate_finished_noreward_trials": "Double Dipping Rate\n(No Reward Trials)",
            "lick_consistency_mean_finished_trials": "Lick Consistency\n(Finished Trials)",
            "lick_consistency_mean_finished_reward_trials": "Lick Consistency\n(Reward Trials)",
            "lick_consistency_mean_finished_noreward_trials": "Lick Consistency\n(No Reward Trials)",
            "avg_trial_length_in_seconds": "Avg Trial Length\n(Seconds)",
            "total_trials_with_autowater": "Total Trials\n(Autowater)",
            "finished_trials_with_autowater": "Finished Trials\n(Autowater)",
            "finished_rate_with_autowater": "Finish Rate\n(Autowater)",
            "ignore_rate_with_autowater": "Ignore Rate\n(Autowater)",
            "autowater_collected": "Autowater\nCollected",
            "autowater_ignored": "Autowater\nIgnored",
            "water_day_total_last_session": "Water Day Total\n(Last Session)",
            "water_after_session_last_session": "Water After\n(Last Session)",
        }

        # Add feature-specific column names
        for feature in self.features_config.keys():
            feature_display = (
                feature.replace("_", " ").replace("abs(", "|").replace(")", "|").title()
            )
            formatted_column_names[f"{feature}_percentile"] = (
                f"{feature_display}\nStrata %ile"
            )
            formatted_column_names[f"{feature}_category"] = f"{feature_display}\nAlert"
            formatted_column_names[f"{feature}_processed"] = (
                f"{feature_display}\nProcessed"
            )
            formatted_column_names[f"{feature}_session_percentile"] = (
                f"{feature_display}\nSession %ile"
            )
            formatted_column_names[f"{feature}_processed_rolling_avg"] = (
                f"{feature_display}\nRolling Avg"
            )
            formatted_column_names[f"{feature}_session_percentile_ci_lower"] = (
                f"{feature_display}\nWilson CI Lower"
            )
            formatted_column_names[f"{feature}_session_percentile_ci_upper"] = (
                f"{feature_display}\nWilson CI Upper"
            )

        formatted_column_names["session_overall_percentile"] = (
            "Session Overall\nPercentile"
        )
        formatted_column_names["overall_percentile"] = "Strata Overall\nPercentile"

        formatted_column_names["session_overall_percentile_ci_lower"] = (
            "Overall Percentile\nWilson CI Lower"
        )
        formatted_column_names["session_overall_percentile_ci_upper"] = (
            "Overall Percentile\nWilson CI Upper"
        )

        formatted_column_names["outlier_weight"] = "Outlier\nWeight"
        formatted_column_names["is_outlier"] = "Is\nOutlier"

        # Create columns with formatted names and custom numeric formatting
        # Create ALL column definitions (for switching between)
        all_table_data = self.format_dataframe(raw_data)  # Get all columns
        all_columns = []
        for col in all_table_data.columns:
            column_def = {
                "name": formatted_column_names.get(col, col.replace("_", " ").title()),
                "id": col,
            }

            # Add specific formatting for float columns
            if col in all_table_data.columns and all_table_data[col].dtype == "float64":
                column_def["type"] = "numeric"
                column_def["format"] = {"specifier": ".2~g"}

            all_columns.append(column_def)

        # Filter to default visible columns
        visible_columns = self.get_filtered_columns(
            all_columns, default_visible_columns
        )

        # Build the complete component with toggle controls and table
        return html.Div(
            [
                # Main data table
                dash_table.DataTable(
                    id="session-table",
                    data=formatted_data.to_dict("records"),
                    columns=visible_columns,
                    page_size=25,
                    fixed_rows={"headers": True},
                    style_data_conditional=[
                        {"if": {"column_id": "subject_id"}, "cursor": "pointer"},
                        {
                            "if": {"filter_query": "{percentile_category} = SB"},
                            "backgroundColor": "#FFE5DB",
                        },
                        {
                            "if": {
                                "filter_query": "{percentile_category} = SB",
                                "column_id": "subject_id",
                            },
                            "borderLeft": "4px solid #FF6B35",
                            "color": "#D84315",
                            "fontWeight": "600",
                        },
                        {
                            "if": {"filter_query": "{percentile_category} = B"},
                            "backgroundColor": "#FFF3E0",
                        },
                        {
                            "if": {
                                "filter_query": "{percentile_category} = B",
                                "column_id": "subject_id",
                            },
                            "borderLeft": "4px solid #FFB366",
                            "color": "#F57C00",
                            "fontWeight": "600",
                        },
                        {
                            "if": {"filter_query": "{percentile_category} = G"},
                            "backgroundColor": "#E3F2FD",
                        },
                        {
                            "if": {
                                "filter_query": "{percentile_category} = G",
                                "column_id": "subject_id",
                            },
                            "borderLeft": "4px solid #4A90E2",
                            "color": "#1976D2",
                            "fontWeight": "600",
                        },
                        {
                            "if": {"filter_query": "{percentile_category} = SG"},
                            "backgroundColor": "#BBDEFB",
                        },
                        {
                            "if": {
                                "filter_query": "{percentile_category} = SG",
                                "column_id": "subject_id",
                            },
                            "borderLeft": "4px solid #2E5A87",
                            "color": "#0D47A1",
                            "fontWeight": "600",
                        },
                        {
                            "if": {"filter_query": '{combined_alert} contains "SB, T"'},
                            "backgroundColor": "#FFCCBC",
                            "border": "1px solid #FF5722",
                        },
                        {
                            "if": {
                                "filter_query": '{combined_alert} contains "SB, T"',
                                "column_id": "subject_id",
                            },
                            "borderLeft": "6px solid #E55100",
                            "color": "#BF360C",
                            "fontWeight": "700",
                        },
                        {
                            "if": {"filter_query": '{combined_alert} contains "B, T"'},
                            "backgroundColor": "#FFE0B2",
                            "border": "1px solid #FF9800",
                        },
                        {
                            "if": {
                                "filter_query": '{combined_alert} contains "B, T"',
                                "column_id": "subject_id",
                            },
                            "borderLeft": "6px solid #F57C00",
                            "color": "#E65100",
                            "fontWeight": "700",
                        },
                        {
                            "if": {"filter_query": '{combined_alert} contains "G, T"'},
                            "backgroundColor": "#C5E1FF",
                            "border": "1px solid #2196F3",
                        },
                        {
                            "if": {
                                "filter_query": '{combined_alert} contains "G, T"',
                                "column_id": "subject_id",
                            },
                            "borderLeft": "6px solid #1976D2",
                            "color": "#0D47A1",
                            "fontWeight": "700",
                        },
                        {
                            "if": {"filter_query": '{combined_alert} contains "SG, T"'},
                            "backgroundColor": "#90CAF9",
                            "border": "1px solid #1976D2",
                        },
                        {
                            "if": {
                                "filter_query": '{combined_alert} contains "SG, T"',
                                "column_id": "subject_id",
                            },
                            "borderLeft": "6px solid #0D47A1",
                            "color": "#01579B",
                            "fontWeight": "700",
                        },
                        {
                            "if": {"filter_query": "{combined_alert} = T"},
                            "backgroundColor": "#EFEBE9",
                        },
                        {
                            "if": {
                                "filter_query": "{combined_alert} = T",
                                "column_id": "subject_id",
                            },
                            "borderLeft": "4px solid #795548",
                            "color": "#5D4037",
                            "fontWeight": "600",
                        },
                        {
                            "if": {
                                "filter_query": '{total_sessions_alert} contains "T |"',
                                "column_id": "total_sessions_alert",
                            },
                            "color": "#795548",
                            "fontWeight": "600",
                        },
                        {
                            "if": {
                                "filter_query": '{stage_sessions_alert} contains "T |"',
                                "column_id": "stage_sessions_alert",
                            },
                            "color": "#795548",
                            "fontWeight": "600",
                        },
                        {
                            "if": {
                                "filter_query": '{water_day_total_alert} contains "T |"',
                                "column_id": "water_day_total_alert",
                            },
                            "color": "#795548",
                            "fontWeight": "600",
                        },
                        {
                            "if": {
                                "filter_query": "{threshold_alert} = T",
                                "column_id": "threshold_alert",
                            },
                            "color": "#795548",
                            "fontWeight": "600",
                        },
                        {
                            "if": {
                                "filter_query": '{total_sessions_alert} contains "T |" || {stage_sessions_alert} contains "T |"',
                                "column_id": "session",
                            },
                            "color": "#795548",
                            "fontWeight": "600",
                        },
                        {
                            "if": {
                                "filter_query": '{water_day_total_alert} contains "T |"',
                                "column_id": "water_day_total",
                            },
                            "color": "#795548",
                            "fontWeight": "600",
                        },
                        {
                            "if": {
                                "filter_query": "{is_outlier} = true",
                                "column_id": ["outlier_weight", "is_outlier"],
                            },
                            "backgroundColor": "#9C27B0",
                            "color": "#ffffff",
                            "fontWeight": "600",
                        },
                        {
                            "if": {
                                "filter_query": "{is_outlier} = true",
                                "column_id": "subject_id",
                            },
                            "borderRight": "3px solid #9C27B0",
                        },
                        {
                            "if": {
                                "filter_query": "{session_overall_percentile_certainty} = certain",
                                "column_id": [
                                    "session_overall_percentile",
                                    "overall_percentile",
                                ],
                            },
                            "borderLeft": "4px solid #4CAF50",
                        },
                        {
                            "if": {
                                "filter_query": "{session_overall_percentile_certainty} = uncertain",
                                "column_id": [
                                    "session_overall_percentile",
                                    "overall_percentile",
                                ],
                            },
                            "borderLeft": "4px solid #F44336",
                        },
                        {
                            "if": {
                                "filter_query": "{finished_trials_certainty} = certain",
                                "column_id": [
                                    "finished_trials",
                                    "finished_trials_session_percentile",
                                ],
                            },
                            "borderLeft": "4px solid #4CAF50",
                        },
                        {
                            "if": {
                                "filter_query": "{finished_trials_certainty} = uncertain",
                                "column_id": [
                                    "finished_trials",
                                    "finished_trials_session_percentile",
                                ],
                            },
                            "borderLeft": "4px solid #F44336",
                        },
                        {
                            "if": {
                                "filter_query": "{ignore_rate_certainty} = certain",
                                "column_id": [
                                    "ignore_rate",
                                    "ignore_rate_session_percentile",
                                ],
                            },
                            "borderLeft": "4px solid #4CAF50",
                        },
                        {
                            "if": {
                                "filter_query": "{ignore_rate_certainty} = uncertain",
                                "column_id": [
                                    "ignore_rate",
                                    "ignore_rate_session_percentile",
                                ],
                            },
                            "borderLeft": "4px solid #F44336",
                        },
                        {
                            "if": {
                                "filter_query": "{total_trials_certainty} = certain",
                                "column_id": [
                                    "total_trials",
                                    "total_trials_session_percentile",
                                ],
                            },
                            "borderLeft": "4px solid #4CAF50",
                        },
                        {
                            "if": {
                                "filter_query": "{total_trials_certainty} = uncertain",
                                "column_id": [
                                    "total_trials",
                                    "total_trials_session_percentile",
                                ],
                            },
                            "borderLeft": "4px solid #F44336",
                        },
                        {
                            "if": {
                                "filter_query": "{foraging_performance_certainty} = certain",
                                "column_id": [
                                    "foraging_performance",
                                    "foraging_performance_session_percentile",
                                ],
                            },
                            "borderLeft": "4px solid #4CAF50",
                        },
                        {
                            "if": {
                                "filter_query": "{foraging_performance_certainty} = uncertain",
                                "column_id": [
                                    "foraging_performance",
                                    "foraging_performance_session_percentile",
                                ],
                            },
                            "borderLeft": "4px solid #F44336",
                        },
                        {
                            "if": {
                                "filter_query": "{abs(bias_naive)_certainty} = certain",
                                "column_id": [
                                    "abs(bias_naive)",
                                    "abs(bias_naive)_session_percentile",
                                ],
                            },
                            "borderLeft": "4px solid #4CAF50",
                        },
                        {
                            "if": {
                                "filter_query": "{abs(bias_naive)_certainty} = uncertain",
                                "column_id": [
                                    "abs(bias_naive)",
                                    "abs(bias_naive)_session_percentile",
                                ],
                            },
                            "borderLeft": "4px solid #F44336",
                        },
                    ]
                    + self._get_percentile_formatting_rules(),
                    style_table={
                        "overflowX": "auto",
                        "backgroundColor": "white",
                        "width": "100%",
                        "marginBottom": "0px",
                        "height": "auto",
                    },
                    style_cell={
                        "textAlign": "left",
                        "padding": "10px 12px",
                        "fontFamily": '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
                        "fontSize": "14px",
                        "height": "48px",
                        "minWidth": "100px",
                        "backgroundColor": "white",
                        "border": "none",
                        "lineHeight": "1.2",
                    },
                    style_header={
                        "backgroundColor": "white",
                        "fontWeight": "600",
                        "border": "none",
                        "borderBottom": "1px solid #e0e0e0",
                        "position": "sticky",
                        "top": 0,
                        "zIndex": 999,
                        "height": "60px",
                        "whiteSpace": "normal",
                        "textAlign": "center",
                        "padding": "10px 5px",
                        "lineHeight": "15px",
                    },
                    cell_selectable=True,
                    row_selectable=False,
                )
            ],
            className="data-table-container",
            style={"width": "100%", "overflow": "visible"},
        )
