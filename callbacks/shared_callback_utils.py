"""
Shared Callback Utilities Module

This module provides centralized imports, component instances, and helper functions
for use across all callback modules in the modular Dash 2.x architecture.

Design Goals:
- Avoid code duplication across callback modules
- Provide single instances of shared components
- Centralize common imports to reduce boilerplate
- Support helper functions that can be reused
- Maintain compatibility with Dash 2.x @callback decorator pattern
- Prevent circular import issues

Usage:
    from callbacks.shared_callback_utils import (
        # Common imports
        Input, Output, State, callback, ALL, ctx,
        html, dcc, dbc, pd, datetime, timedelta,
        go, json,

        # Shared component instances
        app_utils, app_dataframe, app_filter,
        session_card, image_loader, compact_info,
        subject_timeseries, subject_percentile_timeseries, percentile_heatmap,
        app_tooltip,

        # Helper functions
        format_multi_value, create_filter_badge,
        extract_highlighted_session, throttle_callback_execution
    )
"""

import json
from datetime import datetime, timedelta

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import (
    ALL,
    Input,
    Output,
    State,
    callback,
    clientside_callback,
    ctx,
    dcc,
    html,
)

from app_elements.app_content.app_dataframe.app_dataframe import AppDataFrame
from app_elements.app_filter.app_filter import AppFilter
from shared_utils import app_utils

app_dataframe = AppDataFrame(app_utils=app_utils)
app_filter = AppFilter()

# Subject detail components
from app_elements.app_subject_detail.app_session_card import AppSessionCard
from app_elements.app_subject_detail.app_subject_compact_info import (
    AppSubjectCompactInfo,
)
from app_elements.app_subject_detail.app_subject_image_loader import (
    AppSubjectImageLoader,
)

session_card = AppSessionCard()
image_loader = AppSubjectImageLoader()
compact_info = AppSubjectCompactInfo()

from app_elements.app_subject_detail.app_subject_percentile_heatmap import (
    AppSubjectPercentileHeatmap,
)
from app_elements.app_subject_detail.app_subject_percentile_timeseries import (
    AppSubjectPercentileTimeseries,
)

# Visualization components
from app_elements.app_subject_detail.app_subject_timeseries import AppSubjectTimeseries

subject_timeseries = AppSubjectTimeseries()
subject_percentile_timeseries = AppSubjectPercentileTimeseries()
percentile_heatmap = AppSubjectPercentileHeatmap()

# Tooltip component
from app_elements.app_content.app_tooltip.app_hover_tooltip import AppHoverTooltip

app_tooltip = AppHoverTooltip(app_utils=app_utils)


def format_multi_value(value):
    """
    Format multi - select filter values for display.

    Used across filter callbacks to provide consistent formatting
    of multi - select dropdown values in filter badges.

    Args:
        value: Single value, list of values, or None

    Returns:
        str: Formatted display string
    """
    if isinstance(value, list):
        if len(value) == 1:
            return value[0]
        elif len(value) <= 3:
            return ", ".join(value)
        else:
            return f"{', '.join(value[:2])}, +{len(value) - 2} more"
    else:
        return str(value) if value is not None else ""


def create_filter_badge(label, filter_type, filter_value):
    """
    Create a removable filter badge component.

    Used across filter callbacks to create consistent filter display badges
    with remove functionality.

    Args:
        label (str): Display text for the badge
        filter_type (str): Type of filter (used in badge ID)
        filter_value: Value of the filter (used in badge ID)

    Returns:
        dbc.Badge: Styled badge component with remove functionality
    """
    filter_id = f"{filter_type}:{filter_value}"
    return dbc.Badge(
        [
            label,
            html.Span(
                "×",
                className="ms - 1",
                id={"type": "remove - filter", "index": filter_id},
            ),
        ],
        color="primary",
        className="me - 1 mb - 1 filter - badge",
        id={"type": "filter - badge", "index": filter_id},
    )


def extract_highlighted_session(n_clicks_list, scroll_state, card_ids):
    """
    Extract highlighted session ID from click events or scroll state.

    This helper function reduces code duplication across visualization callbacks
    by centralizing the session highlighting logic.

    Args:
        n_clicks_list (list): List of click counts for session cards
        scroll_state (dict): Current scroll state data
        card_ids (list): List of session card IDs

    Returns:
        int or None: Session number to highlight, or None if no session should be highlighted
    """
    highlighted_session = None

    # Check for card clicks first (priority over scroll)
    if n_clicks_list and any(n_clicks_list):
        max_clicks = max(n_clicks_list)
        if max_clicks > 0:
            clicked_idx = n_clicks_list.index(max_clicks)
            if clicked_idx < len(card_ids):
                card_id = card_ids[clicked_idx]
                session_str = card_id.get("index", "").split("-")[-1]
                try:
                    highlighted_session = int(float(session_str))
                except (ValueError, IndexError):
                    pass

    # Check scroll state if no click detected
    elif scroll_state and scroll_state.get("visible_session"):
        visible_session = scroll_state.get("visible_session")
        if "-" in visible_session:
            session_str = visible_session.split("-")[-1]
            try:
                highlighted_session = int(float(session_str))
            except (ValueError, IndexError):
                pass

    return highlighted_session


def get_alert_color(category):
    """
    Get alert color based on category for consistent styling.

    Used across tooltip and table formatting to ensure consistent
    color coding of alert categories.

    Args:
        category (str): Alert category ('SB', 'B', 'G', 'SG', 'T', etc.)

    Returns:
        str or None: CSS color value or None if category not recognized
    """
    colors = {
        "SB": "#FF6B35",
        "B": "#FFB366",
        "G": "#4A90E2",
        "SG": "#2E5A87",
        "T": "#795548",
    }
    return colors.get(category)


def throttle_callback_execution(callback_func, delay_ms=100):
    """
    Throttle callback execution to improve performance.

    Useful for callbacks that may be triggered frequently (scroll, resize, etc.)
    to prevent overwhelming the client.

    Args:
        callback_func: Function to throttle
        delay_ms (int): Delay in milliseconds between executions

    Returns:
        function: Throttled version of the callback function
    """
    import time

    def throttled_func(*args, **kwargs):
        current_time = time.time() * 1000  # Convert to ms

        if not hasattr(throttled_func, "last_execution"):
            throttled_func.last_execution = 0

        if current_time - throttled_func.last_execution >= delay_ms:
            throttled_func.last_execution = current_time
            return callback_func(*args, **kwargs)
        else:
            # Return previous result or default
            return getattr(throttled_func, "last_result", None)

    return throttled_func


def safe_extract_subject_data(table_data, active_cell, page_current=0, page_size=50):
    """
    Safely extract subject data from table with pagination support.

    Used across subject detail callbacks to safely access table data
    with proper bounds checking and pagination calculations.

    Args:
        table_data (list): Table data rows
        active_cell (dict): Active cell information
        page_current (int): Current page number (0 - indexed)
        page_size (int): Number of rows per page

    Returns:
        tuple: (subject_data_dict, subject_id) or (None, None) if invalid
    """
    if not active_cell or not table_data:
        return None, None

    # Handle None values properly like the original code
    current_page = page_current if page_current is not None else 0
    rows_per_page = page_size if page_size is not None else 50

    # Calculate the actual row index in the full dataset
    absolute_row_idx = (current_page * rows_per_page) + active_cell["row"]

    # Safety check for row bounds
    if absolute_row_idx >= len(table_data):
        return None, None

    # Extract subject data and ID
    subject_data = table_data[absolute_row_idx]
    subject_id = subject_data.get("subject_id")

    return subject_data, subject_id


def build_formatted_column_names():
    """
    Build formatted column name mappings for table display.

    Used across table callbacks to ensure consistent column naming
    and formatting across different table views.

    Returns:
        dict: Mapping of column IDs to formatted display names
    """
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
        "curriculum_name": "Curriculum",
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
        "water_in_session_foraging": "Water In - Session\n(Foraging)",
        "water_in_session_manual": "Water In - Session\n(Manual)",
        "water_in_session_total": "Water In - Session\n(Total)",
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

    # Add feature - specific column names
    features_config = {
        "finished_trials": False,
        "ignore_rate": True,
        "total_trials": False,
        "foraging_performance": False,
        "abs(bias_naive)": True,
    }

    for feature in features_config.keys():
        feature_display = (
            feature.replace("_", " ").replace("abs(", "|").replace(")", "|").title()
        )
        formatted_column_names[f"{feature}_percentile"] = (
            f"{feature_display}\nStrata %ile"
        )
        formatted_column_names[f"{feature}_category"] = f"{feature_display}\nAlert"
        formatted_column_names[f"{feature}_processed"] = f"{feature_display}\nProcessed"
        formatted_column_names[f"{feature}_session_percentile"] = (
            f"{feature_display}\nSession %ile"
        )
        formatted_column_names[f"{feature}_processed_rolling_avg"] = (
            f"{feature_display}\nRolling Avg"
        )

    # Add overall percentile columns
    formatted_column_names["session_overall_percentile"] = "Session Overall\nPercentile"
    formatted_column_names["overall_percentile"] = "Strata Overall\nPercentile"

    return formatted_column_names


def validate_callback_inputs(*inputs):
    """
    Validate callback inputs and provide default handling.

    Used to ensure callback robustness across modules by providing
    consistent input validation and default value handling.

    Args:
        *inputs: Variable number of callback inputs to validate

    Returns:
        tuple: Validated inputs with defaults applied where needed
    """
    validated = []
    for input_val in inputs:
        if input_val is None:
            validated.append(None)
        elif isinstance(input_val, list) and len(input_val) == 0:
            validated.append(None)
        else:
            validated.append(input_val)
    return tuple(validated)


__all__ = [
    "Input",
    "Output",
    "State",
    "callback",
    "ALL",
    "ctx",
    "clientside_callback",
    "html",
    "dcc",
    "dbc",
    "pd",
    "go",
    "json",
    "datetime",
    "timedelta",
    "app_utils",
    "app_dataframe",
    "app_filter",
    "session_card",
    "image_loader",
    "compact_info",
    "subject_timeseries",
    "subject_percentile_timeseries",
    "percentile_heatmap",
    "app_tooltip",
    "format_multi_value",
    "create_filter_badge",
    "extract_highlighted_session",
    "get_alert_color",
    "throttle_callback_execution",
    "safe_extract_subject_data",
    "build_formatted_column_names",
    "validate_callback_inputs",
]
