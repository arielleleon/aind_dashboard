# Column groups configuration for collapsible DataTable functionality

COLUMN_GROUPS = {
    "core": {
        "label": "Core Info",
        "collapsible": False,  # Always visible
        "icon": "fas fa-star",
        "columns": [
            "subject_id",
            "session_date",
            "session",
            "PI",
            "trainer",
            "rig",
            "water_day_total",
            "total_trials",
            "finished_trials",
            "ignore_rate",
            "foraging_performance",
            "abs(bias_naive)",
        ],
    },
    "alerts": {
        "label": "Alerts & Categories",
        "collapsible": True,
        "default_expanded": False,
        "icon": "fas fa-exclamation-triangle",
        "columns": [
            "strata_abbr",
            "combined_alert",
            "percentile_category",
            "threshold_alert",
            "total_sessions_alert",
            "stage_sessions_alert",
            "water_day_total_alert",
            "ns_reason",
        ],
    },
    "percentiles_and_confidence": {
        "label": "Percentiles & Wilson Confidence Intervals",
        "collapsible": True,
        "default_expanded": False,
        "icon": "fas fa-chart-line",
        "columns": [
            # Core percentile columns (always shown when group is expanded)
            "session_overall_percentile",
            "session_overall_percentile_rolling_avg",
            # Overall percentile Wilson CIs and certainty
            "session_overall_percentile_ci_lower",
            "session_overall_percentile_ci_upper",
            "session_overall_percentile_certainty",
            # PHASE 2: Outlier Detection Information
            "outlier_weight",
            "is_outlier",
            # Feature-specific percentile columns with Wilson CIs and certainty - will be populated dynamically
            "finished_trials_session_percentile",
            "finished_trials_session_percentile_ci_lower",
            "finished_trials_session_percentile_ci_upper",
            "finished_trials_certainty",
            "ignore_rate_session_percentile",
            "ignore_rate_session_percentile_ci_lower",
            "ignore_rate_session_percentile_ci_upper",
            "ignore_rate_certainty",
            "total_trials_session_percentile",
            "total_trials_session_percentile_ci_lower",
            "total_trials_session_percentile_ci_upper",
            "total_trials_certainty",
            "foraging_performance_session_percentile",
            "foraging_performance_session_percentile_ci_lower",
            "foraging_performance_session_percentile_ci_upper",
            "foraging_performance_certainty",
            "abs(bias_naive)_session_percentile",
            "abs(bias_naive)_session_percentile_ci_lower",
            "abs(bias_naive)_session_percentile_ci_upper",
            "abs(bias_naive)_certainty",
        ],
    },
    "processed_features": {
        "label": "Processed Features & Rolling Averages",
        "collapsible": True,
        "default_expanded": False,
        "icon": "fas fa-cogs",
        "columns": [
            # Rolling averages (used for percentile calculation)
            "finished_trials_processed_rolling_avg",
            "ignore_rate_processed_rolling_avg",
            "total_trials_processed_rolling_avg",
            "foraging_performance_processed_rolling_avg",
            "abs(bias_naive)_processed_rolling_avg",
            "session_overall_rolling_avg",
            # Processed features (standardized within strata)
            "finished_trials_processed",
            "ignore_rate_processed",
            "total_trials_processed",
            "foraging_performance_processed",
            "abs(bias_naive)_processed",
        ],
    },
    "experimental_metadata": {
        "label": "Experimental Metadata",
        "collapsible": True,
        "default_expanded": False,
        "icon": "fas fa-flask",
        "columns": ["strata", "current_stage_actual", "curriculum_name"],
    },
    "weight_water": {
        "label": "Weight & Water",
        "collapsible": True,
        "default_expanded": False,
        "icon": "fas fa-tint",
        "columns": [
            "base_weight",
            "target_weight",
            "target_weight_ratio",
            "weight_after",
            "weight_after_ratio",
            "water_in_session_foraging",
            "water_in_session_manual",
            "water_in_session_total",
            "water_after_session",
        ],
    },
    "behavioral_details": {
        "label": "Behavioral Details",
        "collapsible": True,
        "default_expanded": False,
        "icon": "fas fa-brain",
        "columns": [
            "reward_volume_left_mean",
            "reward_volume_right_mean",
            "reaction_time_median",
            "reaction_time_mean",
            "early_lick_rate",
            "invalid_lick_ratio",
            "finished_rate",
            "double_dipping_rate_finished_trials",
            "double_dipping_rate_finished_reward_trials",
            "double_dipping_rate_finished_noreward_trials",
            "lick_consistency_mean_finished_trials",
            "lick_consistency_mean_finished_reward_trials",
            "lick_consistency_mean_finished_noreward_trials",
            "avg_trial_length_in_seconds",
        ],
    },
    "autowater": {
        "label": "Autowater Metrics",
        "collapsible": True,
        "default_expanded": False,
        "icon": "fas fa-faucet",
        "columns": [
            "total_trials_with_autowater",
            "finished_trials_with_autowater",
            "finished_rate_with_autowater",
            "ignore_rate_with_autowater",
            "autowater_collected",
            "autowater_ignored",
            "water_day_total_last_session",
            "water_after_session_last_session",
        ],
    },
}


def get_default_visible_columns():
    """Get default visible columns (core only)"""
    visible = []
    visible.extend(COLUMN_GROUPS["core"]["columns"])
    return visible


def get_columns_for_groups(expanded_groups):
    """Get columns for expanded groups"""
    columns = []

    # Always include core columns
    columns.extend(COLUMN_GROUPS["core"]["columns"])

    # Add columns from expanded groups
    for group_name in expanded_groups:
        if group_name in COLUMN_GROUPS and COLUMN_GROUPS[group_name]["collapsible"]:
            columns.extend(COLUMN_GROUPS[group_name]["columns"])

    return columns
