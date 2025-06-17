from typing import Any, Dict

import pandas as pd

from app_utils.simple_logger import get_logger

# Initialize logger for filter operations
logger = get_logger("filters")


def apply_time_window_filter(df: pd.DataFrame, time_window_value: int) -> pd.DataFrame:
    """
    Apply time window filtering to show sessions from last N days only

    Parameters:
        df: DataFrame to filter
        time_window_value: Number of days to include (None = no filter)

    Returns:
        Filtered DataFrame
    """
    if df.empty or time_window_value is None:
        return df

    if time_window_value <= 0:
        return df

    # Get the current date and calculate the cutoff
    current_date = pd.Timestamp.now()
    cutoff_date = current_date - pd.Timedelta(days=time_window_value)

    # Ensure session_date is datetime
    if "session_date" in df.columns:
        df = df.copy()
        df["session_date"] = pd.to_datetime(df["session_date"])

        # Filter sessions within the time window
        filtered_df = df[df["session_date"] >= cutoff_date]

        # Remove duplicate subjects (keep most recent session)
        if not filtered_df.empty:
            filtered_df = (
                filtered_df.sort_values("session_date")
                .groupby("subject_id")
                .last()
                .reset_index()
            )

        logger.info(
            f"Applied {time_window_value} day window filter: {len(filtered_df)} subjects"
        )
        return filtered_df

    return df


def _apply_subject_id_filter(df: pd.DataFrame, subject_id_value: Any) -> pd.DataFrame:
    """
    Apply subject ID filtering logic

    This function handles both single subject ID and list of subject IDs,
    converting them to strings for consistent comparison.

    Parameters:
        df: DataFrame to filter
        subject_id_value: Subject ID value(s) to filter by

    Returns:
        Filtered DataFrame
    """
    if not subject_id_value:
        return df

    if isinstance(subject_id_value, list):
        # Convert subject IDs to strings for consistent comparison
        subject_id_strings = [str(sid) for sid in subject_id_value]
        filtered_df = df[df["subject_id"].astype(str).isin(subject_id_strings)]
        logger.info(
            f"Applied subject ID filter for {len(subject_id_value)} subjects: {len(filtered_df)} subjects remaining"
        )
    else:
        # Single subject ID
        subject_id_string = str(subject_id_value)
        filtered_df = df[df["subject_id"].astype(str) == subject_id_string]
        logger.info(
            f"Applied subject ID filter for {subject_id_string}: {len(filtered_df)} subjects remaining"
        )

    return filtered_df


def _apply_single_filter(
    df: pd.DataFrame, column_name: str, filter_value: Any
) -> pd.DataFrame:
    """
    Apply a single filter to a specific column

    This function handles both single values and lists of values for filtering.

    Parameters:
        df: DataFrame to filter
        column_name: Name of the column to filter on
        filter_value: Value(s) to filter by

    Returns:
        Filtered DataFrame
    """
    if not filter_value:
        return df

    if isinstance(filter_value, list):
        return df[df[column_name].isin(filter_value)]
    else:
        return df[df[column_name] == filter_value]


def apply_multi_select_filters(
    df: pd.DataFrame, filter_configs: Dict[str, Any]
) -> pd.DataFrame:
    """
    Apply multiple filters (stage, curriculum, rig, trainer, PI, subject_id)

    Parameters:
        df: DataFrame to filter
        filter_configs: Dictionary with filter keys and values
            Expected keys: 'stage', 'curriculum', 'rig', 'trainer', 'pi', 'subject_id'

    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df

    filtered_df = df.copy()

    # Apply subject ID filter (special handling for string conversion)
    subject_id_value = filter_configs.get("subject_id")
    filtered_df = _apply_subject_id_filter(filtered_df, subject_id_value)

    # Define mapping of filter keys to column names
    filter_mappings = {
        "stage": "current_stage_actual",
        "curriculum": "curriculum_name",
        "rig": "rig",
        "trainer": "trainer",
        "pi": "PI",
    }

    # Apply each filter using the helper function
    for filter_key, column_name in filter_mappings.items():
        filter_value = filter_configs.get(filter_key)
        filtered_df = _apply_single_filter(filtered_df, column_name, filter_value)

    return filtered_df


def apply_alert_category_filter(df: pd.DataFrame, alert_category: str) -> pd.DataFrame:
    """
    Apply alert category filtering using AlertCoordinator business logic

    This function now delegates to the AlertCoordinator which orchestrates
    the complex alert filtering logic through the AlertService.

    Parameters:
        df: DataFrame to filter
        alert_category: Alert category to filter by ('all', 'T', 'NS', 'B', 'G', 'SB', 'SG')

    Returns:
        Filtered DataFrame
    """
    if df.empty or alert_category == "all":
        return df

    try:
        # Try to get app_utils from the common import pattern
        try:
            from callbacks.shared_callback_utils import app_utils

            # Get or create alert coordinator
            if (
                hasattr(app_utils, "alert_coordinator")
                and app_utils.alert_coordinator is not None
            ):
                alert_coordinator = app_utils.alert_coordinator
                # Ensure alert service is initialized
                if alert_coordinator.alert_service is None:
                    alert_coordinator.initialize_alert_service(app_utils)

                # Use the new business logic
                return alert_coordinator.filter_by_alert_category(df, alert_category)
            else:
                # Fallback to old logic if coordinator not available
                logger.info(" Alert coordinator not available, using fallback logic")
                return _apply_alert_category_filter_fallback(df, alert_category)

        except ImportError:
            # If shared_callback_utils not available, use fallback
            logger.info(" Cannot import app_utils, using fallback logic")
            return _apply_alert_category_filter_fallback(df, alert_category)

    except Exception as e:
        logger.error(f" Error in alert category filtering: {str(e)}")
        logger.info(" Using fallback logic")
        return _apply_alert_category_filter_fallback(df, alert_category)


def _apply_alert_category_filter_fallback(
    df: pd.DataFrame, alert_category: str
) -> pd.DataFrame:
    """
    Fallback alert category filtering logic for when AlertCoordinator is not available

    This maintains the original complex logic as a backup to ensure the application
    continues to work even if the alert modules are not properly initialized.

    Parameters:
        df: DataFrame to filter
        alert_category: Alert category to filter by

    Returns:
        Filtered DataFrame
    """
    if alert_category == "T":
        # Original threshold alert logic as fallback
        threshold_mask = (
            # Overall threshold alert column set to 'T'
            (df.get("threshold_alert", pd.Series(dtype="object")) == "T")
            # Individual threshold alerts contain "T |" pattern
            | (
                df.get("total_sessions_alert", pd.Series(dtype="object")).str.contains(
                    r"T \|", na=False
                )
            )
            | (
                df.get("stage_sessions_alert", pd.Series(dtype="object")).str.contains(
                    r"T \|", na=False
                )
            )
            | (
                df.get("water_day_total_alert", pd.Series(dtype="object")).str.contains(
                    r"T \|", na=False
                )
            )
        )

        before_count = len(df)
        filtered_df = df[threshold_mask]
        after_count = len(filtered_df)

        logger.info(
            f"Fallback threshold filter applied: {before_count} → {after_count} subjects"
        )
        return filtered_df

    elif alert_category == "NS":
        # Not Scored subjects
        percentile_col = (
            "overall_percentile_category"
            if "overall_percentile_category" in df.columns
            else "percentile_category"
        )
        return df[df[percentile_col] == "NS"]
    else:
        # Percentile category filtering (B, G, SB, SG)
        percentile_col = (
            "overall_percentile_category"
            if "overall_percentile_category" in df.columns
            else "percentile_category"
        )
        return df[df[percentile_col] == alert_category]


def apply_sorting_logic(df: pd.DataFrame, sort_option: str) -> pd.DataFrame:
    """
    Apply sorting logic to the filtered dataframe

    This function handles the different sorting options for the main data table,
    with fallback logic for different column name variations.

    Parameters:
        df: DataFrame to sort
        sort_option: Sort option string

    Returns:
        Sorted DataFrame
    """
    if df.empty or sort_option is None:
        return df

    try:
        if sort_option == "overall_percentile_asc":
            if "session_overall_percentile" in df.columns:
                df = df.sort_values("session_overall_percentile", ascending=True)
            elif "overall_percentile" in df.columns:
                df = df.sort_values("overall_percentile", ascending=True)
            else:
                logger.warning("No overall percentile column found for sorting")

        elif sort_option == "overall_percentile_desc":
            if "session_overall_percentile" in df.columns:
                df = df.sort_values("session_overall_percentile", ascending=False)
            elif "overall_percentile" in df.columns:
                df = df.sort_values("overall_percentile", ascending=False)
            else:
                logger.warning("No overall percentile column found for sorting")

    except Exception as e:
        logger.error(f"Error applying sort option '{sort_option}': {str(e)}")

    return df


def apply_all_filters(
    df: pd.DataFrame,
    time_window_value: int,
    stage_value: Any,
    curriculum_value: Any,
    rig_value: Any,
    trainer_value: Any,
    pi_value: Any,
    sort_option: str,
    alert_category: str,
    subject_id_value: Any,
) -> pd.DataFrame:
    """
    Apply all filters in sequence using the extracted filtering functions

    This function orchestrates all filtering operations in the correct order:
    1. Time window filtering
    2. Multi-select filters (stage, curriculum, rig, trainer, PI, subject_id)
    3. Alert category filtering
    4. Sorting

    Parameters:
        df: Base DataFrame to filter
        time_window_value: Number of days for time window filter
        stage_value: Stage filter value(s)
        curriculum_value: Curriculum filter value(s)
        rig_value: Rig filter value(s)
        trainer_value: Trainer filter value(s)
        pi_value: PI filter value(s)
        sort_option: Sort option
        alert_category: Alert category filter
        subject_id_value: Subject ID filter value(s)

    Returns:
        Fully filtered and sorted DataFrame
    """
    # Apply filters using business logic functions
    filtered_df = apply_time_window_filter(df, time_window_value)

    filtered_df = apply_multi_select_filters(
        filtered_df,
        {
            "stage": stage_value,
            "curriculum": curriculum_value,
            "rig": rig_value,
            "trainer": trainer_value,
            "pi": pi_value,
            "subject_id": subject_id_value,
        },
    )

    filtered_df = apply_alert_category_filter(filtered_df, alert_category)
    filtered_df = apply_sorting_logic(filtered_df, sort_option)

    # Count percentile categories for debugging - use the correct column name
    percentile_col = (
        "overall_percentile_category"
        if "overall_percentile_category" in filtered_df.columns
        else "percentile_category"
    )
    if percentile_col in filtered_df.columns:
        percentile_counts = filtered_df[percentile_col].value_counts().to_dict()
        logger.info(f"Percentile categories: {percentile_counts}")
    else:
        logger.info(
            f"No percentile category column found. Available columns: {list(filtered_df.columns)}"
        )
    logger.info(f"Applied sorting: {sort_option}")

    return filtered_df
