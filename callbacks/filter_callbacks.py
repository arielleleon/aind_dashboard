# Import shared utilities (replaces multiple individual imports)
# Import new filtering utilities
from app_utils.filter_utils import apply_all_filters
from callbacks.shared_callback_utils import (
    ALL,
    Input,
    Output,
    State,
    app_dataframe,
    app_filter,
    app_utils,
    callback,
    create_filter_badge,
    ctx,
    datetime,
    dbc,
    format_multi_value,
    html,
    pd,
    timedelta,
)

# No need to re-create instances - they're shared from utilities
# app_dataframe = AppDataFrame(app_utils=app_utils)  # Removed - using shared instance
# app_filter = AppFilter()  # Removed - using shared instance

# Helper functions moved to shared_callback_utils.py
# def format_multi_value(value): ... # Removed - using shared function
# def create_filter_badge(label, filter_type, filter_value): ... # Removed - using shared function


# Callback to update active filters display and count
@callback(
    [Output("active-filters", "children"), Output("filter-count", "children")],
    [
        Input("time-window-filter", "value"),
        Input("stage-filter", "value"),
        Input("curriculum-filter", "value"),
        Input("rig-filter", "value"),
        Input("trainer-filter", "value"),
        Input("pi-filter", "value"),
        Input("sort-option", "value"),
        Input("alert-category-filter", "value"),
        Input("subject-id-filter", "value"),
        Input("clear-filters", "n_clicks"),
    ],
)
def update_active_filters(
    time_window_value,
    stage_value,
    curriculum_value,
    rig_value,
    trainer_value,
    pi_value,
    sort_option,
    alert_category,
    subject_id_value,
    clear_clicks,
):
    """Update the display of active filters and count"""
    # Initialize active filters
    active_filters = []

    # Reset if clear button was clicked
    if ctx.triggered_id == "clear-filters":
        return [], ""

    # Get time window label for the selected value
    time_window_label = next(
        (
            opt["label"]
            for opt in app_filter.time_window_options
            if opt["value"] == time_window_value
        ),
        f"Last {time_window_value} days",
    )

    # Add sort option if not default
    if sort_option != "none":
        sort_label = next(
            (
                opt["label"]
                for opt in app_filter.sort_options
                if opt["value"] == sort_option
            ),
            "Sorted",
        )
        active_filters.append(
            create_filter_badge(f"Sort: {sort_label}", "sort-option", sort_option)
        )

    # Add alert category filter if not "all"
    if alert_category != "all":
        alert_label = next(
            (
                opt["label"]
                for opt in app_filter.alert_category_options
                if opt["value"] == alert_category
            ),
            alert_category,
        )
        active_filters.append(
            create_filter_badge(
                f"Alert: {alert_label}", "alert-category-filter", alert_category
            )
        )

    # Add time window filter badge (always shown)
    active_filters.append(
        create_filter_badge(
            f"Time Window: {time_window_label}", "time-window-filter", time_window_value
        )
    )

    # Add subject ID filter
    if subject_id_value:
        formatted_value = format_multi_value(subject_id_value)
        key_value = (
            subject_id_value[0]
            if isinstance(subject_id_value, list)
            else subject_id_value
        )
        active_filters.append(
            create_filter_badge(
                f"Subject ID: {formatted_value}", "subject-id-filter", key_value
            )
        )

    # Add each active filter - handling multi-select
    if stage_value:
        formatted_value = format_multi_value(stage_value)
        key_value = stage_value[0] if isinstance(stage_value, list) else stage_value
        active_filters.append(
            create_filter_badge(f"Stage: {formatted_value}", "stage-filter", key_value)
        )

    if curriculum_value:
        formatted_value = format_multi_value(curriculum_value)
        key_value = (
            curriculum_value[0]
            if isinstance(curriculum_value, list)
            else curriculum_value
        )
        active_filters.append(
            create_filter_badge(
                f"Curriculum: {formatted_value}", "curriculum-filter", key_value
            )
        )

    if rig_value:
        formatted_value = format_multi_value(rig_value)
        key_value = rig_value[0] if isinstance(rig_value, list) else rig_value
        active_filters.append(
            create_filter_badge(f"Rig: {formatted_value}", "rig-filter", key_value)
        )

    if trainer_value:
        formatted_value = format_multi_value(trainer_value)
        key_value = (
            trainer_value[0] if isinstance(trainer_value, list) else trainer_value
        )
        active_filters.append(
            create_filter_badge(
                f"Trainer: {formatted_value}", "trainer-filter", key_value
            )
        )

    if pi_value:
        formatted_value = format_multi_value(pi_value)
        key_value = pi_value[0] if isinstance(pi_value, list) else pi_value
        active_filters.append(
            create_filter_badge(f"PI: {formatted_value}", "pi-filter", key_value)
        )

    # Return active filters and count
    filter_count = len(active_filters)
    count_display = filter_count if filter_count > 0 else ""

    return active_filters, count_display


# Callback to handle filter badge removal
@callback(
    [
        Output("time-window-filter", "value"),
        Output("stage-filter", "value"),
        Output("curriculum-filter", "value"),
        Output("rig-filter", "value"),
        Output("trainer-filter", "value"),
        Output("pi-filter", "value"),
        Output("sort-option", "value"),
        Output("alert-category-filter", "value"),
        Output("subject-id-filter", "value"),
    ],
    [
        Input({"type": "remove-filter", "index": ALL}, "n_clicks"),
        Input("clear-filters", "n_clicks"),
    ],
    [
        State("time-window-filter", "value"),
        State({"type": "remove-filter", "index": ALL}, "id"),
        State("stage-filter", "value"),
        State("curriculum-filter", "value"),
        State("rig-filter", "value"),
        State("trainer-filter", "value"),
        State("pi-filter", "value"),
        State("sort-option", "value"),
        State("alert-category-filter", "value"),
        State("subject-id-filter", "value"),
    ],
    prevent_initial_call=True,
)
def remove_filter(
    remove_clicks,
    clear_clicks,
    time_window_value,
    remove_ids,
    stage_value,
    curriculum_value,
    rig_value,
    trainer_value,
    pi_value,
    sort_value,
    alert_category_value,
    subject_id_value,
):
    """Handle removal of individual filter badges or clearing all filters"""
    # Initialize return values with current state
    outputs = [
        time_window_value,
        stage_value,
        curriculum_value,
        rig_value,
        trainer_value,
        pi_value,
        sort_value,
        alert_category_value,
        subject_id_value,
    ]

    # Add debugging to track clear button clicks
    print(f"remove_filter callback triggered. ctx.triggered_id: {ctx.triggered_id}")
    if clear_clicks:
        print(f"Clear button clicks: {clear_clicks}")

    # If clear button was clicked, clear all filters
    if ctx.triggered_id == "clear-filters":
        print("🧹 CLEAR ALL FILTERS button pressed - resetting all values")
        return [30, None, None, None, None, None, "none", "all", None]

    # Find which filter was clicked to be removed
    for i, clicks in enumerate(remove_clicks):
        if clicks:
            # Get the filter info that needs to be removed
            filter_id = remove_ids[i]["index"]
            filter_type, filter_value = filter_id.split(":", 1)

            print(f"Removing individual filter: {filter_type} = {filter_value}")

            # Don't allow removing time window filter via badge (keep as is)
            if filter_type == "time-window-filter":
                continue

            # Clear the corresponding filter - properly handles multi-select
            if filter_type == "stage-filter":
                outputs[1] = None  # Clear entire stage filter
            elif filter_type == "curriculum-filter":
                outputs[2] = None  # Clear entire curriculum filter
            elif filter_type == "rig-filter":
                outputs[3] = None  # Clear entire rig filter
            elif filter_type == "trainer-filter":
                outputs[4] = None  # Clear entire trainer filter
            elif filter_type == "pi-filter":
                outputs[5] = None  # Clear entire PI filter
            elif filter_type == "sort-option":
                outputs[6] = "none"  # Reset to default sort
            elif filter_type == "alert-category-filter":
                outputs[7] = "all"  # Reset to show all alerts
            elif filter_type == "subject-id-filter":
                outputs[8] = None  # Clear entire subject ID filter

            # Only process one removal at a time
            break

    return outputs


# Callback to update data table based on filters
@callback(
    Output("session-table", "data"),
    [
        Input("time-window-filter", "value"),
        Input("stage-filter", "value"),
        Input("curriculum-filter", "value"),
        Input("rig-filter", "value"),
        Input("trainer-filter", "value"),
        Input("pi-filter", "value"),
        Input("sort-option", "value"),
        Input("alert-category-filter", "value"),
        Input("subject-id-filter", "value"),
        Input("clear-filters", "n_clicks"),
    ],
)
def update_table_data(
    time_window_value,
    stage_value,
    curriculum_value,
    rig_value,
    trainer_value,
    pi_value,
    sort_option,
    alert_category,
    subject_id_value,
    clear_clicks,
):
    """Update the session table data based on active filters"""
    print(f"Updating table with time window: {time_window_value} days")

    # Use UI-optimized table display data from the shared app_utils instance
    formatted_df = pd.DataFrame(app_utils.get_table_display_data(use_cache=True))

    if formatted_df.empty:
        print(" No table display data found, falling back to formatted data cache")
        # Fallback to formatted data cache if UI cache is empty
        if app_utils._cache["formatted_data"] is not None:
            formatted_df = app_utils._cache["formatted_data"].copy()
        else:
            print("No cached data found, triggering format_dataframe")
            df = app_utils.get_session_data(use_cache=True)
            formatted_df = app_dataframe.format_dataframe(df)

    # Apply all filters using the extracted business logic functions
    filtered_df = apply_all_filters(
        df=formatted_df,
        time_window_value=time_window_value,
        stage_value=stage_value,
        curriculum_value=curriculum_value,
        rig_value=rig_value,
        trainer_value=trainer_value,
        pi_value=pi_value,
        sort_option=sort_option,
        alert_category=alert_category,
        subject_id_value=subject_id_value,
    )

    # Convert to records for datatable
    return filtered_df.to_dict("records")
