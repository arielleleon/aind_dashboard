# Import shared utilities (replaces multiple individual imports)
from callbacks.shared_callback_utils import (
    Input, Output, State, callback, ALL, ctx, html, dbc, pd, datetime, timedelta,
    app_utils, app_dataframe, app_filter,
    format_multi_value, create_filter_badge
)

# No need to re-create instances - they're shared from utilities
# app_dataframe = AppDataFrame(app_utils=app_utils)  # Removed - using shared instance
# app_filter = AppFilter()  # Removed - using shared instance

# Helper functions moved to shared_callback_utils.py
# def format_multi_value(value): ... # Removed - using shared function
# def create_filter_badge(label, filter_type, filter_value): ... # Removed - using shared function

# Callback to update active filters display and count
@callback(
    [Output("active-filters", "children"),
     Output("filter-count", "children")],
    [Input("time-window-filter", "value"),
     Input("stage-filter", "value"),
     Input("curriculum-filter", "value"),
     Input("rig-filter", "value"),
     Input("trainer-filter", "value"),
     Input("pi-filter", "value"),
     Input("sort-option", "value"),
     Input("alert-category-filter", "value"),
     Input("subject-id-filter", "value"),
     Input("clear-filters", "n_clicks")]
)
def update_active_filters(
    time_window_value, stage_value, curriculum_value, rig_value, trainer_value, pi_value, sort_option, 
    alert_category, subject_id_value, clear_clicks
):
    """Update the display of active filters and count"""
    # Initialize active filters
    active_filters = []
    
    # Reset if clear button was clicked
    if ctx.triggered_id == "clear-filters":
        return [], ""
    
    # Get time window label for the selected value
    time_window_label = next((opt["label"] for opt in app_filter.time_window_options
                             if opt["value"] == time_window_value), f"Last {time_window_value} days")
    
    # Add sort option if not default
    if sort_option != "none":
        sort_label = next((opt["label"] for opt in app_filter.sort_options 
                         if opt["value"] == sort_option), "Sorted")
        active_filters.append(
            create_filter_badge(f"Sort: {sort_label}", "sort-option", sort_option)
        )
    
    # Add alert category filter if not "all"
    if alert_category != "all":
        alert_label = next((opt["label"] for opt in app_filter.alert_category_options
                          if opt["value"] == alert_category), alert_category)
        active_filters.append(
            create_filter_badge(f"Alert: {alert_label}", "alert-category-filter", alert_category)
        )
    
    # Add time window filter badge (always shown)
    active_filters.append(
        create_filter_badge(f"Time Window: {time_window_label}", "time-window-filter", time_window_value)
    )
    
    # Add subject ID filter
    if subject_id_value:
        formatted_value = format_multi_value(subject_id_value)
        key_value = subject_id_value[0] if isinstance(subject_id_value, list) else subject_id_value
        active_filters.append(
            create_filter_badge(f"Subject ID: {formatted_value}", "subject-id-filter", key_value)
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
        key_value = curriculum_value[0] if isinstance(curriculum_value, list) else curriculum_value
        active_filters.append(
            create_filter_badge(f"Curriculum: {formatted_value}", "curriculum-filter", key_value)
        )
    
    if rig_value:
        formatted_value = format_multi_value(rig_value)
        key_value = rig_value[0] if isinstance(rig_value, list) else rig_value
        active_filters.append(
            create_filter_badge(f"Rig: {formatted_value}", "rig-filter", key_value)
        )
    
    if trainer_value:
        formatted_value = format_multi_value(trainer_value)
        key_value = trainer_value[0] if isinstance(trainer_value, list) else trainer_value
        active_filters.append(
            create_filter_badge(f"Trainer: {formatted_value}", "trainer-filter", key_value)
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
    [Output("time-window-filter", "value"),
     Output("stage-filter", "value"),
     Output("curriculum-filter", "value"),
     Output("rig-filter", "value"),
     Output("trainer-filter", "value"),
     Output("pi-filter", "value"),
     Output("sort-option", "value"),
     Output("alert-category-filter", "value"),
     Output("subject-id-filter", "value")],
    [Input({"type": "remove-filter", "index": ALL}, "n_clicks"),
     Input("clear-filters", "n_clicks")],
    [State("time-window-filter", "value"),
     State({"type": "remove-filter", "index": ALL}, "id"),
     State("stage-filter", "value"),
     State("curriculum-filter", "value"),
     State("rig-filter", "value"),
     State("trainer-filter", "value"),
     State("pi-filter", "value"),
     State("sort-option", "value"),
     State("alert-category-filter", "value"),
     State("subject-id-filter", "value")],
    prevent_initial_call=True
)
def remove_filter(remove_clicks, clear_clicks, time_window_value, remove_ids, 
                 stage_value, curriculum_value, rig_value, trainer_value, 
                 pi_value, sort_value, alert_category_value, subject_id_value):
    """Handle removal of individual filter badges or clearing all filters"""
    # Initialize return values with current state
    outputs = [time_window_value, stage_value, curriculum_value, rig_value, trainer_value, pi_value, 
              sort_value, alert_category_value, subject_id_value]
    
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
                outputs[7] = "all"   # Reset to show all alerts
            elif filter_type == "subject-id-filter":
                outputs[8] = None  # Clear entire subject ID filter
            
            # Only process one removal at a time
            break
    
    return outputs

# Callback to update data table based on filters
@callback(
    Output("session-table", "data"),
    [Input("time-window-filter", "value"),
     Input("stage-filter", "value"),
     Input("curriculum-filter", "value"),
     Input("rig-filter", "value"),
     Input("trainer-filter", "value"),
     Input("pi-filter", "value"),
     Input("sort-option", "value"),
     Input("alert-category-filter", "value"),
     Input("subject-id-filter", "value"),
     Input("clear-filters", "n_clicks")]
)
def update_table_data(time_window_value, stage_value, curriculum_value, 
                     rig_value, trainer_value, pi_value, sort_option, 
                     alert_category, subject_id_value, clear_clicks):
    """Update the session table data based on active filters"""
    print(f"Updating table with time window: {time_window_value} days")
    
    # Use UI-optimized table display data from the shared app_utils instance
    formatted_df = pd.DataFrame(app_utils.get_table_display_data(use_cache=True))
    
    if formatted_df.empty:
        print("⚠️  No table display data found, falling back to formatted data cache")
        # Fallback to formatted data cache if UI cache is empty
        if app_utils._cache['formatted_data'] is not None:
            formatted_df = app_utils._cache['formatted_data'].copy()
        else:
            print("🔄 No cached data found, triggering format_dataframe")
            df = app_utils.get_session_data(use_cache=True)
            formatted_df = app_dataframe.format_dataframe(df)
    
    print(f"📊 Starting with {len(formatted_df)} subjects before filtering")

    # Apply time window filter directly to the session_date column
    if time_window_value:
        reference_date = formatted_df['session_date'].max()
        start_date = reference_date - timedelta(days=time_window_value)
        # Filter to sessions within time window
        time_filtered = formatted_df[formatted_df['session_date'] >= start_date]
        # Get most recent session for each subject in the window
        time_filtered = time_filtered.sort_values('session_date', ascending=False)
        formatted_df = time_filtered.drop_duplicates(subset=['subject_id'], keep='first')
        print(f"Applied {time_window_value} day window filter: {len(formatted_df)} subjects")

    # Apply subject ID filter if specified
    if subject_id_value:
        if isinstance(subject_id_value, list):
            # Convert subject IDs to strings for consistent comparison
            subject_id_strings = [str(sid) for sid in subject_id_value]
            formatted_df = formatted_df[formatted_df["subject_id"].astype(str).isin(subject_id_strings)]
            print(f"Applied subject ID filter for {len(subject_id_value)} subjects: {len(formatted_df)} subjects remaining")
        else:
            # Single subject ID
            subject_id_string = str(subject_id_value)
            formatted_df = formatted_df[formatted_df["subject_id"].astype(str) == subject_id_string]
            print(f"Applied subject ID filter for {subject_id_string}: {len(formatted_df)} subjects remaining")

    # Apply each filter if it has a value - handling multi-select
    if stage_value:
        if isinstance(stage_value, list):
            formatted_df = formatted_df[formatted_df["current_stage_actual"].isin(stage_value)]
        else:
            formatted_df = formatted_df[formatted_df["current_stage_actual"] == stage_value]
    
    if curriculum_value:
        if isinstance(curriculum_value, list):
            formatted_df = formatted_df[formatted_df["curriculum_name"].isin(curriculum_value)]
        else:
            formatted_df = formatted_df[formatted_df["curriculum_name"] == curriculum_value]
    
    if rig_value:
        if isinstance(rig_value, list):
            formatted_df = formatted_df[formatted_df["rig"].isin(rig_value)]
        else:
            formatted_df = formatted_df[formatted_df["rig"] == rig_value]
    
    if trainer_value:
        if isinstance(trainer_value, list):
            formatted_df = formatted_df[formatted_df["trainer"].isin(trainer_value)]
        else:
            formatted_df = formatted_df[formatted_df["trainer"] == trainer_value]
    
    if pi_value:
        if isinstance(pi_value, list):
            formatted_df = formatted_df[formatted_df["PI"].isin(pi_value)]
        else:
            formatted_df = formatted_df[formatted_df["PI"] == pi_value]
    
    # Apply alert category filter if selected
    if alert_category != "all":
        if alert_category == "T":
            # Filter for threshold alerts specifically
            print("🔍 DEBUG: Checking threshold alert values...")
            
            # Debug: Check what threshold alert values we have
            if 'threshold_alert' in formatted_df.columns:
                threshold_values = formatted_df['threshold_alert'].value_counts()
                print(f"  threshold_alert values: {threshold_values.to_dict()}")
            
            if 'total_sessions_alert' in formatted_df.columns:
                total_values = formatted_df['total_sessions_alert'].value_counts()
                print(f"  total_sessions_alert values: {total_values.to_dict()}")
            
            if 'stage_sessions_alert' in formatted_df.columns:
                stage_values = formatted_df['stage_sessions_alert'].value_counts()
                print(f"  stage_sessions_alert values: {stage_values.to_dict()}")
            
            if 'water_day_total_alert' in formatted_df.columns:
                water_values = formatted_df['water_day_total_alert'].value_counts()
                print(f"  water_day_total_alert values: {water_values.to_dict()}")
            
            # Match the actual threshold alert patterns
            threshold_mask = (
                # Overall threshold alert column set to 'T'
                (formatted_df.get("threshold_alert", pd.Series(dtype='object')) == "T") |
                # Individual threshold alerts contain "T |" pattern
                (formatted_df.get("total_sessions_alert", pd.Series(dtype='object')).str.contains(r'T \|', na=False)) |
                (formatted_df.get("stage_sessions_alert", pd.Series(dtype='object')).str.contains(r'T \|', na=False)) |
                (formatted_df.get("water_day_total_alert", pd.Series(dtype='object')).str.contains(r'T \|', na=False))
            )
            
            before_count = len(formatted_df)
            formatted_df = formatted_df[threshold_mask]
            after_count = len(formatted_df)
            
            print(f"🔽 Threshold filter applied: {before_count} → {after_count} subjects")
            
        elif alert_category == "NS":
            # Filter for Not Scored subjects
            formatted_df = formatted_df[formatted_df["percentile_category"] == "NS"]
        else:
            # Filter for specific percentile category (B, G, SB, SG)
            formatted_df = formatted_df[formatted_df["percentile_category"] == alert_category]
    
    # Apply sorting if specified
    if sort_option != "none":
        print(f"Available columns for sorting: {list(formatted_df.columns)}")
        
        if sort_option == "overall_percentile_asc":
            # Try session_overall_percentile first, fall back to overall_percentile
            if 'session_overall_percentile' in formatted_df.columns:
                print("Sorting by session_overall_percentile (ascending)")
                formatted_df = formatted_df.sort_values("session_overall_percentile", ascending=True, na_position='last')
            elif 'overall_percentile' in formatted_df.columns:
                print("Sorting by overall_percentile (ascending)")
                formatted_df = formatted_df.sort_values("overall_percentile", ascending=True, na_position='last')
            else:
                print("WARNING: No overall percentile column found for sorting")
        elif sort_option == "overall_percentile_desc":
            # Try session_overall_percentile first, fall back to overall_percentile
            if 'session_overall_percentile' in formatted_df.columns:
                print("Sorting by session_overall_percentile (descending)")
                formatted_df = formatted_df.sort_values("session_overall_percentile", ascending=False, na_position='last')
            elif 'overall_percentile' in formatted_df.columns:
                print("Sorting by overall_percentile (descending)")
                formatted_df = formatted_df.sort_values("overall_percentile", ascending=False, na_position='last')
            else:
                print("WARNING: No overall percentile column found for sorting")
    
    # Count percentile categories for debugging
    percentile_counts = formatted_df["percentile_category"].value_counts().to_dict()
    print(f"Percentile categories: {percentile_counts}")
    print(f"Applied sorting: {sort_option}")
    
    # Convert to records for datatable
    return formatted_df.to_dict("records") 