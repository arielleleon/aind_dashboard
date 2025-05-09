from dash import Input, Output, State, callback, ALL, MATCH, ctx, clientside_callback
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import datetime, timedelta
from app_utils import AppUtils
from app_elements.app_content.app_dataframe.app_dataframe import AppDataFrame
from app_elements.app_filter.app_filter import AppFilter
from app_elements.app_subject_detail.app_feature_chart import AppFeatureChart
from app_elements.app_subject_detail.app_session_card import AppSessionCard
from app_elements.app_subject_detail.app_subject_image_loader import AppSubjectImageLoader
import plotly.graph_objects as go
import json

app_utils = AppUtils()
app_dataframe = AppDataFrame()
app_filter = AppFilter()
feature_chart = AppFeatureChart()
session_card = AppSessionCard()
image_loader = AppSubjectImageLoader()

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
     Input("clear-filters", "n_clicks")]
)
def update_active_filters(
    time_window_value, stage_value, curriculum_value, rig_value, trainer_value, pi_value, sort_option, 
    alert_category, clear_clicks
):
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
    
    # Add each active filter
    if stage_value:
        active_filters.append(
            create_filter_badge(f"Stage: {stage_value}", "stage-filter", stage_value)
        )
    
    if curriculum_value:
        active_filters.append(
            create_filter_badge(f"Curriculum: {curriculum_value}", "curriculum-filter", curriculum_value)
        )
    
    if rig_value:
        active_filters.append(
            create_filter_badge(f"Rig: {rig_value}", "rig-filter", rig_value)
        )
    
    if trainer_value:
        active_filters.append(
            create_filter_badge(f"Trainer: {trainer_value}", "trainer-filter", trainer_value)
        )
    
    if pi_value:
        active_filters.append(
            create_filter_badge(f"PI: {pi_value}", "pi-filter", pi_value)
        )
    
    # Return active filters and count
    filter_count = len(active_filters)
    count_display = filter_count if filter_count > 0 else ""
    
    return active_filters, count_display

# Helper function to create filter badge
def create_filter_badge(label, filter_type, filter_value):
    filter_id = f"{filter_type}:{filter_value}"
    return dbc.Badge(
        [
            label,
            html.Span("×", className="ms-1", id={"type": "remove-filter", "index": filter_id})
        ],
        color="primary",
        className="me-1 mb-1 filter-badge",
        id={"type": "filter-badge", "index": filter_id}
    )

# Callback to handle filter badge removal
@callback(
    [Output("stage-filter", "value"),
     Output("curriculum-filter", "value"),
     Output("rig-filter", "value"),
     Output("trainer-filter", "value"),
     Output("pi-filter", "value"),
     Output("sort-option", "value"),
     Output("alert-category-filter", "value")],
    [Input({"type": "remove-filter", "index": ALL}, "n_clicks"),
     Input("clear-filters", "n_clicks")],
    [State({"type": "remove-filter", "index": ALL}, "id"),
     State("stage-filter", "value"),
     State("curriculum-filter", "value"),
     State("rig-filter", "value"),
     State("trainer-filter", "value"),
     State("pi-filter", "value"),
     State("sort-option", "value"),
     State("alert-category-filter", "value")],
    prevent_initial_call=True
)
def remove_filter(remove_clicks, clear_clicks, remove_ids, 
                 stage_value, curriculum_value, rig_value, trainer_value, 
                 pi_value, sort_value, alert_category_value):
    # Initialize return values with current state
    outputs = [stage_value, curriculum_value, rig_value, trainer_value, pi_value, 
              sort_value, alert_category_value]
    
    # If clear button was clicked, clear all filters
    if ctx.triggered_id == "clear-filters":
        return [None, None, None, None, None, "none", "all"]
    
    # Find which filter was clicked to be removed
    for i, clicks in enumerate(remove_clicks):
        if clicks:
            # Get the filter info that needs to be removed
            filter_id = remove_ids[i]["index"]
            filter_type, filter_value = filter_id.split(":", 1)
            
            # Don't allow removing time window filter
            if filter_type == "time-window-filter":
                continue
                
            # Clear the corresponding filter
            if filter_type == "stage-filter":
                outputs[0] = None
            elif filter_type == "curriculum-filter":
                outputs[1] = None
            elif filter_type == "rig-filter":
                outputs[2] = None
            elif filter_type == "trainer-filter":
                outputs[3] = None
            elif filter_type == "pi-filter":
                outputs[4] = None
            elif filter_type == "sort-option":
                outputs[5] = "none"  # Reset to default sort
            elif filter_type == "alert-category-filter":
                outputs[6] = "all"   # Reset to show all alerts
            
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
     Input("clear-filters", "n_clicks")]
)
def update_table_data(time_window_value, stage_value, curriculum_value, 
                     rig_value, trainer_value, pi_value, sort_option, 
                     alert_category, clear_clicks):
    print(f"Updating table with time window: {time_window_value} days")
    
    # Get the cached formatted data (all subjects, all time)
    if app_utils._cache['formatted_data'] is None:
        # If not cached, format the data once
        df = app_utils.get_session_data(use_cache=True)
        formatter = AppDataFrame()
        formatted_df = formatter.format_dataframe(df)
        app_utils._cache['formatted_data'] = formatted_df
    else:
        formatted_df = app_utils._cache['formatted_data'].copy()
    
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
    
    # Apply each filter if it has a value
    if stage_value:
        formatted_df = formatted_df[formatted_df["current_stage_actual"] == stage_value]
    
    if curriculum_value:
        formatted_df = formatted_df[formatted_df["curriculum_name"] == curriculum_value]
    
    if rig_value:
        formatted_df = formatted_df[formatted_df["rig"] == rig_value]
    
    if trainer_value:
        formatted_df = formatted_df[formatted_df["trainer"] == trainer_value]
    
    if pi_value:
        formatted_df = formatted_df[formatted_df["PI"] == pi_value]
    
    # Apply alert category filter if selected
    if alert_category != "all":
        if alert_category == "T":
            # Filter for threshold alerts specifically
            # Updated to handle the new "T | STAGE_NAME" format in stage_sessions_alert
            formatted_df = formatted_df[
                (formatted_df["threshold_alert"] == "T") | 
                (formatted_df["total_sessions_alert"] == "T") | 
                (formatted_df["stage_sessions_alert"].str.contains("T |", na=False)) | 
                (formatted_df["water_day_total_alert"] == "T")
            ]
        elif alert_category == "NS":
            # Filter for Not Scored subjects
            formatted_df = formatted_df[formatted_df["percentile_category"] == "NS"]
        else:
            # Filter for specific percentile category (B, G, SB, SG)
            formatted_df = formatted_df[formatted_df["percentile_category"] == alert_category]
    
    # Apply sorting if specified
    if sort_option != "none":
        if sort_option == "session_date":
            # Sort by session date (descending)
            formatted_df = formatted_df.sort_values("session_date", ascending=False)
        elif sort_option == "overall_percentile":
            # Sort by overall percentile (ascending)
            formatted_df = formatted_df.sort_values("overall_percentile", ascending=True)
        elif sort_option == "alert":
            # Custom sort order for alerts: SB, B, N, G, SG, NS
            alert_order = {"SB": 0, "B": 1, "N": 2, "G": 3, "SG": 4, "NS": 5}
            formatted_df["alert_sort"] = formatted_df["percentile_category"].map(alert_order)
            formatted_df = formatted_df.sort_values("alert_sort", ascending=True)
            formatted_df = formatted_df.drop(columns=["alert_sort"])
        elif sort_option == "session_count":
            # Sort by session count (descending)
            formatted_df = formatted_df.sort_values("session", ascending=False)
    
    # Count percentile categories for debugging
    percentile_counts = formatted_df["percentile_category"].value_counts().to_dict()
    print(f"Percentile categories: {percentile_counts}")
    
    # Convert to records for datatable
    return formatted_df.to_dict("records")

# Subject selection from table
@callback(
    [Output("subject-detail-footer", "style"),
     Output("detail-strata", "children"),
     Output("detail-subject-id", "children"),
     Output("detail-pi", "children"),
     Output("detail-trainer", "children"),
     Output("detail-percentile", "children"),
     Output("detail-last-session", "children"),
     Output("detail-consistency", "children"),
     Output("detail-threshold-alerts", "children"),
     Output("detail-ns-reason", "children"),
     Output("detail-ns-reason-container", "style"),
     Output("feature-chart-container", "children")],
    [Input("session-table", "active_cell")],
    [State("session-table", "data"),
     State("session-table", "page_current"),
     State("session-table", "page_size")]
)
def update_subject_detail(active_cell, table_data, page_current, page_size):
    print("Subject detail callback triggered")
    print(f"Active cell: {active_cell}")
    print(f"Current page: {page_current}, Page size: {page_size}")
    
    default_style = {"display": "none"}
    default_ns_style = {"display": "none"}
    empty_chart = feature_chart.build(None)

    # Check if cell is clicked
    if not active_cell or active_cell['column_id'] != 'subject_id':
        print("No active cell or not in subject_id column")
        return default_style, "", "", "", "", "", "", "", "", "", default_ns_style, empty_chart
    
    # Calculate the actual row index in the full dataset
    # If page_current is None (which can happen on initial load), default to 0
    current_page = page_current if page_current is not None else 0
    # If page_size is None, default to 20 (or whatever your default page size is)
    rows_per_page = page_size if page_size is not None else 20
    
    # Calculate the absolute row index in the dataset
    absolute_row_idx = (current_page * rows_per_page) + active_cell['row']
    
    # Safety check to make sure we don't go out of bounds
    if absolute_row_idx >= len(table_data):
        print(f"Row index {absolute_row_idx} is out of bounds for data length {len(table_data)}")
        return default_style, "", "", "", "", "", "", "", "", "", default_ns_style, empty_chart
    
    # Get the correct subject data using the absolute row index
    subject_data = table_data[absolute_row_idx]
    subject_id = subject_data['subject_id']
    print(f"Selected subject: {subject_id} at absolute row {absolute_row_idx}")

    # Extract data for display
    strata = f"Current Strata: {subject_data.get('strata_abbr', 'N/A')}"
    pi = subject_data.get('PI', 'N/A')
    trainer = subject_data.get('trainer', 'N/A')

    # Format percent with color based on category
    percentile_val = subject_data.get('overall_percentile')
    percentile_cat = subject_data.get('percentile_category', 'NS')

    if percentile_cat == 'NS' or pd.isna(percentile_val):
        percentile = "NS"
    else:
        percentile = f"{percentile_val:.2f}%"
    
    # Format last session
    session_date = subject_data.get('session_date')
    if session_date:
        try:
            if isinstance(session_date, str):
                session_date = datetime.strptime(session_date, '%Y-%m-%d %H:%M:%S')
            last_session = session_date.strftime('%Y-%m-%d')
        except:
            last_session = str(session_date)
    else:
        last_session = 'N/A'
    
    # Format training consistency
    consistency = f"{subject_data.get('training_consistency', 'N/A')}%" # NEED TO IMPLEMENT

    # Format threshold alerts
    threshold_alerts = []

    # Check total sessions alert
    total_sessions_alert = subject_data.get('total_sessions_alert', 'N')
    if 'T |' in total_sessions_alert:
        # Parse the value: "T | 45"
        parts = total_sessions_alert.split('|')
        value = parts[1].strip()
        threshold_alerts.append(html.Div([
            html.Span("Total Sessions: ", className="alert-label"),
            html.Span(f"Out of Range ({value})", className="alert-value")
        ]))

    # Check stage sessions alert
    stage_alert = subject_data.get('stage_sessions_alert', '')
    if 'T |' in stage_alert:
        # Parse the format: "T | STAGE_FINAL | 30"
        parts = stage_alert.split('|')
        stage_name = parts[1].strip()
        sessions = parts[2].strip() if len(parts) > 2 else ""
        threshold_alerts.append(html.Div([
            html.Span("Stage-Specific: ", className="alert-label"),
            html.Span(f"Out of Range | {stage_name} ({sessions})", className="alert-value")
        ]))

    # Check water day total alert
    water_alert = subject_data.get('water_day_total_alert', '')
    if 'T |' in water_alert:
        # Parse the format: "T | 3.7"
        parts = water_alert.split('|')
        value = parts[1].strip()
        threshold_alerts.append(html.Div([
            html.Span("water_day_total: ", className="alert-label"),
            html.Span(f"Out of Range ({value} mL)", className="alert-value")
        ]))

    # Format threshold alerts text
    if threshold_alerts:
        threshold_text = html.Div(threshold_alerts, className="threshold-alerts-container")
    else:
        threshold_text = "None"

    # Check NS reason
    ns_reason = subject_data.get('ns_reason', '')
    if percentile_cat == 'NS' and ns_reason:
        ns_reason_style = {'display': 'block'}
    else:
        ns_reason_style = {'display': 'none'}

    # Build feature chart
    chart = feature_chart.build(subject_data)

    # Return values to update UI
    return (
        {'display': 'block'}, # Show footer
        strata,
        subject_id,
        pi,
        trainer,
        percentile,
        last_session,
        consistency,
        threshold_text,
        ns_reason,
        ns_reason_style,
        chart
    )

# Updated callback with notification
@callback(
    [Output("subject-detail-page", "style"),
     Output("view-details-button", "children")],
    Input("view-details-button", "n_clicks"),
    prevent_initial_call=True
)
def show_subject_detail_page(n_clicks):
    if not n_clicks:
        return {'display': 'none'}, "Show Subject Details Section"
    
    # Show page and update button text to indicate it worked
    return {'display': 'block'}, [
        "Subject Details Loaded ", 
        html.I(className="fas fa-check ml-1", style={"color": "white"})
    ]
# Callback to populate the session list when a subject is selected
@callback(
    [Output("session-list-container", "children"),
     Output("session-list-state", "data"),
     Output("session-count", "children")],
    [Input("detail-subject-id", "children"),
     Input("load-more-sessions-btn", "n_clicks")],
    [State("session-list-state", "data")]
)
def update_session_list(subject_id, load_more_clicks, session_list_state):
    # Default return values
    empty_list = []
    updated_state = {"subject_id": None, "sessions_loaded": 0, "total_sessions": 0}
    session_count = "0"
    
    # Check if triggered by a subject change or load more button
    is_load_more = ctx.triggered_id == "load-more-sessions-btn"
    
    # If no subject selected or empty, return defaults
    if not subject_id:
        return empty_list, updated_state, session_count
    
    # Initialize with current state for load more
    current_subject = session_list_state.get("subject_id", None)
    sessions_loaded = session_list_state.get("sessions_loaded", 0)
    
    # If subject changed, reset state
    if subject_id != current_subject:
        sessions_loaded = 0
        is_load_more = False
    
    # Define how many sessions to load
    initial_session_count = 5
    load_more_count = 5
    
    # Calculate how many sessions to fetch
    if is_load_more:
        session_limit = sessions_loaded + load_more_count
    else:
        session_limit = initial_session_count
    
    # Get all sessions for this subject
    all_sessions = app_utils.get_subject_sessions(subject_id)
    
    if all_sessions is None or all_sessions.empty:
        return empty_list, updated_state, session_count
    
    # Sort by session date (descending)
    all_sessions = all_sessions.sort_values('session_date', ascending=False)
    
    # Limit to the requested number
    sessions_to_display = all_sessions.head(session_limit)
    
    # Build session cards
    session_cards = []
    for idx, session_row in sessions_to_display.iterrows():
        # First card is active by default
        is_active = idx == 0 and not is_load_more
        
        # Create session card
        card = session_card.build(session_row.to_dict(), is_active=is_active)
        session_cards.append(card)
    
    # Update state
    updated_state = {
        "subject_id": subject_id,
        "sessions_loaded": len(sessions_to_display),
        "total_sessions": len(all_sessions)
    }
    
    # Update session count
    session_count = str(len(sessions_to_display))
    
    return session_cards, updated_state, session_count

# Callback to handle session card selection
@callback(
    Output({"type": "session-card", "index": ALL}, "className"),
    [Input({"type": "session-card", "index": ALL}, "n_clicks")],
    [State({"type": "session-card", "index": ALL}, "id")]
)
def handle_session_selection(n_clicks_list, card_ids):
    # Find which card was clicked
    clicked_idx = None
    for i, clicks in enumerate(n_clicks_list):
        if clicks and clicks > 0:
            clicked_idx = i
            break
    
    # If no card was clicked, return unchanged
    if clicked_idx is None:
        return ["session-card" for _ in card_ids]
    
    # Update card classes - only the clicked one is active
    return ["session-card active" if i == clicked_idx else "session-card" 
            for i in range(len(card_ids))]