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
from app_elements.app_subject_detail.app_subject_timeseries import AppSubjectTimeseries
import plotly.graph_objects as go
import json

app_utils = AppUtils()
app_dataframe = AppDataFrame()
app_filter = AppFilter()
feature_chart = AppFeatureChart()
session_card = AppSessionCard()
image_loader = AppSubjectImageLoader()
subject_timeseries = AppSubjectTimeseries()

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

# Subject selection from table - modified to also show subject detail page automatically
@callback(
    [Output("subject-detail-footer", "style"),
     Output("subject-detail-page", "style"),  # Added output for the detail page
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
    
    default_footer_style = {"display": "none"}
    default_page_style = {"display": "none"}  # Style for the detail page
    default_ns_style = {"display": "none"}
    empty_chart = feature_chart.build(None)

    # Check if cell is clicked
    if not active_cell or active_cell['column_id'] != 'subject_id':
        print("No active cell or not in subject_id column")
        return default_footer_style, default_page_style, "", "", "", "", "", "", "", "", "", default_ns_style, empty_chart
    
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
        return default_footer_style, default_page_style, "", "", "", "", "", "", "", "", "", default_ns_style, empty_chart
    
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

    # Return values to update UI - now also showing the detail page automatically
    return (
        {'display': 'block'},    # Show footer 
        {'display': 'block'},    # Show detail page automatically
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

# Track scroll position and detect which session card is most visible
clientside_callback(
    """
    function(dummy, n_intervals) {
        const scrollContainer = document.getElementById('session-list-scroll-container');
        if (!scrollContainer) return {visible_session: null};
        
        // Function to throttle scroll events
        function throttle(func, limit) {
            let inThrottle;
            return function() {
                const args = arguments;
                const context = this;
                if (!inThrottle) {
                    func.apply(context, args);
                    inThrottle = true;
                    setTimeout(() => inThrottle = false, limit);
                }
            }
        }
        
        // Function to determine which session card is most visible
        function getMostVisibleSession() {
            // Use data-* attributes instead of complex JSON id selector
            const cards = document.querySelectorAll('[data-debug^="session-card-"]');
            if (!cards.length) return null;
            
            const containerRect = scrollContainer.getBoundingClientRect();
            const containerTop = containerRect.top;
            const containerBottom = containerRect.bottom;
            const containerHeight = containerBottom - containerTop;
            
            let maxVisibleArea = 0;
            let mostVisibleCard = null;
            
            cards.forEach(card => {
                const cardRect = card.getBoundingClientRect();
                
                // Calculate the visible area of this card
                const visibleTop = Math.max(cardRect.top, containerTop);
                const visibleBottom = Math.min(cardRect.bottom, containerBottom);
                
                if (visibleBottom > visibleTop) {
                    const visibleArea = visibleBottom - visibleTop;
                    const visibleRatio = visibleArea / cardRect.height;
                    
                    if (visibleRatio > maxVisibleArea) {
                        maxVisibleArea = visibleRatio;
                        mostVisibleCard = card;
                    }
                }
            });
            
            if (mostVisibleCard) {
                // Extract subject and session from data attributes
                const subjectId = mostVisibleCard.getAttribute('data-subject-id');
                const sessionNum = mostVisibleCard.getAttribute('data-session-num');
                
                if (subjectId && sessionNum) {
                    return `${subjectId}-${sessionNum}`;
                }
                
                // Fallback to parsing ID if data attributes not available
                try {
                    const idStr = mostVisibleCard.id;
                    const idJson = JSON.parse(idStr);
                    return idJson.index;
                } catch (e) {
                    console.error("Error extracting session info:", e);
                    return null;
                }
            }
            
            return null;
        }
        
        // Initialize session tracking if not already done
        if (!window.sessionScrollTracking) {
            window.sessionScrollTracking = {
                lastSession: null,
                throttledUpdate: throttle(function() {
                    window.sessionScrollTracking.lastSession = getMostVisibleSession();
                    // Allow the callback to update on next interval
                    window.sessionScrollTracking.needsUpdate = true;
                }, 50), // Throttle to 50ms
                needsUpdate: true
            };
            
            // Add scroll event listener with throttling
            scrollContainer.addEventListener('scroll', window.sessionScrollTracking.throttledUpdate);
        }
        
        // Only get the current visible session and trigger updates when needed
        // for better performance (avoid unnecessary processing on each interval)
        if (window.sessionScrollTracking.needsUpdate) {
            const currentVisibleSession = getMostVisibleSession();
            window.sessionScrollTracking.lastSession = currentVisibleSession;
            window.sessionScrollTracking.needsUpdate = false;
            return {visible_session: currentVisibleSession};
        }
        
        // Return the last known visible session without recalculating
        return {visible_session: window.sessionScrollTracking.lastSession};
    }
    """,
    Output("session-scroll-state", "data"),
    [Input("session-list-container", "children"),
     Input("scroll-tracker-interval", "n_intervals")]
)

# Combined callback to handle session card highlighting from both clicks and scrolling
clientside_callback(
    """
    function(n_clicks_list, visible_session_data) {
        // Check if this was triggered by a click
        let triggeredByClick = false;
        let clickedIdx = null;
        
        for (let i = 0; i < n_clicks_list.length; i++) {
            if (n_clicks_list[i] && n_clicks_list[i] > 0) {
                triggeredByClick = true;
                clickedIdx = i;
                break;
            }
        }
        
        // Get all cards using data attribute instead of complex JSON id
        const cards = document.querySelectorAll('[data-debug^="session-card-"]');
        
        // Default all cards to inactive
        let classes = Array(n_clicks_list.length).fill("session-card");
        
        if (triggeredByClick && clickedIdx !== null) {
            // If triggered by a click, only highlight the clicked card
            classes[clickedIdx] = "session-card active";
        } else {
            // Otherwise, check the scroll position
            const visibleSession = visible_session_data && visible_session_data.visible_session;
            
            if (visibleSession) {
                // For each card, determine if it matches the visible session
                cards.forEach((card, index) => {
                    // Skip if index is out of bounds for our classes array
                    if (index >= classes.length) return;
                    
                    // Get the session info from data attributes
                    const subjectId = card.getAttribute('data-subject-id');
                    const sessionNum = card.getAttribute('data-session-num');
                    const cardIndex = `${subjectId}-${sessionNum}`;
                    
                    if (cardIndex === visibleSession) {
                        classes[index] = "session-card active";
                    }
                });
            }
        }
        
        return classes;
    }
    """,
    Output({"type": "session-card", "index": ALL}, "className"),
    [Input({"type": "session-card", "index": ALL}, "n_clicks"),
     Input("session-scroll-state", "data")]
)

# Callback to update the timeseries data store when a subject is selected
@callback(
    Output("timeseries-data-store", "data"),
    [Input("detail-subject-id", "children")]
)
def update_timeseries_data(subject_id):
    """Update the timeseries data store with all sessions for the selected subject"""
    print(f"\nupdate_timeseries_data called with subject_id: {subject_id}")
    
    # Default return value
    empty_data = {
        "subject_id": None,
        "selected_session": None,
        "all_sessions": []
    }
    
    # If no subject selected, return empty data
    if not subject_id:
        print("No subject ID provided")
        return empty_data
        
    # Get all sessions for this subject
    print(f"Getting sessions for subject {subject_id}")
    all_sessions = app_utils.get_subject_sessions(subject_id)
    
    if all_sessions is None or all_sessions.empty:
        print(f"No sessions found for subject {subject_id}")
        return empty_data
        
    print(f"Found {len(all_sessions)} sessions for subject {subject_id}")
    print(f"Columns in sessions: {all_sessions.columns.tolist()}")
    
    # Check if processed feature columns exist
    feature_columns = [col for col in all_sessions.columns if '_processed' in col]
    print(f"Found {len(feature_columns)} processed feature columns: {feature_columns}")
    
    # Fix for missing processed values: manually add them if they don't exist
    if not feature_columns:
        print("WARNING: No processed feature columns found - adding them manually")
        for feature in subject_timeseries.features_config.keys():
            if feature not in all_sessions.columns:
                print(f"Feature {feature} not in columns, can't create processed values")
                continue
                
            # Create basic processed values - normalize the raw feature values
            # This is a simplified version without proper standardization across subjects
            # but should work for visualization purposes
            feature_vals = all_sessions[feature].values
            if len(feature_vals) == 0 or pd.isna(feature_vals).all():
                print(f"No valid values for feature {feature}, skipping")
                all_sessions[f"{feature}_processed"] = float('nan')
                continue
                
            # Simple standardization (z-score)
            try:
                mean = feature_vals[~pd.isna(feature_vals)].mean()
                std = feature_vals[~pd.isna(feature_vals)].std()
                if std == 0:  # Avoid division by zero
                    all_sessions[f"{feature}_processed"] = 0
                else:
                    processed = (feature_vals - mean) / std
                    # Flip sign if lower is better (so higher always = better)
                    if subject_timeseries.features_config[feature]:
                        processed = -processed
                    all_sessions[f"{feature}_processed"] = processed
                print(f"Created {feature}_processed values")
            except Exception as e:
                print(f"Error creating processed values for {feature}: {str(e)}")
                all_sessions[f"{feature}_processed"] = float('nan')
    
    # Sort by session number (ascending)
    if 'session' in all_sessions.columns:
        all_sessions = all_sessions.sort_values('session', ascending=True)
    else:
        all_sessions = all_sessions.sort_values('session_date', ascending=True)
    
    # Convert to records
    sessions_list = all_sessions.to_dict('records')
    
    # Check the first session for key data
    if sessions_list:
        first_session = sessions_list[0]
        print(f"First session has keys: {list(first_session.keys())[:10]}...")
        for feature in subject_timeseries.features_config:
            processed_col = f"{feature}_processed"
            if processed_col in first_session:
                print(f"{processed_col}: {first_session[processed_col]}")
            else:
                print(f"{processed_col} not found in session data")
    
    # Return data with all sessions
    return {
        "subject_id": subject_id,
        "selected_session": sessions_list[0]['session'] if sessions_list else None,
        "all_sessions": sessions_list
    }

# Callback to update the timeseries plot with session data and selected features
@callback(
    Output("subject-timeseries-graph", "figure"),
    [Input("timeseries-data-store", "data"),
     Input("feature-select-dropdown", "value")]
)
def update_timeseries_plot(timeseries_data, selected_features):
    """Update the timeseries plot with the selected features"""
    print(f"update_timeseries_plot called with: {timeseries_data.get('subject_id')}")
    print(f"Selected features: {selected_features}")
    
    sessions = timeseries_data.get("all_sessions", [])
    print(f"Number of sessions: {len(sessions)}")
    
    if not sessions:
        print("No sessions found, returning empty figure")
        fig = go.Figure()
        fig.update_layout(
            title=None,
            xaxis_title="Session Number",
            yaxis_title="Performance Metrics",
            template="plotly_white",
            margin=dict(l=20, r=10, t=10, b=30)
        )
        fig.add_annotation(
            text="Select a subject to view timeseries data",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Create a default figure
    fig = go.Figure()
    
    # Get all features if 'all' is selected
    all_features = list(subject_timeseries.features_config.keys())
    features_to_plot = all_features if 'all' in selected_features else selected_features
    print(f"Features to plot: {features_to_plot}")
    
    # Debug: Print the first session to check structure
    if sessions:
        print(f"First session keys: {list(sessions[0].keys())}")
        
        # Check if the processed columns exist
        processed_columns = [f"{feature}_processed" for feature in features_to_plot if feature != 'all']
        for col in processed_columns:
            if col in sessions[0]:
                print(f"Found column {col}: {sessions[0][col]}")
            else:
                print(f"Column {col} not found in session data")
    
    # Add traces for each feature
    traces_added = 0
    for feature in features_to_plot:
        if feature == 'all':
            continue
        
        # Get the processed values for this feature
        processed_column = f"{feature}_processed"
        
        # Extract data points
        session_numbers = []
        processed_values = []
        original_values = []
        hover_texts = []
        dates = []
        
        for session in sessions:
            if processed_column in session and not pd.isna(session[processed_column]):
                session_numbers.append(session['session'])
                processed_values.append(session[processed_column])
                
                # Store original feature value if available
                if feature in session:
                    original_values.append(session[feature])
                else:
                    original_values.append(None)
                
                # Store session date
                if 'session_date' in session:
                    dates.append(session['session_date'])
                else:
                    dates.append(None)
                
                # Enhanced hover text will be updated after smoothing
                hover_texts.append("")  # Placeholder
        
        print(f"Feature {feature}: Found {len(session_numbers)} data points")
        
        # Skip if no valid data points for this feature
        if not session_numbers:
            print(f"No data points for feature {feature}, skipping")
            continue
        
        # Apply moving average smoothing
        smoothed_values = subject_timeseries.moving_average(processed_values)
        
        # Update hover text with both original and smoothed values
        for i in range(len(hover_texts)):
            if smoothed_values[i] is not None:
                hover_text = f"Session: {session_numbers[i]}<br>"
                
                # Add date if available
                if dates[i] and not pd.isna(dates[i]):
                    if hasattr(dates[i], 'strftime'):
                        hover_text += f"Date: {dates[i].strftime('%Y-%m-%d')}<br>"
                    else:
                        hover_text += f"Date: {dates[i]}<br>"
                
                # Add original value if available
                if original_values[i] is not None and not pd.isna(original_values[i]):
                    hover_text += f"Raw {feature}: {original_values[i]:.2f}<br>"
                
                # Add processed value
                hover_text += f"Processed: {processed_values[i]:.2f}<br>"
                
                # Add smoothed value
                hover_text += f"Smoothed: {smoothed_values[i]:.2f}"
                
                hover_texts[i] = hover_text
            
        # Get the color for this feature
        color = subject_timeseries.feature_colors.get(feature, '#000000')
        
        # Create a more readable name for the feature
        feature_display = feature.replace('_', ' ').replace('abs(', '|').replace(')', '|').title()
        
        # Add the trace for this feature with smoothed line
        fig.add_trace(go.Scatter(
            x=session_numbers,
            y=smoothed_values,
            mode='lines',  # Lines only, no markers
            name=feature_display,
            line=dict(
                color=color,
                width=3,
                shape='spline',  # Smoother line interpolation
                smoothing=1.3
            ),
            hoverinfo="text",
            hovertext=hover_texts
        ))
        traces_added += 1
    
    print(f"Added {traces_added} traces to the plot")
    
    if traces_added > 0 and 'session_numbers' in locals() and session_numbers:
        # Add a horizontal reference line at y=0
        fig.add_shape(
            type="line",
            x0=min(session_numbers),
            y0=0,
            x1=max(session_numbers),
            y1=0,
            line=dict(color="gray", width=1, dash="dot")
        )
    
    # Update layout for better appearance
    fig.update_layout(
        title=None,
        xaxis_title="Session Number",
        yaxis_title="Performance Metrics",
        template="plotly_white",
        margin=dict(l=20, r=10, t=10, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="closest",
        # Add grid lines for better readability
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(211,211,211,0.3)'  # Light gray
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(211,211,211,0.3)'  # Light gray
        )
    )
    
    if traces_added == 0:
        print("WARNING: No traces were added to the plot. It will be blank.")
        # Add a annotation to explain the blank plot
        fig.add_annotation(
            text="No feature data available for this subject",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
    
    return fig

# Session highlighting callback
@callback(
    Output("subject-timeseries-graph", "figure", allow_duplicate=True),
    [Input({"type": "session-card", "index": ALL}, "n_clicks"),
     Input("session-scroll-state", "data"),
     Input("subject-timeseries-graph", "figure")],
    [State({"type": "session-card", "index": ALL}, "id")],
    prevent_initial_call=True
)
def direct_session_highlight(n_clicks_list, scroll_state, current_figure, card_ids):
    """Update the timeseries plot with a vertical line at the selected session"""
    # Check for valid inputs
    if not current_figure or not current_figure.get('data'):
        return current_figure
    
    # Check if this was triggered by a click or a scroll event
    trigger_id = ctx.triggered_id
    
    try:
        session_str = None
        
        # Handle click events (prioritize these over scroll events)
        if trigger_id != "session-scroll-state" and n_clicks_list and any(n_clicks_list):
            # Find the card that was clicked (with highest n_clicks)
            max_clicks = 0
            clicked_idx = None
            
            for i, clicks in enumerate(n_clicks_list):
                if clicks and clicks > max_clicks:
                    max_clicks = clicks
                    clicked_idx = i
            
            # If a card was clicked, get its info
            if clicked_idx is not None:
                # Get the card ID and extract session number
                card_id = card_ids[clicked_idx]
                index_str = str(card_id.get('index', ''))
                
                # Parse session number from id format: "subject_id-session_num"
                print(f"Card clicked with index: {index_str}")
                
                if "-" in index_str:
                    _, session_str = index_str.split("-")
        
        # If no card was clicked or no valid session found, check scroll state
        if not session_str and trigger_id == "session-scroll-state" and scroll_state and scroll_state.get('visible_session'):
            visible_session = scroll_state.get('visible_session')
            print(f"Scroll event detected, visible session: {visible_session}")
            
            # Parse session number from id format: "subject_id-session_num"
            if visible_session and "-" in visible_session:
                _, session_str = visible_session.split("-")
        
        # If we don't have a session to highlight, return unchanged
        if not session_str:
            return current_figure
            
        # Handle session numbers with decimal points
        try:
            session_num = float(session_str)
            session_num = int(session_num) if session_num.is_integer() else session_num
        except ValueError:
            print(f"Invalid session number: {session_str}")
            return current_figure
            
        print(f"Highlighting session: {session_num}")
        
        # Create a modified figure with the vertical line
        fig = go.Figure(current_figure)
        
        # Calculate y-axis range if needed
        y_range = [-2, 2]  # Default range
        for trace in fig.data:
            if hasattr(trace, 'y') and trace.y:
                valid_y = [y for y in trace.y if y is not None]
                if valid_y:
                    trace_min = min(valid_y)
                    trace_max = max(valid_y)
                    y_range[0] = min(y_range[0], trace_min - 0.1)
                    y_range[1] = max(y_range[1], trace_max + 0.1)
        
        # Create simplified shapes and annotations
        # First, remove any existing highlights
        layout_data = {}
        if hasattr(fig.layout, 'shapes'):
            layout_data['shapes'] = [shape for shape in fig.layout.shapes 
                                   if not getattr(shape, 'name', '') == 'session_highlight']
        
        # Add the new highlight line
        highlight_shape = {
            'type': 'line',
            'name': 'session_highlight',
            'x0': session_num,
            'y0': y_range[0],
            'x1': session_num,
            'y1': y_range[1],
            'line': {'color': 'rgba(65, 105, 225, 0.3)', 'width': 6, 'dash': 'solid'}
        }
        
        if 'shapes' in layout_data:
            layout_data['shapes'].append(highlight_shape)
        else:
            layout_data['shapes'] = [highlight_shape]
        
        # Add annotation to indicate the selected session
        if hasattr(fig.layout, 'annotations'):
            layout_data['annotations'] = [ann for ann in fig.layout.annotations 
                                    if not getattr(ann, 'name', '') == 'session_annotation']
        else:
            layout_data['annotations'] = []
            
        # Add annotation for the current session - simple number, no arrow
        layout_data['annotations'].append({
            'name': 'session_annotation',
            'x': session_num - 0.5,  # Position slightly to the left of the line
            'y': y_range[1] * 0.95,  # Position slightly below the top
            'xref': 'x',
            'yref': 'y',
            'text': f'{session_num}',
            'showarrow': False,
            'font': {'color': 'rgba(65, 105, 225, 0.7)', 'size': 13}  # Less bold, slightly more transparent
        })
        
        # Update the figure layout
        fig.update_layout(**layout_data)
        
        return fig
        
    except Exception as e:
        print(f"ERROR in direct_session_highlight: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return current_figure  # Return unchanged on error

# Ensure the hidden output target exists somewhere in the layout
def create_hidden_div():
    return html.Div(id="hidden-callback-target", style={"display": "none"})