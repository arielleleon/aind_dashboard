from dash import Input, Output, State, callback, ALL, MATCH, ctx, clientside_callback
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import datetime, timedelta
from app_elements.app_content.app_dataframe.app_dataframe import AppDataFrame
from app_elements.app_filter.app_filter import AppFilter
from app_elements.app_subject_detail.app_feature_chart import AppFeatureChart
from app_elements.app_subject_detail.app_session_card import AppSessionCard
from app_elements.app_subject_detail.app_subject_image_loader import AppSubjectImageLoader
from app_elements.app_subject_detail.app_subject_timeseries import AppSubjectTimeseries
from app_elements.app_subject_detail.app_subject_percentile_timeseries import AppSubjectPercentileTimeseries
import plotly.graph_objects as go
import json

# CRITICAL FIX: Import the shared app_utils instance from shared_utils module
from shared_utils import app_utils

# CRITICAL FIX: Pass the shared app_utils instance to AppDataFrame
app_dataframe = AppDataFrame(app_utils=app_utils)

app_filter = AppFilter()
feature_chart = AppFeatureChart()
session_card = AppSessionCard()
image_loader = AppSubjectImageLoader()
subject_timeseries = AppSubjectTimeseries()
subject_percentile_timeseries = AppSubjectPercentileTimeseries()

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
    
    # Helper function to format multi-select values
    def format_multi_value(value):
        if isinstance(value, list):
            if len(value) == 1:
                return value[0]
            elif len(value) <= 3:
                return ", ".join(value)
            else:
                return f"{', '.join(value[:2])}, +{len(value)-2} more"
        else:
            return str(value)
    
    # Add each active filter - now handling multi-select
    if stage_value:
        formatted_value = format_multi_value(stage_value)
        # Use the first value as the key for removal
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
    [Output("time-window-filter", "value"),
     Output("stage-filter", "value"),
     Output("curriculum-filter", "value"),
     Output("rig-filter", "value"),
     Output("trainer-filter", "value"),
     Output("pi-filter", "value"),
     Output("sort-option", "value"),
     Output("alert-category-filter", "value")],
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
     State("alert-category-filter", "value")],
    prevent_initial_call=True
)
def remove_filter(remove_clicks, clear_clicks, time_window_value, remove_ids, 
                 stage_value, curriculum_value, rig_value, trainer_value, 
                 pi_value, sort_value, alert_category_value):
    # Initialize return values with current state
    outputs = [time_window_value, stage_value, curriculum_value, rig_value, trainer_value, pi_value, 
              sort_value, alert_category_value]
    
    # Add debugging to track clear button clicks
    print(f"remove_filter callback triggered. ctx.triggered_id: {ctx.triggered_id}")
    if clear_clicks:
        print(f"Clear button clicks: {clear_clicks}")
    
    # If clear button was clicked, clear all filters
    if ctx.triggered_id == "clear-filters":
        print("🧹 CLEAR ALL FILTERS button pressed - resetting all values")
        return [30, None, None, None, None, None, "none", "all"]  # Reset to default time window (30 days)
    
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
                
            # Clear the corresponding filter - now properly handles multi-select
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
    
    # CRITICAL FIX: Use UI-optimized table display data from the shared app_utils instance
    # This data is already processed and cached, avoiding pipeline re-runs
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

    # Apply each filter if it has a value - now handling multi-select
    if stage_value:
        # Handle both single values and lists
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
    
    # Apply sorting if specified - updated to handle new sort values
    if sort_option != "none":
        # Debug: Check what columns are available
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
    empty_chart = feature_chart.build_legacy(None)

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
    if percentile_val is None or pd.isna(percentile_val):
        # Try session_overall_percentile as fallback
        percentile_val = subject_data.get('session_overall_percentile')
    
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

    # Build feature chart using optimized data structure
    print(f"Building feature chart for subject: {subject_id}")
    try:
        chart = feature_chart.build(subject_id=subject_id, app_utils=app_utils)
        print(f"✓ Feature chart built successfully")
    except Exception as e:
        print(f"❌ Error building feature chart: {str(e)}")
        # Return fallback empty chart
        chart = feature_chart.build_legacy(None)

    # Return values to update UI - now also showing the detail page automatically
    print(f"🔄 SETTING detail-subject-id to: {subject_id} (type: {type(subject_id)})")
    print(f"   This should trigger the timeseries callback with subject_id: {repr(subject_id)}")
    print(f"🖥️  SHOWING subject detail page and footer:")
    print(f"   Footer style: {{'display': 'block'}}")
    print(f"   Detail page style: {{'display': 'block'}}")
    
    return (
        {'display': 'block'},    # Show footer 
        {'display': 'block'},    # Show detail page automatically
        strata,
        subject_id,              # This is the critical value for detail-subject-id
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
    initial_session_count = 8  # Increased from 5 to accommodate longer page
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

# Callback to load timeseries data when subject is selected
@callback(
    Output("timeseries-store", "data"),
    [Input("detail-subject-id", "children")]
)
def load_timeseries_data(subject_id):
    """Load optimized timeseries data for selected subject"""
    if not subject_id:
        return {}
    
    # Get optimized time series data from app_utils
    time_series_data = app_utils.get_time_series_data(subject_id, use_cache=True)
    
    if not time_series_data:
        print(f"No timeseries data found for subject {subject_id}")
        return {}
    
    print(f"Loaded timeseries data for {subject_id}: {len(time_series_data.get('sessions', []))} sessions")
    return time_series_data

# Callback to update timeseries plot
@callback(
    Output("timeseries-plot", "figure"),
    [Input("timeseries-store", "data"),
     Input("timeseries-feature-dropdown", "value"),
     Input({"type": "session-card", "index": ALL}, "n_clicks"),
     Input("session-scroll-state", "data")],
    [State({"type": "session-card", "index": ALL}, "id")]
)
def update_timeseries_plot(timeseries_data, selected_features, n_clicks_list, scroll_state, card_ids):
    """Update timeseries plot with data and session highlighting"""
    
    # Determine highlighted session from clicks or scroll
    highlighted_session = None
    
    # Check for card clicks first (priority over scroll)
    if n_clicks_list and any(n_clicks_list):
        max_clicks = max(n_clicks_list)
        if max_clicks > 0:
            clicked_idx = n_clicks_list.index(max_clicks)
            if clicked_idx < len(card_ids):
                card_id = card_ids[clicked_idx]
                session_str = card_id.get('index', '').split('-')[-1]
                try:
                    highlighted_session = int(float(session_str))
                except (ValueError, IndexError):
                    pass
    
    # Check scroll state if no click detected
    elif scroll_state and scroll_state.get('visible_session'):
        visible_session = scroll_state.get('visible_session')
        if '-' in visible_session:
            session_str = visible_session.split('-')[-1]
            try:
                highlighted_session = int(float(session_str))
            except (ValueError, IndexError):
                pass
    
    # Create the plot
    return subject_timeseries.create_plot(
        subject_data=timeseries_data,
        selected_features=selected_features or ['all'],
        highlighted_session=highlighted_session
    )

# Callback to update percentile timeseries plot
@callback(
    Output("percentile-timeseries-plot", "figure"),
    [Input("timeseries-store", "data"),
     Input("percentile-timeseries-feature-dropdown", "value"),
     Input({"type": "session-card", "index": ALL}, "n_clicks"),
     Input("session-scroll-state", "data")],
    [State({"type": "session-card", "index": ALL}, "id")]
)
def update_percentile_timeseries_plot(timeseries_data, selected_features, n_clicks_list, scroll_state, card_ids):
    """Update percentile timeseries plot with data and session highlighting"""
    
    # Determine highlighted session from clicks or scroll (same logic as raw timeseries)
    highlighted_session = None
    
    # Check for card clicks first (priority over scroll)
    if n_clicks_list and any(n_clicks_list):
        max_clicks = max(n_clicks_list)
        if max_clicks > 0:
            clicked_idx = n_clicks_list.index(max_clicks)
            if clicked_idx < len(card_ids):
                card_id = card_ids[clicked_idx]
                session_str = card_id.get('index', '').split('-')[-1]
                try:
                    highlighted_session = int(float(session_str))
                except (ValueError, IndexError):
                    pass
    
    # Check scroll state if no click detected
    elif scroll_state and scroll_state.get('visible_session'):
        visible_session = scroll_state.get('visible_session')
        if '-' in visible_session:
            session_str = visible_session.split('-')[-1]
            try:
                highlighted_session = int(float(session_str))
            except (ValueError, IndexError):
                pass
    
    # Create the percentile plot
    return subject_percentile_timeseries.create_plot(
        subject_data=timeseries_data,
        selected_features=selected_features or ['all'],
        highlighted_session=highlighted_session
    )

