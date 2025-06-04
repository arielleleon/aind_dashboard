from dash import Input, Output, State, callback, ALL, MATCH, ctx, clientside_callback
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import datetime, timedelta
from app_elements.app_content.app_dataframe.app_dataframe import AppDataFrame
from app_elements.app_content.app_tooltip.app_hover_tooltip import AppHoverTooltip
from app_elements.app_filter.app_filter import AppFilter
from app_elements.app_subject_detail.app_session_card import AppSessionCard
from app_elements.app_subject_detail.app_subject_image_loader import AppSubjectImageLoader
from app_elements.app_subject_detail.app_subject_timeseries import AppSubjectTimeseries
from app_elements.app_subject_detail.app_subject_percentile_timeseries import AppSubjectPercentileTimeseries
from app_elements.app_subject_detail.app_subject_percentile_heatmap import AppSubjectPercentileHeatmap
from app_elements.app_subject_detail.app_subject_compact_info import AppSubjectCompactInfo
import plotly.graph_objects as go
import json

# CRITICAL FIX: Import the shared app_utils instance from shared_utils module
from shared_utils import app_utils

# CRITICAL FIX: Pass the shared app_utils instance to AppDataFrame
app_dataframe = AppDataFrame(app_utils=app_utils)
app_tooltip = AppHoverTooltip(app_utils=app_utils)

app_filter = AppFilter()
session_card = AppSessionCard()
image_loader = AppSubjectImageLoader()
subject_timeseries = AppSubjectTimeseries()
subject_percentile_timeseries = AppSubjectPercentileTimeseries()
percentile_heatmap = AppSubjectPercentileHeatmap()
compact_info = AppSubjectCompactInfo()

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
    
    # NEW: Add subject ID filter
    if subject_id_value:
        formatted_value = format_multi_value(subject_id_value)
        key_value = subject_id_value[0] if isinstance(subject_id_value, list) else subject_id_value
        active_filters.append(
            create_filter_badge(f"Subject ID: {formatted_value}", "subject-id-filter", key_value)
        )
    
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
            # FIXED: Handle the actual format of threshold alerts which contain "T |" not just "T"
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
            
            # CORRECTED FILTER: Match the actual threshold alert patterns
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

# Subject selection from table - modified to use new layout components
@callback(
    [Output("subject-detail-page", "style"),  # Only need the page style now
     Output("compact-subject-info-container", "children"),  # NEW: Compact info
     Output("selected-subject-store", "data")],  # Store the selected subject ID
    [Input("session-table", "active_cell")],
    [State("session-table", "data"),
     State("session-table", "page_current"),
     State("session-table", "page_size")]
)
def update_subject_detail(active_cell, table_data, page_current, page_size):
    print("Subject detail callback triggered")
    print(f"Active cell: {active_cell}")
    print(f"Current page: {page_current}, Page size: {page_size}")
    
    default_page_style = {"display": "none"}
    empty_compact_info = compact_info.build()
    empty_subject_store = {"subject_id": None}

    # Check if cell is clicked
    if not active_cell or active_cell['column_id'] != 'subject_id':
        print("No active cell or not in subject_id column")
        return default_page_style, empty_compact_info, empty_subject_store
    
    # Calculate the actual row index in the full dataset
    current_page = page_current if page_current is not None else 0
    rows_per_page = page_size if page_size is not None else 50
    absolute_row_idx = (current_page * rows_per_page) + active_cell['row']
    
    # Safety check
    if absolute_row_idx >= len(table_data):
        print(f"Row index {absolute_row_idx} is out of bounds for data length {len(table_data)}")
        return default_page_style, empty_compact_info, empty_subject_store
    
    # Get the subject data
    subject_data = table_data[absolute_row_idx]
    subject_id = subject_data['subject_id']
    print(f"Selected subject: {subject_id} at absolute row {absolute_row_idx}")

    # Build components using optimized data structures
    print(f"Building components for subject: {subject_id}")
    
    try:
        # Build compact info
        compact_subject_info = compact_info.build(subject_id=subject_id, app_utils=app_utils)
        print(f"✓ Compact info built successfully")
        
    except Exception as e:
        print(f"❌ Error building components: {str(e)}")
        compact_subject_info = empty_compact_info

    print(f"🖥️  SHOWING subject detail page with new layout (heatmap handled by separate callback)")
    
    return (
        {'display': 'block'},    # Show detail page
        compact_subject_info,    # Compact info (full width)
        {"subject_id": subject_id}  # Store the selected subject ID
    )

# Callback to populate the session list when a subject is selected
@callback(
    [Output("session-list-container", "children"),
     Output("session-list-state", "data"),
     Output("session-count", "children")],
    [Input("selected-subject-store", "data")],  # Removed load-more-sessions-btn input
    [State("session-list-state", "data")]
)
def update_session_list(selected_subject_data, session_list_state):
    # Extract subject_id from the store data
    subject_id = selected_subject_data.get('subject_id') if selected_subject_data else None
    
    print(f"Session list callback triggered with subject_id: {subject_id}")
    
    # Default return values
    empty_list = []
    updated_state = {"subject_id": None, "sessions_loaded": 0, "total_sessions": 0}
    session_count = "0"
    
    # If no subject selected or empty, return defaults
    if not subject_id:
        print("No subject selected - returning empty session list")
        return empty_list, updated_state, session_count
    
    # Get all sessions for this subject
    all_sessions = app_utils.get_subject_sessions(subject_id)
    
    if all_sessions is None or all_sessions.empty:
        return empty_list, updated_state, session_count
    
    # Sort by session date (descending) and load ALL sessions
    all_sessions = all_sessions.sort_values('session_date', ascending=False)
    
    print(f"Loading all {len(all_sessions)} sessions for subject {subject_id}")
    
    # Build session cards for ALL sessions
    session_cards = []
    for idx, session_row in all_sessions.iterrows():
        # First card is active by default
        is_active = idx == 0
        
        # Create session card
        card = session_card.build(session_row.to_dict(), is_active=is_active)
        session_cards.append(card)
    
    # Update state with all sessions loaded
    updated_state = {
        "subject_id": subject_id,
        "sessions_loaded": len(all_sessions),
        "total_sessions": len(all_sessions)
    }
    
    # Update session count
    session_count = str(len(all_sessions))
    
    return session_cards, updated_state, session_count

# Track scroll position and detect which session card is most visible
clientside_callback(
    """
    function(dummy, n_intervals) {
        // Only run if the subject detail page is visible
        const subjectDetailPage = document.getElementById('subject-detail-page');
        if (!subjectDetailPage || subjectDetailPage.style.display === 'none') {
            return {visible_session: null};
        }
        
        const scrollContainer = document.getElementById('session-list-scroll-container');
        if (!scrollContainer) return {visible_session: null};
        
        // Check if the scroll container is actually visible in the viewport
        const containerRect = scrollContainer.getBoundingClientRect();
        if (containerRect.height === 0 || containerRect.width === 0) {
            return {visible_session: null};
        }
        
        // Function to throttle scroll events - increased throttle time
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
        
        // Function to determine which session card is most visible (center-based approach)
        function getMostVisibleSession() {
            // Use data-* attributes instead of complex JSON id selector
            const cards = document.querySelectorAll('[data-debug^="session-card-"]');
            if (!cards.length) return null;
            
            const containerRect = scrollContainer.getBoundingClientRect();
            const containerTop = containerRect.top;
            const containerBottom = containerRect.bottom;
            const containerCenter = (containerTop + containerBottom) / 2;
            
            let minDistanceToCenter = Infinity;
            let mostVisibleCard = null;
            
            cards.forEach(card => {
                const cardRect = card.getBoundingClientRect();
                
                // Only consider cards that are at least partially visible
                const visibleTop = Math.max(cardRect.top, containerTop);
                const visibleBottom = Math.min(cardRect.bottom, containerBottom);
                
                if (visibleBottom > visibleTop) {
                    // Calculate the center of the visible portion of the card
                    const visibleCenter = (visibleTop + visibleBottom) / 2;
                    
                    // Calculate distance from visible card center to container center
                    const distanceToCenter = Math.abs(visibleCenter - containerCenter);
                    
                    // Prefer the card whose visible center is closest to the container center
                    if (distanceToCenter < minDistanceToCenter) {
                        minDistanceToCenter = distanceToCenter;
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
                    // Only update if the subject detail page is still visible
                    const subjectDetailPage = document.getElementById('subject-detail-page');
                    if (subjectDetailPage && subjectDetailPage.style.display !== 'none') {
                        window.sessionScrollTracking.lastSession = getMostVisibleSession();
                        window.sessionScrollTracking.needsUpdate = true;
                    }
                }, 150), // Increased throttle from 50ms to 150ms
                needsUpdate: true,
                listenerAttached: false
            };
        }
        
        // Only attach scroll listener if not already attached and container exists
        if (!window.sessionScrollTracking.listenerAttached && scrollContainer) {
            // Use passive event listener to improve performance and prevent interference
            scrollContainer.addEventListener('scroll', window.sessionScrollTracking.throttledUpdate, { passive: true });
            window.sessionScrollTracking.listenerAttached = true;
        }
        
        // Reduce update frequency - only check every 3rd interval (300ms instead of 100ms)
        if (n_intervals % 3 !== 0) {
            return {visible_session: window.sessionScrollTracking.lastSession};
        }
        
        // Only get the current visible session and trigger updates when needed
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

# Cleanup scroll tracking when subject detail page is hidden
clientside_callback(
    """
    function(page_style) {
        // Clean up scroll tracking when subject detail page is hidden
        if (page_style && page_style.display === 'none' && window.sessionScrollTracking) {
            // Remove the scroll event listener to prevent interference with page scrolling
            const scrollContainer = document.getElementById('session-list-scroll-container');
            if (scrollContainer && window.sessionScrollTracking.listenerAttached) {
                scrollContainer.removeEventListener('scroll', window.sessionScrollTracking.throttledUpdate);
                window.sessionScrollTracking.listenerAttached = false;
                window.sessionScrollTracking.needsUpdate = false;
                window.sessionScrollTracking.lastSession = null;
            }
        }
        return {};  // Return empty object since we're not updating any outputs
    }
    """,
    Output("session-card-selected", "data"),  # Use existing store as dummy output
    [Input("subject-detail-page", "style")]
)

# Update the timeseries callback to use the store
@callback(
    Output("timeseries-store", "data"),
    [Input("selected-subject-store", "data")]
)
def load_timeseries_data(selected_subject_data):
    """Load optimized timeseries data for selected subject"""
    if not selected_subject_data or not selected_subject_data.get('subject_id'):
        return {}
    
    subject_id = selected_subject_data['subject_id']
    
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
     Input("percentile-ci-toggle", "value"),  # Add CI toggle input
     Input({"type": "session-card", "index": ALL}, "n_clicks"),
     Input("session-scroll-state", "data")],
    [State({"type": "session-card", "index": ALL}, "id")]
)
def update_percentile_timeseries_plot(timeseries_data, selected_features, ci_toggle_value, n_clicks_list, scroll_state, card_ids):
    """Update percentile timeseries plot with data, session highlighting, and confidence interval toggle"""
    
    # Determine if confidence intervals should be shown
    show_confidence_intervals = ci_toggle_value and 'show_ci' in ci_toggle_value
    
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
    
    # Create the percentile plot with confidence intervals toggle
    return subject_percentile_timeseries.create_plot(
        subject_data=timeseries_data,
        selected_features=selected_features or ['all'],
        highlighted_session=highlighted_session,
        show_confidence_intervals=show_confidence_intervals
    )

# Callback to update percentile heatmap with session highlighting
@callback(
    Output("percentile-heatmap-container", "children"),
    [Input("selected-subject-store", "data"),
     Input({"type": "session-card", "index": ALL}, "n_clicks"),
     Input("session-scroll-state", "data"),
     Input("heatmap-colorscale-state", "data")],
    [State({"type": "session-card", "index": ALL}, "id")]
)
def update_percentile_heatmap_with_highlighting(selected_subject_data, n_clicks_list, scroll_state, colorscale_state, card_ids):
    """Update percentile heatmap with session highlighting and colorscale mode"""
    
    # Extract subject_id from the store data
    subject_id = selected_subject_data.get('subject_id') if selected_subject_data else None
    
    if not subject_id:
        # Return empty heatmap if no subject selected
        return percentile_heatmap.build()
    
    # Get colorscale mode from state
    colorscale_mode = colorscale_state.get('mode', 'binned') if colorscale_state else 'binned'
    
    # Determine highlighted session from clicks or scroll (same logic as timeseries)
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
    
    # Build heatmap with highlighting and colorscale mode
    return percentile_heatmap.build(
        subject_id=subject_id, 
        app_utils=app_utils,
        highlighted_session=highlighted_session,
        colorscale_mode=colorscale_mode
    )

# Callback to handle heatmap colorscale toggle button
@callback(
    [Output("heatmap-colorscale-state", "data"),
     Output("heatmap-colorscale-toggle", "children"),
     Output("heatmap-colorscale-toggle", "color")],
    [Input("heatmap-colorscale-toggle", "n_clicks")],
    [State("heatmap-colorscale-state", "data")]
)
def toggle_heatmap_colorscale(n_clicks, current_state):
    """Toggle between binned and continuous colorscale modes"""
    
    # If button hasn't been clicked, return current state
    if not n_clicks:
        current_mode = current_state.get('mode', 'binned') if current_state else 'binned'
        button_text = "Binned" if current_mode == 'binned' else "Continuous" 
        button_color = "outline-secondary" if current_mode == 'binned' else "outline-primary"
        return current_state or {"mode": "binned"}, button_text, button_color
    
    # Toggle the mode
    current_mode = current_state.get('mode', 'binned') if current_state else 'binned'
    new_mode = 'continuous' if current_mode == 'binned' else 'binned'
    
    # Update button appearance based on new mode
    button_text = "Continuous" if new_mode == 'continuous' else "Binned"
    button_color = "outline-primary" if new_mode == 'continuous' else "outline-secondary"
    
    print(f"🎨 Heatmap colorscale toggled from {current_mode} to {new_mode}")
    
    return {"mode": new_mode}, button_text, button_color

# Tooltip Callbacks

# Responsive Table Page Size Calculation
clientside_callback(
    """
    function(table_data, time_window, dummy_trigger) {
        // Calculate optimal page size based on viewport dimensions
        const viewportHeight = window.innerHeight;
        
        // Account for fixed elements (estimated heights)
        const topBarHeight = 50;        // Top navigation bar
        const filterHeight = 170;       // Filter section
        const tableHeaderHeight = 60;   // Table header (fixed)
        const paddingMargin = 50;       // Various padding and margins
        
        // Calculate available height for table rows
        const availableHeight = viewportHeight - topBarHeight - filterHeight - tableHeaderHeight - paddingMargin;
        
        // Each table row is approximately 48px (from style_cell height)
        const rowHeight = 48;
        
        // Calculate number of rows that can fit
        const calculatedRows = Math.floor(availableHeight / rowHeight);
        
        // Set reasonable bounds
        const minRows = 8;   // Minimum rows to show
        const maxRows = 50;  // Maximum rows (don't make it too large)
        
        const optimalRows = Math.max(minRows, Math.min(maxRows, calculatedRows));
        
        console.log(`📐 Responsive table: viewport ${viewportHeight}px → showing ${optimalRows} rows`);
        
        return optimalRows;
    }
    """,
    Output("session-table", "page_size"),
    [Input("session-table", "data"),
     Input("time-window-filter", "value"),
     Input("resize-interval", "n_intervals")]
)

# Simple window resize detection
clientside_callback(
    """
    function(table_id) {
        // Set up a simple window resize listener
        let resizeTimeout;
        
        function handleResize() {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                // Force a refresh of the table data to trigger page_size recalculation
                const tableElement = document.getElementById('session-table');
                if (tableElement) {
                    // Trigger a minor update to force callback refresh
                    tableElement.style.minHeight = window.innerHeight > 1000 ? '400px' : '300px';
                }
            }, 400);
        }
        
        // Clean up previous listener
        if (window.dashResizeHandler) {
            window.removeEventListener('resize', window.dashResizeHandler);
        }
        
        // Add new listener
        window.dashResizeHandler = handleResize;
        window.addEventListener('resize', handleResize, { passive: true });
        
        return 'setup-complete';
    }
    """,
    Output("resize-trigger", "data"),
    [Input("session-table", "id")]
)

# Server callback to update tooltip content
@callback(
    Output("subject-hover-tooltip", "children"),
    [Input("session-table", "active_cell")],
    [State("session-table", "data")]
)
def update_tooltip_content(active_cell, table_data):
    """
    Update tooltip content when hovering over subject ID cells
    We'll use the active_cell as a proxy since it's triggered by cell interactions
    
    Returns empty tooltip since we'll handle the actual tooltip display via clientside
    """
    return app_tooltip.get_empty_tooltip()

# Clientside callback for complete tooltip functionality
clientside_callback(
    """
    function(table_data, app_utils_placeholder) {
        // Initialize tooltip functionality
        console.log('Setting up tooltip functionality...');
        
        const table = document.getElementById('session-table');
        const tooltip = document.getElementById('subject-hover-tooltip');
        
        if (!table || !tooltip) {
            console.log('Table or tooltip not found');
            return window.dash_clientside.no_update;
        }
        
        let currentHoveredSubject = null;
        let hideTimeout = null;
        
        // Cache for tooltip data to avoid repeated server calls
        const tooltipDataCache = {};
        
        // Function to get alert color based on category
        function getAlertColor(category) {
            const colors = {
                'SB': '#FF6B35',  // Dark orange (Severely Below)
                'B': '#FFB366',   // Light orange (Below)  
                'G': '#4A90E2',   // Light blue (Good)
                'SG': '#2E5A87',  // Dark blue (Severely Good)
                'T': '#795548'    // Brown (Threshold alerts)
            };
            return colors[category] || null;
        }
        
        // Function to create tooltip HTML content
        function createTooltipContent(subjectData) {
            if (!subjectData) return '';
            
            let html = '<div class="tooltip-content">';
            
            // Header
            html += '<div class="tooltip-header">';
            html += `<div class="tooltip-subject-id">${subjectData.subject_id}</div>`;
            html += `<div class="tooltip-strata">Strata: ${subjectData.strata_abbr || 'N/A'}</div>`;
            html += '</div>';
            
            // Overall percentile
            const overallPercentile = subjectData.session_overall_percentile || subjectData.overall_percentile;
            const overallCategory = subjectData.overall_percentile_category || 'NS';
            
            html += '<div class="tooltip-overall">';
            html += '<span class="tooltip-label">Overall: </span>';
            
            if (overallPercentile != null && !isNaN(overallPercentile)) {
                const color = getAlertColor(overallCategory);
                const colorStyle = color ? `color: ${color}; font-weight: 600;` : 'font-weight: 600;';
                html += `<span class="tooltip-value" style="${colorStyle}">${overallPercentile.toFixed(1)}%</span>`;
            } else {
                html += '<span class="tooltip-value tooltip-ns">Not Scored</span>';
            }
            html += '</div>';
            
            // Active feature alerts
            const features = ['finished_trials', 'ignore_rate', 'total_trials', 'foraging_performance', 'abs(bias_naive)'];
            const featureNames = {
                'finished_trials': 'Finished Trials',
                'ignore_rate': 'Ignore Rate',
                'total_trials': 'Total Trials',
                'foraging_performance': 'Foraging Performance',
                'abs(bias_naive)': 'Bias'
            };
            
            let activeFeatures = [];
            features.forEach(feature => {
                const category = subjectData[feature + '_category'];
                const percentile = subjectData[feature + '_session_percentile'];
                
                if (category && category !== 'N' && category !== 'NS') {
                    activeFeatures.push({
                        name: featureNames[feature],
                        category: category,
                        percentile: percentile
                    });
                }
            });
            
            if (activeFeatures.length > 0) {
                html += '<div class="tooltip-features">';
                activeFeatures.forEach(feature => {
                    html += '<div class="tooltip-feature-item">';
                    html += `<span class="tooltip-feature-label">${feature.name}: </span>`;
                    
                    const percentileText = (feature.percentile != null && !isNaN(feature.percentile)) 
                        ? feature.percentile.toFixed(1) + '%' 
                        : 'N/A';
                    const color = getAlertColor(feature.category);
                    const colorStyle = color ? `color: ${color}; font-weight: 600;` : 'font-weight: 600;';
                    
                    html += `<span class="tooltip-feature-value" style="${colorStyle}">${percentileText}</span>`;
                    html += '</div>';
                });
                html += '</div>';
            }
            
            // Threshold alerts
            const thresholdAlert = subjectData.threshold_alert;
            if (thresholdAlert === 'T') {
                let thresholdAlerts = [];
                
                // Check specific threshold types
                const totalSessionsAlert = subjectData.total_sessions_alert || '';
                const stageSessionsAlert = subjectData.stage_sessions_alert || '';
                const waterDayAlert = subjectData.water_day_total_alert || '';
                
                if (totalSessionsAlert.includes('T |')) {
                    const value = totalSessionsAlert.split('|')[1]?.trim();
                    thresholdAlerts.push({
                        type: 'Total Sessions',
                        value: `${value} sessions`
                    });
                }
                
                if (stageSessionsAlert.includes('T |')) {
                    const parts = stageSessionsAlert.split('|');
                    const stage = parts[1]?.trim();
                    const sessions = parts[2]?.trim();
                    thresholdAlerts.push({
                        type: 'Stage Sessions',
                        value: `${stage}: ${sessions}`
                    });
                }
                
                if (waterDayAlert.includes('T |')) {
                    const value = waterDayAlert.split('|')[1]?.trim();
                    thresholdAlerts.push({
                        type: 'Water Day Total',
                        value: `${value} mL`
                    });
                }
                
                if (thresholdAlerts.length > 0) {
                    html += '<div class="tooltip-thresholds">';
                    thresholdAlerts.forEach(alert => {
                        html += '<div class="tooltip-threshold-item">';
                        html += `<span class="tooltip-threshold-label">${alert.type}: </span>`;
                        html += `<span class="tooltip-threshold-value" style="color: #795548; font-weight: 600;">${alert.value}</span>`;
                        html += '</div>';
                    });
                    html += '</div>';
                }
            }
            
            html += '</div>';
            return html;
        }
        
        // Function to show tooltip
        function showTooltip(subjectId, mouseX, mouseY) {
            if (!tooltip) return;
            
            // Clear any hide timeout
            if (hideTimeout) {
                clearTimeout(hideTimeout);
                hideTimeout = null;
            }
            
            // Find subject data in table
            const subjectData = table_data.find(row => row.subject_id === subjectId);
            if (!subjectData) return;
            
            // Create tooltip content
            const content = createTooltipContent(subjectData);
            tooltip.innerHTML = content;
            
            // Position tooltip near cursor but avoid edges
            const viewportWidth = window.innerWidth;
            const viewportHeight = window.innerHeight;
            const tooltipWidth = 220;
            const tooltipHeight = 150;
            
            let left = mouseX + 10;
            let top = mouseY - 10;
            
            if (left + tooltipWidth > viewportWidth) {
                left = mouseX - tooltipWidth - 10;
            }
            if (top + tooltipHeight > viewportHeight) {
                top = mouseY - tooltipHeight - 10;
            }
            if (left < 0) left = 10;
            if (top < 0) top = 10;
            
            // Set position and show
            tooltip.style.left = left + 'px';
            tooltip.style.top = top + 'px';
            tooltip.style.opacity = '1';
            tooltip.classList.remove('hidden');
            
            currentHoveredSubject = subjectId;
        }
        
        // Function to hide tooltip
        function hideTooltip() {
            if (!tooltip) return;
            
            hideTimeout = setTimeout(() => {
                tooltip.style.opacity = '0';
                tooltip.classList.add('hidden');
                currentHoveredSubject = null;
            }, 100);
        }
        
        // Setup hover listeners
        function setupHoverListeners() {
            const cells = table.querySelectorAll('td[data-dash-column="subject_id"]');
            
            cells.forEach(cell => {
                // Remove existing listeners
                if (cell._tooltipMouseEnter) cell.removeEventListener('mouseenter', cell._tooltipMouseEnter);
                if (cell._tooltipMouseLeave) cell.removeEventListener('mouseleave', cell._tooltipMouseLeave);
                if (cell._tooltipMouseMove) cell.removeEventListener('mousemove', cell._tooltipMouseMove);
                
                // Add new listeners
                cell._tooltipMouseEnter = function(e) {
                    const subjectId = this.textContent.trim();
                    if (subjectId && subjectId !== currentHoveredSubject) {
                        showTooltip(subjectId, e.clientX, e.clientY);
                    }
                };
                
                cell._tooltipMouseLeave = function(e) {
                    hideTooltip();
                };
                
                cell._tooltipMouseMove = function(e) {
                    const subjectId = this.textContent.trim();
                    if (subjectId === currentHoveredSubject && tooltip.style.opacity === '1') {
                        // Update position to follow cursor
                        const viewportWidth = window.innerWidth;
                        const viewportHeight = window.innerHeight;
                        const tooltipWidth = 220;
                        const tooltipHeight = 150;
                        
                        let left = e.clientX + 10;
                        let top = e.clientY - 10;
                        
                        if (left + tooltipWidth > viewportWidth) {
                            left = e.clientX - tooltipWidth - 10;
                        }
                        if (top + tooltipHeight > viewportHeight) {
                            top = e.clientY - tooltipHeight - 10;
                        }
                        if (left < 0) left = 10;
                        if (top < 0) top = 10;
                        
                        tooltip.style.left = left + 'px';
                        tooltip.style.top = top + 'px';
                    }
                };
                
                cell.addEventListener('mouseenter', cell._tooltipMouseEnter);
                cell.addEventListener('mouseleave', cell._tooltipMouseLeave);
                cell.addEventListener('mousemove', cell._tooltipMouseMove);
            });
        }
        
        // Setup listeners when table data changes
        setupHoverListeners();
        
        console.log('Tooltip setup complete');
        return window.dash_clientside.no_update;
    }
    """,
    Output("tooltip-setup-complete", "data"),
    [Input("session-table", "data"),
     Input("session-table", "id")]  # This acts as a trigger for setup
)

