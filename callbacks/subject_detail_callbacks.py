"""
Subject Detail Callbacks

This module handles callbacks related to subject selection and data loading:
- Subject selection from session table
- Session list population for selected subject
- Timeseries data loading for visualizations

Uses Dash 2.x @callback decorator pattern to avoid circular imports.
"""

# Import shared utilities (replaces multiple individual imports)
from callbacks.shared_callback_utils import (
    Input,
    Output,
    State,
    app_utils,
    callback,
    compact_info,
    safe_extract_subject_data,
    session_card,
)

# Component instances are now shared from utilities - no need to re-initialize
# session_card = AppSessionCard()  # Removed - using shared instance
# compact_info = AppSubjectCompactInfo()  # Removed - using shared instance


@callback(
    [
        Output("subject-detail-page", "style"),
        Output("compact-subject-info-container", "children"),
        Output("selected-subject-store", "data"),
    ],
    [Input("session-table", "active_cell")],
    [
        State("session-table", "data"),
        State("session-table", "page_current"),
        State("session-table", "page_size"),
    ],
)
def update_subject_detail(active_cell, table_data, page_current, page_size):
    """
    Handle subject selection from session table.

    When a user clicks on a subject_id cell in the table, this callback:
    1. Shows the subject detail page
    2. Builds compact subject info component
    3. Stores the selected subject ID for other callbacks
    """
    print("Subject detail callback triggered")
    print(f"Active cell: {active_cell}")
    print(f"Current page: {page_current}, Page size: {page_size}")

    default_page_style = {"display": "none"}
    empty_compact_info = compact_info.build()
    empty_subject_store = {"subject_id": None}

    # Check if cell is clicked and it's the subject_id column
    if not active_cell or active_cell["column_id"] != "subject_id":
        print("No active cell or not in subject_id column")
        return default_page_style, empty_compact_info, empty_subject_store

    # Use shared safe data extraction function
    subject_data, subject_id = safe_extract_subject_data(
        table_data, active_cell, page_current, page_size
    )

    if not subject_data or not subject_id:
        return default_page_style, empty_compact_info, empty_subject_store

    print(f"Selected subject: {subject_id}")

    # Build compact subject info component
    print(f"Building components for subject: {subject_id}")

    try:
        compact_subject_info = compact_info.build(
            subject_id=subject_id, app_utils=app_utils
        )

    except Exception as e:
        print(f"Error building components: {str(e)}")
        compact_subject_info = empty_compact_info

    return (
        {"display": "block"},  # Show detail page
        compact_subject_info,  # Compact info component
        {"subject_id": subject_id},  # Store the selected subject ID
    )


@callback(
    [
        Output("session-list-container", "children"),
        Output("session-list-state", "data"),
        Output("session-count", "children"),
    ],
    [Input("selected-subject-store", "data")],
)
def update_session_list(selected_subject_data):
    """
    Populate session list when a subject is selected.

    Loads all sessions for the selected subject and creates session cards.
    Sessions are sorted by date (descending) with the most recent active.
    """
    # Extract subject_id from the store data
    subject_id = (
        selected_subject_data.get("subject_id") if selected_subject_data else None
    )

    print(f"Session list callback triggered with subject_id: {subject_id}")

    # Default return values
    empty_list = []
    updated_state = {"subject_id": None, "sessions_loaded": 0, "total_sessions": 0}
    session_count = "0"

    # If no subject selected, return defaults
    if not subject_id:
        print("No subject selected - returning empty session list")
        return empty_list, updated_state, session_count

    # Get all sessions for this subject using app_utils
    all_sessions = app_utils.get_subject_sessions(subject_id)

    if all_sessions is None or all_sessions.empty:
        return empty_list, updated_state, session_count

    # Sort by session date (descending) and load ALL sessions
    all_sessions = all_sessions.sort_values("session_date", ascending=False)

    print(f"Loading all {len(all_sessions)} sessions for subject {subject_id}")

    # Build session cards for ALL sessions
    session_cards = []
    for idx, session_row in all_sessions.iterrows():
        # First card (most recent) is active by default
        is_active = idx == 0

        # Create session card using the session card component
        card = session_card.build(session_row.to_dict(), is_active=is_active)
        session_cards.append(card)

    # Update state with all sessions loaded
    updated_state = {
        "subject_id": subject_id,
        "sessions_loaded": len(all_sessions),
        "total_sessions": len(all_sessions),
    }

    # Update session count for UI display
    session_count = str(len(all_sessions))

    return session_cards, updated_state, session_count


@callback(Output("timeseries-store", "data"), [Input("selected-subject-store", "data")])
def load_timeseries_data(selected_subject_data):
    """
    Load optimized timeseries data for selected subject.

    This callback loads and caches time series data that will be used
    by multiple visualization components (raw timeseries, percentile plots, etc.).
    Using a store pattern to avoid redundant data loading.
    """
    if not selected_subject_data or not selected_subject_data.get("subject_id"):
        return {}

    subject_id = selected_subject_data["subject_id"]

    # Get optimized time series data from app_utils with caching
    time_series_data = app_utils.get_time_series_data(subject_id, use_cache=True)

    if not time_series_data:
        print(f"No timeseries data found for subject {subject_id}")
        return {}

    print(
        f"Loaded timeseries data for {subject_id}: {len(time_series_data.get('sessions', []))} sessions"
    )
    return time_series_data
