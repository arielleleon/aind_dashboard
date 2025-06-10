from callbacks.shared_callback_utils import (
    Input, Output, State, callback, ALL, ctx, clientside_callback,
    html, pd, app_utils, build_formatted_column_names
)
from app_elements.app_content.app_dataframe.column_groups_config import (
    COLUMN_GROUPS, get_columns_for_groups, get_default_visible_columns
)
from app_elements.app_content.app_dataframe.app_dataframe import AppDataFrame

# Column Toggle Callbacks

@callback(
    [Output("column-groups-state", "data"),
     Output({'type': 'column-group-toggle', 'group': ALL}, "color")],
    [Input({'type': 'column-group-toggle', 'group': ALL}, "n_clicks")],
    [State("column-groups-state", "data"),
     State({'type': 'column-group-toggle', 'group': ALL}, "id")]
)
def toggle_column_groups(n_clicks_list, current_state, button_ids):
    """
    Handle column group toggle button clicks
    
    This callback updates the state of expanded/collapsed column groups
    and returns the updated state plus button colors.
    """
    print(f" Column toggle callback triggered. n_clicks: {n_clicks_list}")
    print(f"   Current state: {current_state}")
    print(f"   Button IDs: {[btn['group'] for btn in button_ids]}")
    
    # Initialize state if empty - ALL GROUPS START COLLAPSED
    if not current_state:
        current_state = {}
        for group_id, group_config in COLUMN_GROUPS.items():
            if group_config.get('collapsible', True):
                # Always start collapsed, ignoring default_expanded
                current_state[group_id] = False
            else:
                current_state[group_id] = True  # Non-collapsible groups stay visible
    
    # Find which button was clicked (only if any clicks occurred)
    from dash import callback_context
    
    if callback_context.triggered and any(n_clicks_list or []):
        triggered_prop_id = callback_context.triggered[0]['prop_id']
        
        # Parse the button ID from the triggered property
        if 'column-group-toggle' in triggered_prop_id and triggered_prop_id != '.':
            try:
                import json
                id_part = triggered_prop_id.split('.')[0]
                button_id = json.loads(id_part)
                group_id = button_id['group']
                
                # Toggle the state for this group
                current_state[group_id] = not current_state.get(group_id, False)
                print(f"   Toggled group '{group_id}' to: {current_state[group_id]}")
            except (json.JSONDecodeError, KeyError, IndexError):
                print(f"   Error parsing button ID: {triggered_prop_id}")
                pass
    
    # Generate button colors based on current state
    button_colors = []
    for button_id in button_ids:
        group_id = button_id['group']
        is_expanded = current_state.get(group_id, False)
        color = "primary" if is_expanded else "outline-secondary"
        button_colors.append(color)
    
    return current_state, button_colors


@callback(
    Output("session-table", "columns"),
    [Input("column-groups-state", "data")]
)
def update_table_columns(column_groups_state):
    """
    Update the DataTable columns based on which column groups are expanded
    
    This callback is triggered when the column groups state changes and
    rebuilds the table column definitions accordingly.
    """
    print(f"📊 Updating table columns. State: {column_groups_state}")
    
    if not column_groups_state:
        # Use default visible columns if no state
        visible_column_ids = get_default_visible_columns()
        print(f"   Using default columns: {len(visible_column_ids)} columns")
    else:
        # Get expanded groups
        expanded_groups = [group_id for group_id, is_expanded in column_groups_state.items() if is_expanded]
        visible_column_ids = get_columns_for_groups(expanded_groups)
        print(f"   Expanded groups: {expanded_groups}")
        print(f"   Visible columns: {len(visible_column_ids)} columns")
    
    # Get all available data to create column definitions
    raw_data = app_utils.get_session_data(use_cache=True)
    app_dataframe = AppDataFrame(app_utils=app_utils)
    all_table_data = app_dataframe.format_dataframe(raw_data)
    
    # Create column definitions with formatting
    formatted_column_names = build_formatted_column_names()
    
    # Feature-specific columns are already included in build_formatted_column_names()
    # Removed duplicate feature column name logic as it's now centralized
    
    # Overall percentile columns are already included in build_formatted_column_names()
    # Removed duplicate overall percentile assignment
    
    # Build column definitions for visible columns only
    visible_columns = []
    for col_id in visible_column_ids:
        if col_id in all_table_data.columns:
            column_def = {
                "name": formatted_column_names.get(col_id, col_id.replace('_', ' ').title()),
                "id": col_id
            }
            
            # Add specific formatting for float columns
            if all_table_data[col_id].dtype == 'float64':
                column_def['type'] = 'numeric'
                column_def['format'] = {"specifier": ".5~g"}
            
            visible_columns.append(column_def)
        else:
            print(f"   WARNING: Column '{col_id}' not found in data!")
    
    print(f"   Final visible columns: {[col['id'] for col in visible_columns]}")
    return visible_columns


# Responsive Table Callbacks

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
        
        console.log(` Responsive table: viewport ${viewportHeight}px → showing ${optimalRows} rows`);
        
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