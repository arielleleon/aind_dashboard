from dash import Input, Output, State, callback, ALL, ctx
from dash import html
import pandas as pd
from shared_utils import app_utils
from app_elements.app_content.app_dataframe.column_groups_config import (
    COLUMN_GROUPS, get_columns_for_groups, get_default_visible_columns
)

# Import the AppDataFrame instance
from app_elements.app_content.app_dataframe import AppDataFrame

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
    formatted_column_names = {
        'subject_id': 'Subject ID',
        'combined_alert': 'Alert',
        'percentile_category': 'Percentile Alert', 
        'ns_reason': 'Not Scored Reason',
        'threshold_alert': 'Threshold Alert',
        'total_sessions_alert': 'Total Sessions Alert',
        'stage_sessions_alert': 'Stage Sessions Alert',
        'water_day_total_alert': 'Water Day Total Alert',
        'overall_percentile': 'Overall\nPercentile',
        'session_overall_percentile': 'Session\nPercentile',
        'strata': 'Strata',
        'strata_abbr': 'Strata (Abbr)',
        'current_stage_actual': 'Stage',
        'curriculum_name': 'Curriculum',
        'session_date': 'Date',
        'session': 'Session',
        'rig': 'Rig',
        'trainer': 'Trainer',
        'PI': 'PI',
        'session_run_time': 'Run Time',
        'total_trials': 'Total Trials',
        'finished_trials': 'Finished Trials',
        'finished_rate': 'Finish Rate',
        'ignore_rate': 'Ignore Rate',
        'water_in_session_foraging': 'Water In-Session\n(Foraging)',
        'water_in_session_manual': 'Water In-Session\n(Manual)',
        'water_in_session_total': 'Water In-Session\n(Total)',
        'water_after_session': 'Water After\nSession',
        'water_day_total': 'Water Day\nTotal',
        'base_weight': 'Base Weight',
        'target_weight': 'Target Weight',
        'target_weight_ratio': 'Target Weight\nRatio',
        'weight_after': 'Weight After',
        'weight_after_ratio': 'Weight After\nRatio',
        'reward_volume_left_mean': 'Reward Volume\nLeft (Mean)',
        'reward_volume_right_mean': 'Reward Volume\nRight (Mean)',
        'reaction_time_median': 'Reaction Time\n(Median)',
        'reaction_time_mean': 'Reaction Time\n(Mean)',
        'early_lick_rate': 'Early Lick\nRate',
        'invalid_lick_ratio': 'Invalid Lick\nRatio',
        'double_dipping_rate_finished_trials': 'Double Dipping Rate\n(Finished Trials)',
        'double_dipping_rate_finished_reward_trials': 'Double Dipping Rate\n(Reward Trials)',
        'double_dipping_rate_finished_noreward_trials': 'Double Dipping Rate\n(No Reward Trials)',
        'lick_consistency_mean_finished_trials': 'Lick Consistency\n(Finished Trials)',
        'lick_consistency_mean_finished_reward_trials': 'Lick Consistency\n(Reward Trials)',
        'lick_consistency_mean_finished_noreward_trials': 'Lick Consistency\n(No Reward Trials)',
        'avg_trial_length_in_seconds': 'Avg Trial Length\n(Seconds)',
        'total_trials_with_autowater': 'Total Trials\n(Autowater)',
        'finished_trials_with_autowater': 'Finished Trials\n(Autowater)',
        'finished_rate_with_autowater': 'Finish Rate\n(Autowater)',
        'ignore_rate_with_autowater': 'Ignore Rate\n(Autowater)',
        'autowater_collected': 'Autowater\nCollected',
        'autowater_ignored': 'Autowater\nIgnored',
        'water_day_total_last_session': 'Water Day Total\n(Last Session)',
        'water_after_session_last_session': 'Water After\n(Last Session)'
    }
    
    # Add feature-specific column names
    features_config = {
        'finished_trials': False,  # Higher is better
        'ignore_rate': True,     # Lower is better
        'total_trials': False,   # Higher is better
        'foraging_performance': False,   # Higher is better
        'abs(bias_naive)': True  # Lower is better 
    }
    
    for feature in features_config.keys():
        feature_display = feature.replace('_', ' ').replace('abs(', '|').replace(')', '|').title()
        formatted_column_names[f'{feature}_percentile'] = f'{feature_display}\nStrata %ile'
        formatted_column_names[f'{feature}_category'] = f'{feature_display}\nAlert'
        formatted_column_names[f'{feature}_processed'] = f'{feature_display}\nProcessed'
        formatted_column_names[f'{feature}_session_percentile'] = f'{feature_display}\nSession %ile'
        formatted_column_names[f'{feature}_processed_rolling_avg'] = f'{feature_display}\nRolling Avg'
    
    # Add overall percentile columns
    formatted_column_names['session_overall_percentile'] = 'Session Overall\nPercentile'
    formatted_column_names['overall_percentile'] = 'Strata Overall\nPercentile'
    
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