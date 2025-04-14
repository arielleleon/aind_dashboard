from dash import Input, Output, State, callback, ALL, MATCH, ctx
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import datetime
from app_utils import AppUtils
from app_elements.app_content.app_dataframe.app_dataframe import AppDataFrame
from app_elements.app_filter.app_filter import AppFilter
from app_elements.app_content.app_plot_content.app_rank_change_plot import RankChangePlot
import plotly.graph_objects as go

# Initialize app utilities
app_utils = AppUtils()

# Initialize dataframe formatter
app_dataframe = AppDataFrame()

# Initialize AppFilter to access time_window_options
app_filter = AppFilter()

# Initialize rank change plot
rank_change_plot = RankChangePlot()

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
    
    # Get the full dataset
    df = app_utils.get_session_data()
    
    # Create a fresh dataframe formatter to avoid state issues
    app_dataframe = AppDataFrame()
    
    # Apply complete formatting with alert population
    formatted_df = app_dataframe.format_dataframe(df, window_days=time_window_value)
    
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
            formatted_df = formatted_df[formatted_df["threshold_alert"] == "T"]
        else:
            # For other alert categories, use combined_alert field
            formatted_df = formatted_df[formatted_df["combined_alert"].str.contains(alert_category)]
    
    # Apply sorting based on selected option
    if sort_option != "none":
        if sort_option == "percentile_asc":
            # Sort by overall percentile (ascending)
            formatted_df = formatted_df.sort_values(by="overall_percentile", ascending=True)
        elif sort_option == "percentile_desc":
            # Sort by overall percentile (descending)
            formatted_df = formatted_df.sort_values(by="overall_percentile", ascending=False)
        elif sort_option == "alert_worst":
            # Define alert category order from worst to best
            alert_order = {"SB": 0, "B": 1, "N": 2, "G": 3, "SG": 4, "NS": 5}
            # Create a temporary column for sorting
            formatted_df["alert_sort"] = formatted_df["percentile_category"].map(alert_order)
            # Sort by alert category (worst first)
            formatted_df = formatted_df.sort_values(by="alert_sort", ascending=True)
            # Remove temporary sort column
            formatted_df = formatted_df.drop(columns=["alert_sort"])
        elif sort_option == "alert_best":
            # Define alert category order from best to worst
            alert_order = {"SG": 0, "G": 1, "N": 2, "B": 3, "SB": 4, "NS": 5}
            # Create a temporary column for sorting
            formatted_df["alert_sort"] = formatted_df["percentile_category"].map(alert_order)
            # Sort by alert category (best first)
            formatted_df = formatted_df.sort_values(by="alert_sort", ascending=True)
            # Remove temporary sort column
            formatted_df = formatted_df.drop(columns=["alert_sort"])
    
    # Count and print alert statistics
    percentile_categories = formatted_df['percentile_category'].value_counts().to_dict()
    print(f"Percentile categories: {percentile_categories}")
    
    return formatted_df.to_dict('records')

# Update the rank change plot based on time window filter
@callback(
    Output("rank-change-plot", "figure"),
    [Input("time-window-filter", "value")]
)
def update_rank_change_plot(time_window_value):
    """
    Update the rank change plot based on the selected time window
    
    Parameters:
        time_window_value (int): Number of days to include in the analysis window
        
    Returns:
        go.Figure: The updated rank change plot
    """
    # Generate the rank change plot using the specified time window
    return rank_change_plot.build(window_days=time_window_value)