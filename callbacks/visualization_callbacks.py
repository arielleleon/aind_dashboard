"""
Visualization Callbacks Module

This module contains all plot and visualization-related callbacks:
- Timeseries plot updates
- Percentile timeseries plot updates
- Heatmap visualization and colorscale controls

Separated from main callbacks for better modularity and maintainability.
"""

from callbacks.shared_callback_utils import (
    ALL,
    Input,
    Output,
    State,
    app_utils,
    callback,
    extract_highlighted_session,
    percentile_heatmap,
    subject_percentile_timeseries,
    subject_timeseries,
)


@callback(
    Output("timeseries-plot", "figure"),
    [
        Input("timeseries-store", "data"),
        Input("timeseries-feature-dropdown", "value"),
        Input({"type": "session-card", "index": ALL}, "n_clicks"),
        Input("session-scroll-state", "data"),
    ],
    [State({"type": "session-card", "index": ALL}, "id")],
)
def update_timeseries_plot(
    timeseries_data, selected_features, n_clicks_list, scroll_state, card_ids
):
    """
    Update timeseries plot with data and session highlighting.
    """
    highlighted_session = extract_highlighted_session(
        n_clicks_list, scroll_state, card_ids
    )

    return subject_timeseries.create_plot(
        subject_data=timeseries_data,
        selected_features=selected_features or ["all"],
        highlighted_session=highlighted_session,
    )


@callback(
    Output("percentile-timeseries-plot", "figure"),
    [
        Input("timeseries-store", "data"),
        Input("percentile-timeseries-feature-dropdown", "value"),
        Input("percentile-ci-toggle", "value"),
        Input({"type": "session-card", "index": ALL}, "n_clicks"),
        Input("session-scroll-state", "data"),
    ],
    [State({"type": "session-card", "index": ALL}, "id")],
)
def update_percentile_timeseries_plot(
    timeseries_data,
    selected_features,
    ci_toggle_value,
    n_clicks_list,
    scroll_state,
    card_ids,
):
    """
    Update percentile timeseries plot with data, session highlighting, and confidence interval toggle.
    """
    # Determine if confidence intervals should be shown
    show_confidence_intervals = ci_toggle_value and "show_ci" in ci_toggle_value

    highlighted_session = extract_highlighted_session(
        n_clicks_list, scroll_state, card_ids
    )

    return subject_percentile_timeseries.create_plot(
        subject_data=timeseries_data,
        selected_features=selected_features or ["all"],
        highlighted_session=highlighted_session,
        show_confidence_intervals=show_confidence_intervals,
    )


@callback(
    Output("percentile-heatmap-container", "children"),
    [
        Input("selected-subject-store", "data"),
        Input({"type": "session-card", "index": ALL}, "n_clicks"),
        Input("session-scroll-state", "data"),
        Input("heatmap-colorscale-state", "data"),
    ],
    [State({"type": "session-card", "index": ALL}, "id")],
)
def update_percentile_heatmap_with_highlighting(
    selected_subject_data, n_clicks_list, scroll_state, colorscale_state, card_ids
):
    """
    Update percentile heatmap with session highlighting and colorscale mode.
    """
    # Extract subject_id from the store data
    subject_id = (
        selected_subject_data.get("subject_id") if selected_subject_data else None
    )

    if not subject_id:
        # Return empty heatmap if no subject selected
        return percentile_heatmap.build()

    # Get colorscale mode from state
    colorscale_mode = (
        colorscale_state.get("mode", "binned") if colorscale_state else "binned"
    )

    highlighted_session = extract_highlighted_session(
        n_clicks_list, scroll_state, card_ids
    )

    return percentile_heatmap.build(
        subject_id=subject_id,
        app_utils=app_utils,
        highlighted_session=highlighted_session,
        colorscale_mode=colorscale_mode,
    )


@callback(
    [
        Output("heatmap-colorscale-state", "data"),
        Output("heatmap-colorscale-toggle", "children"),
        Output("heatmap-colorscale-toggle", "color"),
    ],
    [Input("heatmap-colorscale-toggle", "n_clicks")],
    [State("heatmap-colorscale-state", "data")],
)
def toggle_heatmap_colorscale(n_clicks, current_state):
    """
    Toggle between binned and continuous colorscale modes for heatmap visualization.
    """
    # If button hasn't been clicked, return current state
    if not n_clicks:
        current_mode = (
            current_state.get("mode", "binned") if current_state else "binned"
        )
        button_text = "Binned" if current_mode == "binned" else "Continuous"
        button_color = (
            "outline-secondary" if current_mode == "binned" else "outline-primary"
        )
        return current_state or {"mode": "binned"}, button_text, button_color

    # Toggle the mode
    current_mode = current_state.get("mode", "binned") if current_state else "binned"
    new_mode = "continuous" if current_mode == "binned" else "binned"

    # Update button appearance based on new mode
    button_text = "Continuous" if new_mode == "continuous" else "Binned"
    button_color = (
        "outline-primary" if new_mode == "continuous" else "outline-secondary"
    )

    return {"mode": new_mode}, button_text, button_color
