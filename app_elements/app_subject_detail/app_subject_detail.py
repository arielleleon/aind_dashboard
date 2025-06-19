import dash_bootstrap_components as dbc
from dash import dcc, html

from .app_session_list import AppSessionList
from .app_subject_compact_info import AppSubjectCompactInfo
from .app_subject_percentile_heatmap import AppSubjectPercentileHeatmap
from .app_subject_percentile_timeseries import AppSubjectPercentileTimeseries
from .app_subject_timeseries import AppSubjectTimeseries


class AppSubjectDetail:
    def __init__(self):
        self.session_list = AppSessionList()
        self.subject_timeseries = AppSubjectTimeseries()
        self.subject_percentile_timeseries = AppSubjectPercentileTimeseries()
        self.percentile_heatmap = AppSubjectPercentileHeatmap()
        self.compact_info = AppSubjectCompactInfo()

    def build(self):
        """
        Build subject detail component with layout:
        - Compact info spanning full width
        - Heatmap spanning both columns (including overall percentile)
        - Session list and timeseries below
        """
        return html.Div(
            [
                # Store component to track selected session card
                dcc.Store(id="session-card-selected", data={"selected_card_id": None}),
                # Store component to track selected subject ID
                dcc.Store(id="selected-subject-store", data={"subject_id": None}),
                # Interval component to check for scroll updates (300ms interval)
                dcc.Interval(id="scroll-tracker-interval", interval=300, n_intervals=0),
                # Subject detail page (initially shown when subject is selected)
                html.Div(
                    [
                        # Top section with compact info and heatmap spanning full width
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        # Compact subject info
                                        html.Div(
                                            id="compact-subject-info-container",
                                            className="compact-info-section mb-3",
                                        ),
                                        # Percentile heatmap
                                        html.Div(
                                            [
                                                # Title row with toggle button
                                                html.Div(
                                                    [
                                                        html.H6(
                                                            "Feature Percentile Progress Over Time",
                                                            className="heatmap-title mb-2",
                                                        ),
                                                        html.Div(
                                                            [
                                                                dbc.Button(
                                                                    "Binned",
                                                                    id="heatmap-colorscale-toggle",
                                                                    size="sm",
                                                                    color="outline-secondary",
                                                                    className="heatmap-toggle-btn",
                                                                    style={
                                                                        "fontSize": "12px",
                                                                        "padding": "4px 12px",
                                                                        "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif",
                                                                        "borderRadius": "6px",
                                                                        "minWidth": "80px",
                                                                    },
                                                                ),
                                                                # Store to track toggle state
                                                                dcc.Store(
                                                                    id="heatmap-colorscale-state",
                                                                    data={
                                                                        "mode": "binned"
                                                                    },
                                                                ),
                                                            ],
                                                            className="heatmap-toggle-container",
                                                        ),
                                                    ],
                                                    className="heatmap-header-row d-flex justify-content-between align-items-center mb-2",
                                                ),
                                                html.Div(
                                                    id="percentile-heatmap-container",
                                                    className="heatmap-container",
                                                ),
                                            ],
                                            className="heatmap-section",
                                        ),
                                    ],
                                    width=12,
                                    className="heatmap-full-width-column",
                                ),
                            ],
                            className="charts-summary-row mb-4",
                        ),
                        # Main two-column layout - session list and dual timeseries plots
                        dbc.Row(
                            [
                                # Left column: Session list
                                dbc.Col(
                                    [self.session_list.build()],
                                    width=6,
                                    className="session-list-column",
                                ),
                                # Right column: Dual timeseries graphs
                                dbc.Col(
                                    [
                                        # Raw values timeseries
                                        html.Div(
                                            [self.subject_timeseries.build()],
                                            className="raw-timeseries-section mb-4",
                                        ),
                                        # Percentile timeseries
                                        html.Div(
                                            [
                                                self.subject_percentile_timeseries.build()
                                            ],
                                            className="percentile-timeseries-section",
                                        ),
                                    ],
                                    width=6,
                                    className="timeseries-column",
                                ),
                            ],
                            className="subject-detail-main-row",
                        ),
                        # Footer row
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div(
                                            [
                                                html.Hr(className="footer-separator"),
                                                html.Div(
                                                    "End of subject analysis",
                                                    className="footer-text text-center text-muted",
                                                ),
                                            ],
                                            className="subject-detail-footer-content",
                                        )
                                    ],
                                    width=12,
                                    className="subject-detail-footer-column",
                                )
                            ],
                            className="subject-detail-footer-row mt-3",
                        ),
                    ],
                    id="subject-detail-page",
                    className="subject-detail-page-container",
                    style={"display": "none"},
                ),
            ],
            id="subject-detail-container",
            className="subject-detail-container",
        )
