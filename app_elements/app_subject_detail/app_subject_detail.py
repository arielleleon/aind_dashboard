from dash import html, dcc
import dash_bootstrap_components as dbc
from .app_feature_chart import AppFeatureChart
from .app_session_list import AppSessionList
from .app_subject_timeseries import AppSubjectTimeseries
from .app_subject_percentile_timeseries import AppSubjectPercentileTimeseries

class AppSubjectDetail:
    def __init__(self):
        self.feature_chart = AppFeatureChart()
        self.session_list = AppSessionList()
        self.subject_timeseries = AppSubjectTimeseries()
        self.subject_percentile_timeseries = AppSubjectPercentileTimeseries()
        
    def build(self):
        """
        Build subject detail component
        """
        return html.Div([
            
            # Store component to track selected session card
            dcc.Store(id="session-card-selected", data={"selected_card_id": None}),
            
            # Interval component to check for scroll updates (300ms interval - reduced frequency)
            dcc.Interval(id="scroll-tracker-interval", interval=300, n_intervals=0),
            
            # Footer content (initially hidden)
            html.Div([
                dbc.Row([
                    # Left column: subject info
                    dbc.Col([
                        # Subject ID 
                        html.Div([
                            html.H2(id='detail-subject-id', className="subject-strata mb-3")
                        ], className="subject-id-container"),

                        dbc.Row([
                            # Subject info 
                            dbc.Col([
                                html.Div([
                                    html.Div("Strata: ", className="detail-label"),
                                    html.Div(id="detail-strata", className="detail-value")
                                ], className="detail-item"),

                                html.Div([
                                    html.Div("PI: ", className="detail-label"),
                                    html.Div(id="detail-pi", className="detail-value")
                                ], className="detail-item"),

                                html.Div([
                                    html.Div("Trainer: ", className="detail-label"),
                                    html.Div(id="detail-trainer", className="detail-value")
                                ], className="detail-item"),
                            ], width=6),

                            # Session and performance information
                            dbc.Col([
                                html.Div([
                                    html.Div("Overall Percentile:", className="detail-label"),
                                    html.Div(id="detail-percentile", className="detail-value")
                                ], className="detail-item"),

                                html.Div([
                                    html.Div("Last Session:", className="detail-label"),
                                    html.Div(id="detail-last-session", className="detail-value")
                                ], className="detail-item"),

                                html.Div([
                                    html.Div("Training Consistency:", className="detail-label"),
                                    html.Div(id="detail-consistency", className="detail-value")
                                ], className="detail-item"),
                            ], width=6),
                        ], className="mb-3"),

                        # Threshold alerts - removed the border container
                        html.Div([
                            html.Div("Threshold Alerts:", className="detail-label mb-2"),
                            html.Div(id="detail-threshold-alerts", className="detail-value")
                        ], className="detail-item mb-3"),

                        # NS reason if applicable
                        html.Div(id="detail-ns-reason-container", className="ns-reason-container mb-3",
                                style={"display": "none"}, children=[
                            html.Div("Not Scored Reason:", className="detail-label mb-2"),
                            html.Div(id="detail-ns-reason", className="detail-ns-reason")
                        ]),

                        # View details button removed
                    ], width=6),

                    # Right column: Feature chart (NOT timeseries)
                    dbc.Col([
                        html.Div(id="feature-chart-container", className="feature-chart-no-border"),
                    ], width=6),
                ], className="subject-detail-content")
            ], id="subject-detail-footer", className="subject-detail-section", style={"display": "none"}),

            # Subject detail page (initially shown when subject is selected)
            html.Div([                
                # Main two-column layout - session list and dual timeseries plots
                dbc.Row([
                    # Left column: Session list
                    dbc.Col([
                        self.session_list.build()
                    ], width=6, className="session-list-column"),
                    
                    # Right column: Dual timeseries graphs
                    dbc.Col([
                        # Raw values timeseries (existing)
                        html.Div([
                            html.H5("Raw Feature Values (3-Session Rolling Average)", className="timeseries-title mb-2"),
                            self.subject_timeseries.build()
                        ], className="raw-timeseries-section mb-4"),
                        
                        # Percentile timeseries (new)
                        html.Div([
                            html.H5("Feature Percentiles", className="timeseries-title mb-2"),
                            self.subject_percentile_timeseries.build()
                        ], className="percentile-timeseries-section")
                    ], width=6, className="timeseries-column"),
                ], className="subject-detail-main-row")
            ], id="subject-detail-page", className="subject-detail-page-container", style={"display": "none"})
        ], id="subject-detail-container", className="subject-detail-container")