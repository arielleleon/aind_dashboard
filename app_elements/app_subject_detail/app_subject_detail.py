from dash import html, dcc
import dash_bootstrap_components as dbc
from .app_feature_chart import AppFeatureChart
from .app_session_list import AppSessionList
from .app_subject_timeseries import AppSubjectTimeseries

class AppSubjectDetail:
    def __init__(self):
        self.feature_chart = AppFeatureChart()
        self.session_list = AppSessionList()
        self.subject_timeseries = AppSubjectTimeseries()
        
    def build(self):
        """
        Build subject detail component
        """
        return html.Div([
            # Footer content (initially hidden)
            html.Div([
                dbc.Row([
                    # Left column: subject info
                    dbc.Col([
                        html.Div([
                            html.H2(id='detail-strata', className="subject-strata mb-3")
                        ], className="strata-container"),

                        dbc.Row([
                            # Subject info
                            dbc.Col([
                                html.Div([
                                    html.Div("Subject ID: ", className="detail-label"),
                                    html.Div(id="detail-subject-id", className="detail-value")
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

                        # Threshold alerts
                        html.Div([
                            html.Div("Threshold Alerts:", className="detail-label mb-2"),
                            html.Div(id="detail-threshold-alerts", className="detail-threshold")
                        ], className="detail-item threshold-container mb-3"),

                        # NS reason if applicable
                        html.Div(id="detail-ns-reason-container", className="ns-reason-container mb-3",
                                style={"display": "none"}, children=[
                            html.Div("Not Scored Reason:", className="detail-label mb-2"),
                            html.Div(id="detail-ns-reason", className="detail-ns-reason")
                        ]),

                        # View details button
                        dbc.Button(
                            "Show Subject Details Section",
                            id="view-details-button",
                            color="primary",
                            className="mt-2"
                        )
                    ], width=6),

                    # Right column: Feature chart
                    dbc.Col([
                        html.Div(id="feature-chart-container", className="chart-container"),
                    ], width=6),
                ], className="subject-detail-content")
            ], id="subject-detail-footer", className="subject-detail-section", style={"display": "none"}),

            # Subject detail page (initially hidden)
            html.Div([
                # Visual indicator to help users see that content is below
                html.Div([
                    html.I(className="fas fa-arrow-down mr-2"),
                    "Scroll down to view detailed subject information",
                ], className="text-center p-3 mb-3 bg-light rounded"),
                
                # Detailed subject page content
                html.Div([
                    html.H3("Subject History", className="mb-4"),
                    
                    # Main two-column layout
                    dbc.Row([
                        # Left column: Session list
                        dbc.Col([
                            self.session_list.build()
                        ], width=6, className="session-list-column"),
                        
                        # Right column: Timeseries graph
                        dbc.Col([
                            self.subject_timeseries.build()
                        ], width=6, className="timeseries-column"),
                    ], className="subject-detail-main-row")
                    
                ], id="subject-detail-page-content", className="detail-page-content")
            ], id="subject-detail-page", className="subject-detail-page-container", style={"display": "none"})
        ], id="subject-detail-container", className="subject-detail-container")