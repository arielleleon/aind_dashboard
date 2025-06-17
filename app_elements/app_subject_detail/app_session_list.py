import dash_bootstrap_components as dbc
from dash import dcc, html

from .app_session_card import AppSessionCard


class AppSessionList:
    def __init__(self):
        """Initialize session list component"""
        self.session_card = AppSessionCard()

    def build(self):
        """Scrollable session list component"""
        return html.Div(
            [
                # Header section
                html.Div(
                    [
                        html.H4("Session History", className="session-list-title"),
                        html.Div(
                            [
                                html.Span(
                                    id="session-count", className="session-count"
                                ),
                                html.Span(
                                    " sessions displayed",
                                    className="session-count-label",
                                ),
                            ],
                            className="session-count-container",
                        ),
                    ],
                    className="session-list-header",
                ),
                # Scrollable container for session cards
                html.Div(
                    [
                        html.Div(
                            id="session-list-container",
                            className="session-list-container",
                        )
                    ],
                    id="session-list-scroll-container",
                    className="session-list-scroll-container",
                ),
                # Store component to track session loading state
                dcc.Store(
                    id="session-list-state",
                    data={
                        "subject_id": None,
                        "sessions_loaded": 0,
                        "total_sessions": 0,
                    },
                ),
                # Store component to track visible session during scrolling
                dcc.Store(id="session-scroll-state", data={"visible_session": None}),
            ],
            id="session-list-wrapper",
            className="session-list-wrapper",
        )
