"""
Loading indicator component for the AIND Dashboard
Shows a loading message while background data processing is ongoing
"""
import dash_bootstrap_components as dbc
from dash import html


class LoadingIndicator:
    """Component to show loading status for async operations"""

    def __init__(self):
        self.loading_messages = {
            "initial": "Loading AIND Dashboard...",
            "data_loading": "Loading session data...",
            "data_processing": "Processing data pipeline...", 
            "ui_optimizing": "Optimizing interface...",
            "complete": "Dashboard ready!"
        }

    def build_loading_spinner(self, message: str = None) -> html.Div:
        """
        Build a loading spinner with optional message
        
        Parameters:
            message: Optional custom loading message
            
        Returns:
            html.Div: Loading indicator component
        """
        display_message = message or self.loading_messages["initial"]
        
        return html.Div([
            dbc.Spinner(
                html.Div(id="loading-content"),
                color="primary",
                type="border",
                fullscreen=False,
                size="md"
            ),
            html.P(
                display_message,
                className="text-center mt-3 text-muted",
                style={"fontSize": "14px"}
            )
        ], 
        className="d-flex flex-column align-items-center justify-content-center",
        style={"minHeight": "200px", "padding": "20px"}
        )

    def build_loading_card(self, title: str = "Loading", message: str = None) -> dbc.Card:
        """
        Build a loading card with spinner and message
        
        Parameters:
            title: Card title
            message: Loading message
            
        Returns:
            dbc.Card: Loading card component
        """
        display_message = message or self.loading_messages["initial"]
        
        return dbc.Card([
            dbc.CardHeader(html.H5(title, className="mb-0")),
            dbc.CardBody([
                self.build_loading_spinner(display_message)
            ])
        ], className="text-center")

    def build_status_alert(self, status: str = "loading", custom_message: str = None) -> dbc.Alert:
        """
        Build a status alert for different loading phases
        
        Parameters:
            status: Loading status - "loading", "processing", "optimizing", "complete", "error"
            custom_message: Optional custom message
            
        Returns:
            dbc.Alert: Status alert component
        """
        status_config = {
            "loading": {"color": "info", "message": "Loading data..."},
            "processing": {"color": "warning", "message": "Processing data pipeline..."},
            "optimizing": {"color": "secondary", "message": "Optimizing interface..."},
            "complete": {"color": "success", "message": "Dashboard ready!"},
            "error": {"color": "danger", "message": "Error loading data"}
        }
        
        config = status_config.get(status, status_config["loading"])
        message = custom_message or config["message"]
        
        return dbc.Alert(
            message,
            color=config["color"],
            className="text-center mb-3"
        )