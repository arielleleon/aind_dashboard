from dash import html


class AppHoverTooltip:
    """
    Hover tooltip component for displaying subject information
    Shows strata, overall percentile, active feature alerts, and threshold alerts
    """

    def __init__(self, app_utils=None):
        """
        Initialize tooltip component

        Parameters:
            app_utils: Shared AppUtils instance for data access
        """
        self.app_utils = app_utils

        # Color mapping matching DataTable alert schema
        self.alert_colors = {
            "SB": "#FF6B35",  # Dark orange (Severely Below)
            "B": "#FFB366",  # Light orange (Below)
            "G": "#4A90E2",  # Light blue (Good)
            "SG": "#2E5A87",  # Dark blue (Severely Good)
            "N": None,  # Normal (no color)
            "NS": None,  # Not Scored (no color)
            "T": "#795548",  # Brown (Threshold alerts)
        }

    def build_tooltip_container(self) -> html.Div:
        """
        Build the tooltip container that will be positioned dynamically

        Returns:
            html.Div: Empty tooltip container for dynamic content
        """
        return html.Div(
            id="subject-hover-tooltip",
            className="subject-tooltip hidden",
            style={
                "position": "absolute",
                "zIndex": 9999,
                "pointerEvents": "none",  # Don't interfere with mouse events
                "opacity": 0,
                "transition": "opacity 0.15s ease-in-out",
            },
        )

    def get_empty_tooltip(self) -> html.Div:
        """
        Return empty tooltip content

        Returns:
            html.Div: Empty div for hiding tooltip
        """
        return html.Div()
