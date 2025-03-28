from dash import html
from dash import dcc
from .app_subject_heatmap import SubjectHeatmap

class AppPlotContent:
    def __init__(self):
        """Initialize plot content with subject heatmap"""
        self.subject_heatmap = SubjectHeatmap()

    def build(self):
        """
        Build app plot content with proper sizing
        """
        return html.Div(
            [
                dcc.Graph(
                    id='plot-content',
                    style={
                        'height': 'calc(100vh - 250px)',  # Match with table height
                        'minHeight': '400px'  # Add minimum height
                    },
                    config={'responsive': True}
                )
            ],
            className="plot-content-wrapper"
        )