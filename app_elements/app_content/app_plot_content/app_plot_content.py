from dash import html, dcc
from .app_rank_change_plot import RankChangePlot

class AppPlotContent:
    def __init__(self):
        """Initialize plot content with rank change plot only"""
        self.rank_change_plot = RankChangePlot()

    def build(self):
        """
        Build app plot content with rank change plot only
        """
        return html.Div([
            dcc.Graph(
                id='rank-change-plot',
                style={
                    'height': '100%',  # Fill container height
                    'width': '100%'    # Fill container width
                },
                config={'responsive': True}
            )
        ], className="rank-plot-wrapper", style={
            'height': '100%',
            'width': '100%'
        })