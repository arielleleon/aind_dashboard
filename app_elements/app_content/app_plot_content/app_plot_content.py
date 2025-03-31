from dash import html, dcc
from .app_subject_heatmap import SubjectHeatmap
from .app_rank_change_plot import RankChangePlot

class AppPlotContent:
    def __init__(self):
        """Initialize plot content with subject heatmap and rank change plot"""
        self.subject_heatmap = SubjectHeatmap()
        self.rank_change_plot = RankChangePlot()

    def build(self):
        """
        Build app plot content with proper sizing to fill the column equally
        """
        return html.Div([
            # Container for both plots with flex layout
            html.Div([
                # Rank Change Plot (top)
                html.Div([
                    dcc.Graph(
                        id='rank-change-plot',
                        style={
                            'height': '100%',  # Fill container height
                            'width': '100%'    # Fill container width
                        },
                        config={'responsive': True}
                    )
                ], style={
                    'flex': '0 0 35%',  # Take 35% of the available height (fixed)
                    'marginBottom': '10px',
                    'position': 'relative'
                }, className="plot-container"),
                
                # Main Plot - Heatmap (bottom)
                html.Div([
                    dcc.Graph(
                        id='plot-content',
                        style={
                            'height': '100%',  # Fill container height
                            'width': '100%'    # Fill container width
                        },
                        config={'responsive': True}
                    )
                ], style={
                    'flex': '1',  # Take remaining space (flexible)
                    'position': 'relative'
                }, className="plot-container")
            ], style={
                'display': 'flex',
                'flexDirection': 'column',
                'height': 'calc(100vh - 80px)',  # Full height minus header and margins
                'width': '100%'
            }, className="plots-wrapper")
        ], className="plot-content-wrapper")