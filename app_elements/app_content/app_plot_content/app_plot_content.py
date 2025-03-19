from dash import html
from dash import dcc

class AppPlotContent:

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