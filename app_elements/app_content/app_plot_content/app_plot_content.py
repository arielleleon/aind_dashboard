from dash import html
from dash import dcc

class AppPlotContent:

    def build(self):
        """
        Build app plot content
        """
        return html.Div(
            [
                dcc.Graph(id = 'plot-content')
            ]
        )
        