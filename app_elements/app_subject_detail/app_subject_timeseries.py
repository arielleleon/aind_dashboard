from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

class AppSubjectTimeseries:
    def __init__(self):
        """ Initialize timeseries component """
        pass 
    def build(self):
        """ Build timeseries component """
        return html.Div([
            dcc.Graph(
                id="subject-timeseries-graph",
                figure=go.Figure(),
                config = {'displayModeBar': True},
                className="subject-timeseries-graph"
            )
        ], id="subject-timeseries-container", className="subject-timeseries-container")