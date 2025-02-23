from dash import html
from .app_content import *

class AppMain:

    def build(self):
        """
        Build app main
        """
        return html.Div(
            [
                AppContent().build()
            ],
            className = 'app-main'
        )