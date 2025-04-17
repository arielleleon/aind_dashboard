from dash import html, Output, Input, State, callback, clientside_callback
import dash_bootstrap_components as dbc
import json 
import pandas as pd
from .subject_tooltip import SubjectTooltip
from .subject_tooltip_service import SubjectTooltipService

class TooltipController:
    """ Controller for managing tooltips in the app """
    
    def __init__(self):
        """ Initialize the tooltip controller """
        self.tooltip = SubjectTooltip()
        self.tooltip_service = SubjectTooltipService()

    def register_callbacks(self, app):
        """ Register callbacks for the tooltip 

        Parameters:
            app (Dash app): The Dash application instance
        """
        # For hover functionality
        @app.callback(
            [Output('subject-tooltip', 'children'),
             Output('subject-tooltip', 'style')],
            [Input('session-table', 'active_cell'),
             Input('session-table', 'derived_viewport_data')],
            [State('session-table', 'data')]
        )
        def update_tooltip(active_cell, viewport_data, data):
            # Hide when not hover
            tooltip_style = {'display': 'none'}

            if not active_cell or active_cell['column_id'] != 'subject_id':
                return html.Div(), tooltip_style
            
            # Get subject ID from active cell
            row_index = active_cell['row']
            if viewport_data and len(viewport_data) > row_index:
                subject_id  = viewport_data[row_index]['subject_id']

                # Update the dataframe in the service
                if data:
                    self.tooltip_service.update_dataframe(pd.DataFrame(data))

                # Get alert data
                alert_data = self.tooltip_service.get_subject_alert_data(subject_id)

                # Build tooltip content
                tooltip_content = self.tooltip.build_tooltip_content(alert_data)

                # Show tooltip near the active cell
                tooltip_style = {
                    'display': 'block',
                    'position': 'fixed',
                    'backgroundColor': 'white',
                    'border': '1px solid #ddd',
                    'border-radius': '3px',
                    'padding': '10px',
                    'zIndex': '1000',
                    'max-width': '300px',
                    'box-shadow': '0 2px 4px rgba(0,0,0,0.2)'
                }

                return tooltip_content, tooltip_style
            
            return html.Div(), tooltip_style
        
        # Client-side callback for positioning
        clientside_callback(
            """
            function(active_cell, data) {
                if (!active_cell || active_cell.column_id !== 'subject_id') {
                    return window.dash_clientside.no_update;
                }
                
                // Add positioning based on mouse position 
                document.addEventListener('mousemove', function(e) {
                    var tooltip = document.getElementById('subject-tooltip');
                    if (tooltip) {
                        tooltip.style.left = (e.pageX + 15) + 'px';
                        tooltip.style.top = (e.pageY + 15) + 'px';
                    }
                });
                
                return window.dash_clientside.no_update;
            }
            """,
            Output('subject-tooltip', 'data-position', allow_duplicate=True),
            [Input('session-table', 'active_cell'),
             Input('session-table', 'derived_viewport_data')], 
            prevent_initial_call=True
        )

    def get_tooltip_container(self):
        """ Get tooltip container element for layout"""
        return html.Div(
            id='subject-tooltip',
            style = { 'display': 'none'},
            className = 'subject-tooltip'
        )
            
    