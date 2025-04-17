from dash import html
import dash_bootstrap_components as dbc

class SubjectTooltip:
    """ Component for displaying subject tooltips in Datatable"""

    def __init__(self):
        """ Initialize the tooltip component """
        pass 

    def build_tooltip_content(self, alert_data):
        """
        Build the content to be displayed in the tooltip

        Parameters:
            alert_data (dict): Dictionary containing alert data for a subject

        Returns:
            html.Div: Tooltip content as a Dash HTML component
        """
        alert_descriptions = {
            'SB': 'Significantly Below Average',
            'B': 'Below Average',
            'M': 'Average',
            'G': 'Above Average',
            'SG': 'Significantly Above Average',
            'T': 'Threshold Alert'
        }

        # Extract alert information
        percentile_alert = alert_data.get('percentile_category', 'NS')
        threshold_alert = alert_data.get('threshold_alert', 'N')
        combined_alert = alert_data.get('combined_alert', 'NS')

        # Build tooltip content
        tooltip_content = html.Div([
            html.H6('Alert Information', className='tooltip-header'),
            html.Hr(className = 'tooltip-divider'),

            # Show overall combined alert status
            html.Div([
                html.Span('Percentile Alerts: ', className='tooltip-label'),
                html.Span(
                    f"{percentile_alert} - {alert_descriptions.get(percentile_alert, '')}",
                    className = f'tooltip-value alert-{percentile_alert}'
                )
            ], className = 'tooltip-row'),

            # Show threshold alert status
            html.Div([
                html.Span('Threshold Alerts: ', className='tooltip-label'),
                html.Span(
                    "Exceeds Threshold" if threshold_alert == 'T' else "Within Threshold",
                    className = f'tooltip-value alert-{threshold_alert}'
                )
            ], className = 'tooltip-row'),
        ], className = 'subject-tooltip-content')

        return tooltip_content
        