import plotly.graph_objects as go
from dash import dcc
import pandas as pd 

class AppFeatureChart:
    def __init__(self):
        pass
    
    def build(self, subject_data=None):
        """
        Build subject-feature specific percentile chart

        Parameters:
            subject_data (dict): Subject data dictionary

        Returns:
            dcc.Graph: Percentile chart
        """
        if not subject_data:
            return dcc.Graph(
                id='feature-percentile-chart',
                figure=go.Figure(),
                config={'displayModeBar': False}
            )
        
        # Extract feature percentiles from subject data
        features = []
        percentiles = []
        colors = []

        # Get feature config from app dataframe
        features_config = {
            'finished_trials': False,
            'ignore_rate': True,
            'total_trials': False,
            'abs(bias_naive)': True,
            'foraging_performance': False
        }

        # Iterate through features and get percentiles
        for feature, is_lower_better in features_config.items():
            percentile_key = f"{feature}_percentile"
            if percentile_key in subject_data and not pd.isna(subject_data[percentile_key]):
                features.append(feature.replace('_', '').replace('(', '').replace(')', ''))

                percentile = subject_data[percentile_key]
                percentiles.append(percentile)

                # Determine color based on percentile -- CHANGE BASED ON ALERT CATEGORY
                if percentile > 50:
                    colors.append('#4D94DA')
                else:
                    colors.append('#FF8C40')

        # Calculate difference from 50th percentile 
        differences = [p - 50 for p in percentiles]

        # Create figure
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=differences,
            y=features,
            orientation='h',
            marker_color=colors,
            text=[f"{p:.1f}%" for p in percentiles],
            textposition='auto',
            hoverinfo='text',
            hovertext=[f"{f}: {p:.1f}%" for f, p in zip(features, percentiles)]
        ))

        fig.update_layout(
            title=None,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis = dict(
                title='Difference from 50th Percentile',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='#CCCCCC',
                range=[-50, 50],
                tickvals=[-50, -25, 0, 25, 50],
                gridcolor='#EEEEEE',
                showgrid=True
            ),
            yaxis = dict(
                title=None,
                automargin=True
            ),
            plot_bgcolor='white',
            height=200,
            width=None
        )

        return dcc.Graph(
            id='feature-percentile-chart',
            figure=fig,
            config={'displayModeBar': False},
            style={'height': '100%', 'width': '100%'}
        )

        
        
        