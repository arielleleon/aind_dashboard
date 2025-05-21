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
        
        # Color mapping for different alert categories
        color_map = {
            'B': '#FF8C40',   # Orange - Bad
            'SB': '#FF8C40',  # Orange - Severely Bad
            'G': '#4D94DA',   # Blue - Good
            'SG': '#4D94DA',  # Blue - Severely Good
            'N': '#CCCCCC',   # Grey - Normal
            'NS': '#CCCCCC'   # Grey - Not Scored
        }

        # Iterate through features and get percentiles
        for feature, is_lower_better in features_config.items():
            percentile_key = f"{feature}_percentile"
            category_key = f"{feature}_category"  # Correct key for alert category
            
            if percentile_key in subject_data and not pd.isna(subject_data[percentile_key]):
                features.append(feature.replace('_', '').replace('(', '').replace(')', ''))
                percentile = subject_data[percentile_key]
                percentiles.append(percentile)

                # Determine color based on alert category 
                if category_key in subject_data:
                    category = subject_data[category_key]
                    colors.append(color_map.get(category, '#CCCCCC'))  # Default to grey if unknown
                else:
                    # Fallback in case category isn't available
                    colors.append('#CCCCCC')

        # Calculate max percentile for y-axis
        max_percentile = max(percentiles) if percentiles else 100
        # Round up to nearest 10
        max_y = min(100, ((max_percentile // 10) + 1) * 10)

        # Create figure with vertical bars showing actual percentiles
        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=percentiles,
            x=features,
            orientation='v',
            marker_color=colors,
            marker_opacity=0.7,  # Reduced opacity as requested
            text=[f"{p:.1f}%" for p in percentiles],
            textposition='auto',
            hoverinfo='text',
            hovertext=[f"{f}: {p:.1f}%" for f, p in zip(features, percentiles)],
            width=0.6
        ))

        fig.update_layout(
            title=None,
            margin=dict(l=0, r=0, t=10, b=10),
            yaxis = dict(
                title='Percentile',
                range=[0, max_y],
                tickvals=[0, 25, 50, 75, 100][:int(max_y/25)+1],
                gridcolor='#EEEEEE',
                showgrid=True
            ),
            xaxis = dict(
                title=None,
                automargin=True,
                tickangle=-45
            ),
            plot_bgcolor='white',
            height=250,
            width=None,
            bargap=0.15
        )

        return dcc.Graph(
            id='feature-percentile-chart',
            figure=fig,
            config={'displayModeBar': False},
            style={'height': '100%', 'width': '100%'}
        )

        
        
        