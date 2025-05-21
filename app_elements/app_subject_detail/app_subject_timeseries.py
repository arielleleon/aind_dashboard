from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash import callback_context

class AppSubjectTimeseries:
    def __init__(self):
        """ Initialize timeseries component """
        # Define the features from AppDataFrame that we want to plot
        self.features_config = {
            'finished_trials': False,  # Higher is better
            'ignore_rate': True,     # Lower is better
            'total_trials': False,   # Higher is better
            'foraging_performance': False,   # Higher is better
            'abs(bias_naive)': True  # Lower is better 
        }
        
        # Define colors for each feature
        self.feature_colors = {
            'finished_trials': '#1f77b4',
            'ignore_rate': '#ff7f0e',
            'total_trials': '#2ca02c',
            'foraging_performance': '#d62728',
            'abs(bias_naive)': '#9467bd'
        }
        
        # Set window size for moving average
        self.window_size = 3  # Default window size for smoothing
    
    def moving_average(self, data, window_size=None):
        """
        Apply moving average smoothing to data
        
        Parameters:
        -----------
        data : list
            Input data to smooth
        window_size : int, optional
            Size of the moving average window. If None, uses self.window_size
            
        Returns:
        --------
        list
            Smoothed data
        """
        if window_size is None:
            window_size = self.window_size
            
        # Convert to numpy array for processing
        data = np.array(data)
        
        # If data is too short or window_size is 1, return original data
        if len(data) < window_size or window_size < 2:
            return data.tolist()
        
        # Create the smoothed data using convolution
        weights = np.ones(window_size) / window_size
        smoothed = np.convolve(data, weights, mode='valid')
        
        # Pad the beginning to maintain the same length
        pad_length = len(data) - len(smoothed)
        padding = [None] * pad_length
        
        return padding + smoothed.tolist()
    
    def get_feature_options(self):
        """ Get the options for the feature selection dropdown """
        options = [{'label': 'All Features', 'value': 'all'}]
        
        # Add each feature as an option
        for feature, is_lower_better in self.features_config.items():
            # Create a more readable label
            feature_display = feature.replace('_', ' ').replace('abs(', '|').replace(')', '|').title()
            options.append({'label': feature_display, 'value': feature})
        
        return options
    
    def build(self):
        """ Build timeseries component with feature selection dropdown """
        return html.Div([
            # Feature selection dropdown
            html.Div([
                html.Label("Select Features:", className="feature-select-label"),
                dcc.Dropdown(
                    id="feature-select-dropdown",
                    options=self.get_feature_options(),
                    value=['all'],  # Default to all features
                    multi=True,
                    className="feature-select-dropdown"
                )
            ], className="feature-select-container mb-3"),
            
            # Main graph component
            dcc.Graph(
                id="subject-timeseries-graph",
                figure=self.create_empty_figure(),
                config={
                    'displayModeBar': False,  # Hide the mode bar completely
                    'responsive': True
                },
                className="subject-timeseries-graph"
            ),
            
            # Store component to track selected session and available data
            dcc.Store(id="timeseries-data-store", data={
                "subject_id": None,
                "selected_session": None,
                "all_sessions": []
            })
        ], id="subject-timeseries-container", className="subject-timeseries-container")
    
    def create_empty_figure(self):
        """ Create an empty figure with basic layout """
        fig = go.Figure()
        
        # Update layout for better appearance
        fig.update_layout(
            title=None,
            xaxis_title="Session Number",
            yaxis_title="Performance Metrics",
            template="plotly_white",
            margin=dict(l=20, r=10, t=10, b=30),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="closest",
            # Initialize with a basic range to ensure shapes are visible
            yaxis=dict(range=[-2, 2])
        )
        
        # Add text when no data is available
        fig.add_annotation(
            text="Select a subject to view timeseries data",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        
        return fig