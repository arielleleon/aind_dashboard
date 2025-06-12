"""
Subject raw timeseries visualization component for AIND Dashboard

This module creates interactive timeseries plots showing raw feature values
over time with rolling averages and outlier detection.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
from app_utils.simple_logger import get_logger
from dash import html, dcc
import dash_bootstrap_components as dbc

logger = get_logger('subject_timeseries')

class AppSubjectTimeseries:
    def __init__(self):
        """Initialize the new, clean timeseries component"""
        # Define features to plot with their optimization preferences
        self.features_config = {
            'finished_trials': False,  # Higher is better
            'ignore_rate': True,      # Lower is better  
            'total_trials': False,    # Higher is better
            'foraging_performance': False,  # Higher is better
            'abs(bias_naive)': True   # Lower is better
        }
        
        # Color scheme for features
        self.feature_colors = {
            'finished_trials': '#1f77b4',    # Blue
            'ignore_rate': '#ff7f0e',        # Orange  
            'total_trials': '#2ca02c',       # Green
            'foraging_performance': '#d62728', # Red
            'abs(bias_naive)': '#9467bd'     # Purple
        }
        
    def build(self):
        """Build the complete timeseries component"""
        return html.Div([
            # Feature selection controls at the top
            html.Div([
                html.Label("Raw Feature Values (3-Session Rolling Average):", className="control-label mb-1", style={"fontSize": "14px", "fontWeight": "600"}),
                dcc.Dropdown(
                    id="timeseries-feature-dropdown",
                    options=self._get_feature_options(),
                    value=['all'],  # Default to all features
                    multi=True,
                    className="timeseries-feature-dropdown"
                )
            ], className="timeseries-controls mb-2"),
            
            # Main timeseries plot - simplified container
                dcc.Graph(
                    id="timeseries-plot",
                    figure=self._create_empty_figure(),
                    config={
                        'displayModeBar': False,
                        'responsive': True
                    },
                className="timeseries-graph",
                style={'height': '550px'}  # Increased height since title removed
            ),
            
            # Data store for timeseries data
            dcc.Store(id="timeseries-store", data={})
            
        ], className="subject-timeseries-component")
    
    def _get_feature_options(self):
        """Get dropdown options for feature selection"""
        options = [{'label': 'All Features', 'value': 'all'}]
        
        for feature in self.features_config.keys():
            # Create readable labels
            label = feature.replace('_', ' ').replace('abs(', '|').replace(')', '|').title()
            options.append({'label': label, 'value': feature})
            
        return options
    
    def _create_empty_figure(self):
        """Create empty plot with proper styling"""
        fig = go.Figure()
        
        fig.update_layout(
            title=None,
            xaxis_title="Session Number", 
            yaxis_title="Rolling Average Performance (Smoothed)",
            template="plotly_white",
            margin=dict(l=40, r=20, t=40, b=40),  # Increased top margin for hover tooltips
            height=550,
            legend=dict(
                orientation="h",
                yanchor="bottom", 
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified"
        )
        
        # Add placeholder text
        fig.add_annotation(
            text="Select a subject to view performance data",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        
        return fig
    
    def create_plot(self, subject_data, selected_features, highlighted_session=None):
        """
        Create the timeseries plot using raw feature values with 3-session rolling averages
        
        Parameters:
            subject_data: dict - Optimized time series data from app_utils
            selected_features: list - Features to plot  
            highlighted_session: int - Session to highlight
        """
        logger.info(f"Creating timeseries plot for subject with data keys: {list(subject_data.keys()) if subject_data else 'None'}")
        
        if not subject_data or 'sessions' not in subject_data:
            return self._create_empty_figure()
            
        sessions = subject_data['sessions']
        strata_data = subject_data.get('strata', [])
        
        if not sessions:
            return self._create_empty_figure()
            
        fig = go.Figure()
        
        # Determine which features to plot
        features_to_plot = []
        if 'all' in selected_features:
            features_to_plot = list(self.features_config.keys())
        else:
            features_to_plot = [f for f in selected_features if f in self.features_config]
            
        if not features_to_plot:
            features_to_plot = list(self.features_config.keys())
        
        logger.info(f"Features to plot: {features_to_plot}")
        
        # Create strata abbreviation mapping for hover info
        strata_sessions_map = {}
        if strata_data and len(strata_data) == len(sessions):
            for session, strata in zip(sessions, strata_data):
                strata_sessions_map[session] = self._get_strata_abbreviation(strata)
        
        # Plot each feature using raw values with our own rolling average
        for i, feature in enumerate(features_to_plot):
            # Look for raw feature data
            raw_key = f"{feature}_raw"
            
            if raw_key not in subject_data:
                logger.info(f"No raw data found for {feature}, skipping")
                continue
                
            raw_data = subject_data[raw_key]
            logger.info(f"Using raw data for {feature}: {len(raw_data)} values")
            
            # Filter out invalid values (-1 or NaN represents missing data)
            valid_data = []
            for j, (session, value) in enumerate(zip(sessions, raw_data)):
                if value is not None and not pd.isna(value) and value != -1:
                    valid_data.append((session, value))
            
            if len(valid_data) < 2:
                logger.info(f"Insufficient valid data for {feature}: {len(valid_data)} points")
                continue
                
            valid_sessions, valid_raw_values = zip(*valid_data)
            valid_sessions = list(valid_sessions)
            valid_raw_values = list(valid_raw_values)
            
            logger.info(f"Valid raw data for {feature}: {len(valid_raw_values)} points, range: {min(valid_raw_values):.3f} to {max(valid_raw_values):.3f}")
            
            # Apply 3-session rolling average to raw values
            rolling_avg_values = self._apply_rolling_average(valid_raw_values, window=3)
            
            # Normalize values for better visualization (shows relative performance)
            normalized_values = self._normalize_for_display(rolling_avg_values, feature)

            if i == 0:  # First trace gets strata info
                strata_hover_info = [strata_sessions_map.get(session, 'Unknown') for session in valid_sessions]
                hover_template = (f"<b>Strata: %{{customdata[2]}}</b><br><br>" +  # Strata at top with spacing
                                f"<b>{feature.replace('_', ' ').title()}</b><br>" +
                                "Raw Value: %{customdata[0]:.3f}<br>" +
                                "3-Session Avg: %{customdata[1]:.3f}<br>" +
                                "Normalized: %{y:.2f}<extra></extra>")
                custom_data = list(zip(valid_raw_values, rolling_avg_values, strata_hover_info))
            else:  # Subsequent traces don't include strata
                hover_template = (f"<b>{feature.replace('_', ' ').title()}</b><br>" +
                                "Raw Value: %{customdata[0]:.3f}<br>" +
                                "3-Session Avg: %{customdata[1]:.3f}<br>" +
                                "Normalized: %{y:.2f}<extra></extra>")
                custom_data = list(zip(valid_raw_values, rolling_avg_values))
            
            # Create trace with integer session numbers and enhanced hover info
            fig.add_trace(go.Scatter(
                x=valid_sessions,  # Keep original integer sessions
                y=normalized_values,
                mode='lines',
                name=feature.replace('_', ' ').replace('abs(', '|').replace(')', '|').title(),
                line=dict(
                    color=self.feature_colors.get(feature, '#000000'),
                    width=3,
                    shape='spline',  # Use plotly's spline smoothing
                    smoothing=1.3    # Plotly smoothing parameter
                ),
                hovertemplate=hover_template,
                customdata=custom_data
            ))
        
        # Add strata transition lines
        if strata_data and len(strata_data) == len(sessions):
            self._add_strata_transitions(fig, sessions, strata_data)
        
        # Update layout
        fig.update_layout(
            title=None,
            xaxis_title="Session Number",
            yaxis_title="3-Session Rolling Average (Normalized & Smoothed)", 
            template="plotly_white",
            margin=dict(l=40, r=20, t=40, b=40),  # Increased top margin for hover tooltips
            height=550,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02, 
                xanchor="right",
                x=1
            ),
            hovermode="x unified",
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(211,211,211,0.3)'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1, 
                gridcolor='rgba(211,211,211,0.3)',
                zeroline=True,
                zerolinecolor='gray',
                zerolinewidth=1
            )
        )
        
        # Add session highlight if specified
        if highlighted_session and sessions:
            if highlighted_session in sessions:
                fig.add_vline(
                    x=highlighted_session,
                    line=dict(
                        color='rgba(65, 105, 225, 0.6)',
                        width=3,
                        dash='solid'
                    ),
                    annotation_text=f"Session {highlighted_session}",
                    annotation_position="top"
                )
        
        # PHASE 2: Add outlier markers to time series plot
        self._add_outlier_markers(fig, sessions, subject_data.get('is_outlier', []))
        
        return fig
    
    def _apply_rolling_average(self, values, window=3):
        """
        Apply rolling average to raw values
        
        Parameters:
            values: list - Raw values
            window: int - Rolling window size (default 3 sessions)
            
        Returns:
            list - Rolling averaged values
        """
        if len(values) < window:
            return values  # Return original if not enough data
        
        rolling_values = []
        
        # For first few values, use expanding window
        for i in range(len(values)):
            if i < window - 1:
                # Use expanding window for early values
                window_values = values[:i+1]
            else:
                # Use fixed window for later values
                window_values = values[i-window+1:i+1]
            
            avg_value = sum(window_values) / len(window_values)
            rolling_values.append(avg_value)
        
        return rolling_values
    
    def _normalize_for_display(self, values, feature):
        """
        Normalize values for better display visualization
        
        Parameters:
            values: list - Values to normalize
            feature: str - Feature name for inversion logic
            
        Returns:
            list - Normalized values
        """
        if not values:
            return []
            
        # Convert to numpy array
        values_array = np.array(values)
        
        # For features where lower is better, invert them so higher is always better on display
        if self.features_config.get(feature, False):  # True means lower is better
            # Invert by subtracting from max and adding min to maintain relative scale
            max_val = np.max(values_array)
            min_val = np.min(values_array)
            values_array = max_val + min_val - values_array
            
        # Apply z-score normalization for consistent scaling across features
        if len(values_array) > 1:
            mean_val = np.mean(values_array)
            std_val = np.std(values_array)
            if std_val > 0:
                normalized = (values_array - mean_val) / std_val
            else:
                normalized = values_array - mean_val
        else:
            normalized = values_array
            
        return normalized.tolist()
    
    def _get_strata_abbreviation(self, strata):
        """Get abbreviated strata name for display"""
        if not strata:
            return 'Unknown'
        
        # Hard coded mappings for common terms
        strata_mappings = {
            'Uncoupled Baiting': 'UB',
            'Coupled Baiting': 'CB', 
            'Uncoupled Without Baiting': 'UWB',
            'Coupled Without Baiting': 'CWB',
            'BEGINNER': 'B',
            'INTERMEDIATE': 'I',
            'ADVANCED': 'A',
            'v1': '1',
            'v2': '2',
            'v3': '3'
        }
        
        # Split the strata name
        parts = strata.split('_')
        
        # Handle different strata formats
        if len(parts) >= 3:
            # Format: curriculum_Stage_Version
            curriculum = '_'.join(parts[:-2])
            stage = parts[-2]
            version = parts[-1]
            
            # Get abbreviations
            curriculum_abbr = strata_mappings.get(curriculum, curriculum[:2].upper())
            stage_abbr = strata_mappings.get(stage, stage[0])
            version_abbr = strata_mappings.get(version, version[-1])
            
            return f"{curriculum_abbr}{stage_abbr}{version_abbr}"
        
        return strata.replace(" ", "")
    
    def _add_strata_transitions(self, fig, sessions, strata_data):
        """Add vertical lines for strata transitions"""
        if not strata_data or len(strata_data) != len(sessions):
            return
        
        # Find transition points
        transitions = []
        current_strata = None
        
        for i, (session, strata) in enumerate(zip(sessions, strata_data)):
            if current_strata is None:
                # First session - record starting strata
                current_strata = strata
                transitions.append({
                    'session': session,
                    'strata': strata,
                    'transition_type': 'start'
                })
            elif strata != current_strata:
                # Strata transition detected
                transitions.append({
                    'session': session,
                    'strata': strata,
                    'transition_type': 'transition'
                })
                current_strata = strata
        
        # Add vertical lines for transitions (skip the first one which is just the start)
        for transition in transitions[1:]:  # Skip first transition (start)
            session = transition['session']
            strata = transition['strata']
            strata_abbr = self._get_strata_abbreviation(strata)
            
            fig.add_vline(
                x=session,
                line=dict(
                    color='rgba(128, 128, 128, 0.6)',
                    width=2,
                    dash='dash'
                ),
                annotation=dict(
                    text=f"→ {strata_abbr}",
                    textangle=-90,
                    font=dict(size=10, color='gray'),
                    showarrow=False,
                    yshift=10
                )
            )

    def _add_outlier_markers(self, fig, sessions, outlier_data):
        """
        Add purple markers for outlier sessions on the time series plot
        
        Parameters:
            fig: plotly.graph_objects.Figure - The time series figure
            sessions: list - List of session numbers
            outlier_data: list - List of boolean outlier indicators for each session
        """
        if not sessions or not outlier_data or len(outlier_data) != len(sessions):
            logger.info("No outlier data available or length mismatch for time series markers")
            return
        
        outlier_sessions = []
        
        # Find outlier sessions
        for session, is_outlier in zip(sessions, outlier_data):
            if is_outlier:
                outlier_sessions.append(session)
        
        if not outlier_sessions:
            return
        
        # Get current y-axis range to position markers at the top
        y_range = fig.layout.yaxis.range if fig.layout.yaxis.range else [-3, 3]
        marker_y_position = y_range[1] * 0.9  # Position near top of plot
        
        # Add purple markers for outlier sessions
        fig.add_trace(go.Scatter(
            x=outlier_sessions,
            y=[marker_y_position] * len(outlier_sessions),
            mode='markers',
            marker=dict(
                color='#9C27B0',  # Purple color matching dataframe and heatmap
                size=12,
                symbol='diamond',
                line=dict(
                    width=2,
                    color='#FFFFFF'
                )
            ),
            name='Outlier Sessions',
            hovertemplate='<b>Outlier Session</b><br>Session: %{x}<extra></extra>',
            showlegend=True
        ))
        
        logger.info(f"Added outlier markers for {len(outlier_sessions)} sessions: {outlier_sessions}")