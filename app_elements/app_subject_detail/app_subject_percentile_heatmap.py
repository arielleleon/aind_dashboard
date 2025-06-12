"""
Subject percentile heatmap visualization component for AIND Dashboard

This module creates interactive heatmap visualizations showing percentile data
across sessions with strata transitions and outlier highlighting.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any
from app_utils.simple_logger import get_logger
from dash import dcc

logger = get_logger('subject_percentile_heatmap')

class AppSubjectPercentileHeatmap:
    def __init__(self):
        
        # Features configuration
        self.features_config = {
            'finished_trials': False,  # Higher is better
            'ignore_rate': True,      # Lower is better
            'total_trials': False,    # Higher is better
            'foraging_performance': False,  # Higher is better
            'abs(bias_naive)': True   # Lower is better
        }
    
    def build(self, subject_id=None, app_utils=None, highlighted_session=None, colorscale_mode="binned"):
        """
        Build percentile heatmap showing progression over time including overall percentile
        
        This method coordinates UI rendering by calling business logic functions for data processing
        and statistical calculations, keeping the component focused on visualization concerns.
        
        Parameters:
            subject_id (str): Subject ID to build heatmap for
            app_utils (AppUtils): App utilities instance for accessing cached data
            highlighted_session (int): Session number to highlight with light blue border
            colorscale_mode (str): Either "binned" or "continuous" for colorscale type
            
        Returns:
            dcc.Graph: Heatmap showing percentile progression over time
        """
        if highlighted_session:
            logger.info(f"Highlighting session: {highlighted_session}")
        
        if not subject_id or not app_utils:
            return self._create_empty_heatmap("No subject selected")
        
        # Get time series data from UI cache
        time_series_data = app_utils.get_time_series_data(subject_id, use_cache=True)
        
        if not time_series_data or not time_series_data.get('sessions'):
            return self._create_empty_heatmap("No session data available")
        
        # Extract session data
        sessions = time_series_data['sessions']
        
        # Process data using business logic functions
        from app_utils.app_analysis.statistical_utils import StatisticalUtils
        from app_utils.percentile_utils import calculate_heatmap_colorscale
        
        # Extract and validate heatmap matrix data
        heatmap_data, feature_names = StatisticalUtils.process_heatmap_matrix_data(
            time_series_data, self.features_config
        )
        
        # Create session labels - show ALL sessions since we have full width
        session_labels = [f"S{s}" for s in sessions]
        logger.info(f"Displaying all {len(sessions)} sessions in heatmap")
        
        if not heatmap_data or not feature_names:
            return self._create_empty_heatmap("No valid feature data")
        
        # Get colorscale using business logic
        colorscale = calculate_heatmap_colorscale(colorscale_mode)
        
        # Create the heatmap visualization (UI responsibility)
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=session_labels,
            y=feature_names,
            colorscale=colorscale,
            zmin=0,
            zmax=100,
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br>Session: %{x}<br>Percentile: %{z:.1f}%<extra></extra>',
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="Percentile",
                    side="right"
                ),
                thickness=15,
                len=0.7,
                x=1.02
            )
        ))
        
        # Add highlighting for selected session using business logic
        if highlighted_session is not None and highlighted_session in sessions:
            session_idx = StatisticalUtils.calculate_session_highlighting_coordinates(
                sessions, highlighted_session
            )
            
            if session_idx is not None:
                # Add a vertical line to highlight the session column
                fig.add_shape(
                    type="rect",
                    x0=session_idx - 0.4,  # Start slightly before the session
                    x1=session_idx + 0.4,  # End slightly after the session
                    y0=-0.5,  # Start below the first row
                    y1=len(feature_names) - 0.5,  # End above the last row
                    line=dict(
                        color="#4A90E2",  # Light blue color matching time series highlighting
                        width=3
                    ),
                    fillcolor="rgba(74, 144, 226, 0.1)",  # Very light blue fill
                    layer="above"
                )
        
        # Add strata boundaries
        self._add_strata_boundaries(fig, sessions, time_series_data.get('strata', []), len(feature_names))
        
        # Add outlier markers to heatmap
        self._add_outlier_markers(fig, sessions, time_series_data.get('is_outlier', []), len(feature_names))
        
        fig.update_layout(
            title=None,
            xaxis_title="Session",
            yaxis_title=None,
            margin=dict(l=10, r=60, t=10, b=30),
            height=300,  # Increased height for more rows and full width
            font=dict(size=9),  # Smaller font to accommodate more sessions
            plot_bgcolor='white',
            xaxis=dict(
                tickangle=-45,  # Angle the session labels for better readability
                tickfont=dict(size=8),  # Smaller font for session labels
                automargin=True
            ),
            yaxis=dict(
                tickfont=dict(size=10),
                automargin=True
            )
        )
        
        return dcc.Graph(
            id='percentile-heatmap',
            figure=fig,
            config={'displayModeBar': False},
            style={'height': '300px', 'width': '100%'}  # Increased height
        )
    
    def _create_custom_colorscale(self):
        """
        DEPRECATED: Use calculate_heatmap_colorscale() from percentile_utils instead
        
        This method is kept for backward compatibility but business logic has been
        moved to percentile_utils.calculate_heatmap_colorscale() for better separation.
        """
        from app_utils.percentile_utils import calculate_heatmap_colorscale
        return calculate_heatmap_colorscale("binned")
    
    def _create_continuous_colorscale(self):
        """
        DEPRECATED: Use calculate_heatmap_colorscale() from percentile_utils instead
        
        This method is kept for backward compatibility but business logic has been
        moved to percentile_utils.calculate_heatmap_colorscale() for better separation.
        """
        from app_utils.percentile_utils import calculate_heatmap_colorscale
        return calculate_heatmap_colorscale("continuous")
    
    def _create_empty_heatmap(self, message):
        """Create empty heatmap with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#666666")
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,  # Updated to match new height
            plot_bgcolor='white',
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False)
        )
        
        return dcc.Graph(
            id='percentile-heatmap',
            figure=fig,
            config={'displayModeBar': False},
            style={'height': '300px', 'width': '100%'}  # Updated to match new height
        )
    
    def _add_strata_boundaries(self, fig, sessions, strata_data, num_feature_rows):
        """
        Add vertical lines for strata transitions in the heatmap
        
        Parameters:
            fig: plotly.graph_objects.Figure - The heatmap figure
            sessions: list - List of session numbers
            strata_data: list - List of strata for each session
            num_feature_rows: int - Number of feature rows in the heatmap
        """
        if not strata_data or len(strata_data) != len(sessions):
            logger.info("No strata data available or length mismatch")
            return
        
        # Find transition points
        transitions = []
        current_strata = None
        
        for i, (session, strata) in enumerate(zip(sessions, strata_data)):
            if current_strata is None:
                # First session - record starting strata
                current_strata = strata
                transitions.append({
                    'session_idx': i,
                    'session': session,
                    'strata': strata,
                    'transition_type': 'start'
                })
            elif strata != current_strata:
                # Strata transition detected
                transitions.append({
                    'session_idx': i,
                    'session': session,
                    'strata': strata,
                    'transition_type': 'transition'
                })
                current_strata = strata
        
        logger.info(f"Found {len(transitions)} strata transitions: {[t['session'] for t in transitions]}")
        
        # Add vertical lines for transitions (skip the first one which is just the start)
        for transition in transitions[1:]:  # Skip first transition (start)
            session_idx = transition['session_idx']
            session = transition['session']
            strata = transition['strata']
            strata_abbr = self._get_strata_abbreviation(strata)
            
            # Add vertical line at the beginning of the new strata
            fig.add_shape(
                type="line",
                x0=session_idx - 0.5,  # Place line between sessions
                x1=session_idx - 0.5,
                y0=-0.5,  # Start below the first row
                y1=num_feature_rows - 0.5,  # End above the last row
                line=dict(
                    color='rgba(128, 128, 128, 0.8)',  # Gray color matching time series
                    width=2,
                    dash='dash'
                ),
                layer="above"
            )
            
            # Add text annotation for the new strata
            fig.add_annotation(
                x=session_idx - 0.5,
                y=num_feature_rows - 0.2,  # Position near the top
                text=f"→ {strata_abbr}",
                showarrow=False,
                font=dict(size=8, color='gray'),
                textangle=-90,  # Rotate text vertically
                xanchor="center",
                yanchor="bottom"
            )
            
            logger.info(f"Added strata boundary at session {session} (index {session_idx}) for strata: {strata_abbr}")
    
    def _get_strata_abbreviation(self, strata):
        """Get abbreviated strata name for display (same as time series)"""
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

    def _add_outlier_markers(self, fig, sessions, outlier_data, num_feature_rows):
        """
        Add outlier markers to the heatmap by highlighting outlier session columns
        
        Parameters:
            fig: plotly.graph_objects.Figure - The heatmap figure
            sessions: list - List of session numbers  
            outlier_data: list - List of outlier indicators for each session
            num_feature_rows: int - Number of feature rows in the heatmap
        """
        if not outlier_data or len(outlier_data) != len(sessions):
            logger.info("No outlier data available or length mismatch")
            return
        
        outlier_count = 0
        
        # Add vertical rectangles for outlier sessions
        for session_idx, (session, is_outlier) in enumerate(zip(sessions, outlier_data)):
            if is_outlier:
                # Add a rectangle with purple border around the entire column for this session
                fig.add_shape(
                    type="rect",
                    x0=session_idx - 0.45,  # Start slightly before the session
                    x1=session_idx + 0.45,  # End slightly after the session
                    y0=-0.45,  # Start slightly below the first row
                    y1=num_feature_rows - 0.55,  # End slightly above the last row
                    line=dict(
                        color="#9C27B0",  # Purple color matching dataframe outlier styling
                        width=2
                    ),
                    fillcolor="rgba(156, 39, 176, 0.1)",  # Very light purple fill
                    layer="above"
                )
                outlier_count += 1
                
        if outlier_count > 0:
            logger.info(f"Added outlier markers for {outlier_count} sessions with purple borders") 