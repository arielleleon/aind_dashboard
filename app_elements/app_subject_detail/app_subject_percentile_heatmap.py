import plotly.graph_objects as go
from dash import dcc
import pandas as pd
import numpy as np

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
        
        Parameters:
            subject_id (str): Subject ID to build heatmap for
            app_utils (AppUtils): App utilities instance for accessing cached data
            highlighted_session (int): Session number to highlight with light blue border
            colorscale_mode (str): Either "binned" or "continuous" for colorscale type
            
        Returns:
            dcc.Graph: Heatmap showing percentile progression over time
        """
        if highlighted_session:
            print(f"Highlighting session: {highlighted_session}")
        
        if not subject_id or not app_utils:
            return self._create_empty_heatmap("No subject selected")
        
        # Get time series data from UI cache
        time_series_data = app_utils.get_time_series_data(subject_id, use_cache=True)
        
        if not time_series_data or not time_series_data.get('sessions'):
            return self._create_empty_heatmap("No session data available")
        
        # Extract session data
        sessions = time_series_data['sessions']
        
        # Build heatmap data matrix
        heatmap_data = []
        feature_names = []
        session_labels = []
        
        # Process each feature
        for feature in self.features_config.keys():
            percentile_key = f"{feature}_percentiles"
            
            if percentile_key in time_series_data:
                percentiles = time_series_data[percentile_key]
                
                # Filter out invalid values (-1 represents NaN)
                valid_percentiles = [p if p != -1 else np.nan for p in percentiles]
                
                if any(not np.isnan(p) for p in valid_percentiles):
                    heatmap_data.append(valid_percentiles)
                    # Clean feature name for display
                    display_name = feature.replace('_', ' ').replace('abs(', '|').replace(')', '|').title()
                    feature_names.append(display_name)
        
        # Add overall percentile row
        if 'overall_percentiles' in time_series_data:
            overall_percentiles = time_series_data['overall_percentiles']
            # Filter out invalid values (-1 represents NaN)
            valid_overall = [p if p != -1 else np.nan for p in overall_percentiles]
            
            if any(not np.isnan(p) for p in valid_overall):
                heatmap_data.append(valid_overall)
                feature_names.append("Overall Percentile")
        
        # Create session labels - show ALL sessions since we have full width
        session_labels = [f"S{s}" for s in sessions]
        print(f"📊 Displaying all {len(sessions)} sessions in heatmap")
        
        if not heatmap_data or not feature_names:
            return self._create_empty_heatmap("No valid feature data")
        
        # Choose colorscale based on mode
        if colorscale_mode == "continuous":
            colorscale = self._create_continuous_colorscale()
        else:
            colorscale = self._create_custom_colorscale()
        
        # Create the heatmap
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
        
        # Add highlighting for selected session
        if highlighted_session is not None and highlighted_session in sessions:
            # Find the index of the highlighted session in the sessions list
            try:
                session_idx = sessions.index(highlighted_session)
                
                # Add a vertical line to highlight the session column
                # We'll add a rectangle with transparent fill and light blue border
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
                                
            except ValueError:
                print(f" Session {highlighted_session} not found in session list")
        
        # Add strata boundaries
        self._add_strata_boundaries(fig, sessions, time_series_data.get('strata', []), len(feature_names))
        
        # PHASE 2: Add outlier markers to heatmap
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
        """Create custom colorscale matching alert categories"""
        # Create a colorscale that maps percentile ranges to alert colors
        return [
            [0.0, '#FF6B35'],    # 0-6.5% (SB) - Dark orange
            [0.065, '#FF6B35'],  # 
            [0.065, '#FFB366'],  # 6.5-28% (B) - Light orange
            [0.28, '#FFB366'],   # 
            [0.28, '#E8E8E8'],   # 28-72% (N) - Light grey
            [0.72, '#E8E8E8'],   # 
            [0.72, '#4A90E2'],   # 72-93.5% (G) - Light blue
            [0.935, '#4A90E2'],  # 
            [0.935, '#2E5A87'],  # 93.5-100% (SG) - Dark blue
            [1.0, '#2E5A87']
        ]
    
    def _create_continuous_colorscale(self):
        """Create smooth continuous colorscale with gradual transitions"""
        # Create a smooth gradient from red (low) through grey (normal) to blue (high)
        return [
            [0.0, '#FF4444'],     # 0% - Bright red (worst performance)
            [0.065, '#FF6B35'],   # 6.5% - Orange-red transition
            [0.15, '#FFA366'],    # 15% - Light orange
            [0.28, '#FFD699'],    # 28% - Very light orange
            [0.40, '#F0F0F0'],    # 40% - Light grey (approaching normal)
            [0.50, '#E8E8E8'],    # 50% - Normal grey (median)
            [0.60, '#E0E8F0'],    # 60% - Very light blue
            [0.72, '#B8D4F0'],    # 72% - Light blue
            [0.85, '#7BB8E8'],    # 85% - Medium blue
            [0.935, '#4A90E2'],   # 93.5% - Good blue
            [1.0, '#1E5A96']      # 100% - Deep blue (best performance)
        ]
    
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
            print("No strata data available or length mismatch")
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
        
        print(f"Found {len(transitions)} strata transitions: {[t['session'] for t in transitions]}")
        
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
            
            print(f"Added strata boundary at session {session} (index {session_idx}) for strata: {strata_abbr}")
    
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
            print("No outlier data available or length mismatch")
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
            print(f"Added outlier markers for {outlier_count} sessions with purple borders") 