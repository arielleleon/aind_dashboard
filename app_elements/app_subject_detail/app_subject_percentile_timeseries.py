from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

class AppSubjectPercentileTimeseries:
    def __init__(self):
        """Initialize the percentile timeseries component"""
        # Define features to plot with their optimization preferences
        self.features_config = {
            'finished_trials': False,  # Higher is better
            'ignore_rate': True,      # Lower is better  
            'total_trials': False,    # Higher is better
            'foraging_performance': False,  # Higher is better
            'abs(bias_naive)': True   # Lower is better
        }
        
        # Color scheme for features (same as raw values plot for consistency)
        self.feature_colors = {
            'finished_trials': '#1f77b4',    # Blue
            'ignore_rate': '#ff7f0e',        # Orange  
            'total_trials': '#2ca02c',       # Green
            'foraging_performance': '#d62728', # Red
            'abs(bias_naive)': '#9467bd'     # Purple
        }
        
    def build(self):
        """Build the complete percentile timeseries component"""
        return html.Div([
            # Feature selection controls
            html.Div([
                html.Label("Select Features (Percentiles):", className="control-label"),
                dcc.Dropdown(
                    id="percentile-timeseries-feature-dropdown",
                    options=self._get_feature_options(),
                    value=['all'],  # Default to all features
                    multi=True,
                    className="percentile-timeseries-feature-dropdown"
                )
            ], className="percentile-timeseries-controls mb-3"),
            
            # Main percentile timeseries plot
            html.Div([
                dcc.Graph(
                    id="percentile-timeseries-plot",
                    figure=self._create_empty_figure(),
                    config={
                        'displayModeBar': False,
                        'responsive': True
                    },
                    className="percentile-timeseries-graph"
                )
            ], className="percentile-timeseries-plot-container"),
            
        ], className="subject-percentile-timeseries-component")
    
    def _get_feature_options(self):
        """Get dropdown options for feature selection"""
        options = [{'label': 'All Features', 'value': 'all'}]
        
        # Add individual features
        for feature in self.features_config.keys():
            # Create readable labels
            label = feature.replace('_', ' ').replace('abs(', '|').replace(')', '|').title()
            options.append({'label': label, 'value': feature})
        
        # Add overall percentile as a selectable option
        options.append({'label': 'Overall Percentile', 'value': 'overall_percentile'})
        
        return options
    
    def _create_empty_figure(self):
        """Create empty plot with proper styling"""
        fig = go.Figure()
        
        fig.update_layout(
            title=None,
            xaxis_title="Session Number", 
            yaxis_title="Feature Percentiles",
            template="plotly_white",
            margin=dict(l=40, r=20, t=20, b=40),
            height=300,  # Smaller height since it's a secondary plot
            legend=dict(
                orientation="h",
                yanchor="bottom", 
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified",
            yaxis=dict(
                range=[0, 100],  # Percentiles range from 0-100
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(211,211,211,0.3)',
                zeroline=False
            )
        )
        
        # Add placeholder text
        fig.add_annotation(
            text="Select a subject to view percentile data",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        
        return fig
    
    def create_plot(self, subject_data, selected_features, highlighted_session=None):
        """
        Create the percentile timeseries plot using session-level percentiles
        
        Parameters:
            subject_data: dict - Time series data from app_utils
            selected_features: list - Features to plot  
            highlighted_session: int - Session to highlight
        """
        print(f"Creating percentile timeseries plot for subject with data keys: {list(subject_data.keys()) if subject_data else 'None'}")
        
        if not subject_data or 'sessions' not in subject_data:
            return self._create_empty_figure()
            
        sessions = subject_data['sessions']
        dates = subject_data.get('dates', [])
        strata_data = subject_data.get('strata', [])
        
        if not sessions:
            return self._create_empty_figure()
            
        fig = go.Figure()
        
        # Determine which features to plot
        features_to_plot = []
        show_overall_percentile = False
        
        if 'all' in selected_features:
            features_to_plot = list(self.features_config.keys())
            show_overall_percentile = True  # Include overall percentile when "all" is selected
        else:
            # Check for individual features
            for feature in selected_features:
                if feature in self.features_config:
                    features_to_plot.append(feature)
                elif feature == 'overall_percentile':
                    show_overall_percentile = True
            
            # If no valid features selected, default to all
            if not features_to_plot and not show_overall_percentile:
                features_to_plot = list(self.features_config.keys())
                show_overall_percentile = True
        
        print(f"Features to plot (percentiles): {features_to_plot}")
        print(f"Show overall percentile: {show_overall_percentile}")
        
        # Create strata abbreviation mapping for hover info
        strata_sessions_map = {}
        if strata_data and len(strata_data) == len(sessions):
            for session, strata in zip(sessions, strata_data):
                strata_sessions_map[session] = self._get_strata_abbreviation(strata)
        
        # Plot each feature using percentile data
        for i, feature in enumerate(features_to_plot):
            # Look for percentile data
            percentile_key = f"{feature}_percentiles"
            
            if percentile_key not in subject_data:
                print(f"No percentile data found for {feature}, skipping")
                continue
                
            percentile_data = subject_data[percentile_key]
            print(f"Using percentile data for {feature}: {len(percentile_data)} values")
            
            # Filter out invalid percentile values (-1 represents missing data)
            valid_data = []
            for j, (session, percentile) in enumerate(zip(sessions, percentile_data)):
                if percentile is not None and not pd.isna(percentile) and percentile != -1:
                    valid_data.append((session, percentile))
            
            if len(valid_data) < 2:
                print(f"Insufficient valid percentile data for {feature}: {len(valid_data)} points")
                continue
                
            valid_sessions, valid_percentiles = zip(*valid_data)
            valid_sessions = list(valid_sessions)
            valid_percentiles = list(valid_percentiles)
            
            print(f"Valid percentile data for {feature}: {len(valid_percentiles)} points, range: {min(valid_percentiles):.1f}% to {max(valid_percentiles):.1f}%")
            
            # Create strata info for hover (only for first trace to avoid repetition)
            if i == 0:  # First trace gets strata info
                strata_hover_info = [strata_sessions_map.get(session, 'Unknown') for session in valid_sessions]
                hover_template = (f"<b>Strata: %{{customdata[1]}}</b><br><br>" +  # Strata at top with spacing
                                f"<b>{feature.replace('_', ' ').title()}</b><br>" +
                                "Percentile: %{y:.1f}%<br>" +
                                "Session: %{x}<extra></extra>")
                custom_data = list(zip(valid_percentiles, strata_hover_info))
            else:  # Subsequent traces don't include strata
                hover_template = (f"<b>{feature.replace('_', ' ').title()}</b><br>" +
                                "Percentile: %{y:.1f}%<br>" +
                                "Session: %{x}<extra></extra>")
                custom_data = list(zip(valid_percentiles))
            
            # Create trace for feature percentiles
            fig.add_trace(go.Scatter(
                x=valid_sessions,
                y=valid_percentiles,
                mode='lines',
                name=feature.replace('_', ' ').replace('abs(', '|').replace(')', '|').title(),
                line=dict(
                    color=self.feature_colors.get(feature, '#000000'),
                    width=2,
                    shape='spline',
                    smoothing=1.0
                ),
                hovertemplate=hover_template,
                customdata=custom_data
            ))
        
        # Add overall percentile trace (distinctive styling) - only if selected
        if show_overall_percentile and 'overall_percentiles' in subject_data:
            overall_percentiles = subject_data['overall_percentiles']
            print(f"Adding overall percentile trace: {len(overall_percentiles)} values")
            
            # Filter out invalid values
            valid_overall_data = []
            for session, percentile in zip(sessions, overall_percentiles):
                if percentile is not None and not pd.isna(percentile) and percentile != -1:
                    valid_overall_data.append((session, percentile))
            
            if valid_overall_data:
                valid_sessions_overall, valid_percentiles_overall = zip(*valid_overall_data)
                valid_sessions_overall = list(valid_sessions_overall)
                valid_percentiles_overall = list(valid_percentiles_overall)
                
                print(f"Valid overall percentile data: {len(valid_percentiles_overall)} points, range: {min(valid_percentiles_overall):.1f}% to {max(valid_percentiles_overall):.1f}%")
                
                # Create strata info for overall percentile hover
                strata_hover_info_overall = [strata_sessions_map.get(session, 'Unknown') for session in valid_sessions_overall]
                
                # Determine if this is the only trace being plotted
                is_only_trace = len(features_to_plot) == 0
                
                # Create hover template with strata info only if it's the first/only trace
                if is_only_trace or not features_to_plot:  # If overall percentile is the only thing selected
                    hover_template = (f"<b>Strata: %{{customdata[1]}}</b><br><br>" +  # Strata at top
                                    "<b>Overall Percentile</b><br>" +
                                    "Percentile: %{y:.1f}%<br>" +
                                    "Session: %{x}<extra></extra>")
                    custom_data = list(zip(valid_percentiles_overall, strata_hover_info_overall))
                else:  # If there are other features, don't repeat strata info
                    hover_template = ("<b>Overall Percentile</b><br>" +
                                    "Percentile: %{y:.1f}%<br>" +
                                    "Session: %{x}<extra></extra>")
                    custom_data = list(zip(valid_percentiles_overall))
                
                # Add overall percentile trace with distinctive styling
                fig.add_trace(go.Scatter(
                    x=valid_sessions_overall,
                    y=valid_percentiles_overall,
                    mode='lines',
                    name='Overall Percentile',
                    line=dict(
                        color='#2E8B57',  # Sea green color for distinction
                        width=4,          # Thicker line
                        dash='dash',      # Dashed line style for distinction
                        shape='spline',
                        smoothing=1.0
                    ),
                    hovertemplate=hover_template,
                    customdata=custom_data
                ))
        
        # Add strata transition lines
        if strata_data and len(strata_data) == len(sessions):
            self._add_strata_transitions(fig, sessions, strata_data)
        
        # Add reference lines for percentile categories
        # Add background regions for different performance categories
        fig.add_hline(y=6.5, line_dash="dash", line_color="red", line_width=1, opacity=0.7,
                     annotation_text="Severely Below (6.5%)", annotation_position="right")
        fig.add_hline(y=28, line_dash="dash", line_color="orange", line_width=1, opacity=0.7,
                     annotation_text="Below (28%)", annotation_position="right")
        fig.add_hline(y=72, line_dash="dash", line_color="orange", line_width=1, opacity=0.7,
                     annotation_text="Good (72%)", annotation_position="right")
        fig.add_hline(y=93.5, line_dash="dash", line_color="green", line_width=1, opacity=0.7,
                     annotation_text="Severely Good (93.5%)", annotation_position="right")
        
        # Update layout
        fig.update_layout(
            title=None,
            xaxis_title="Session Number",
            yaxis_title="Feature Percentiles (%)", 
            template="plotly_white",
            margin=dict(l=40, r=20, t=20, b=40),
            height=300,
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
                range=[0, 100],  # Fixed range for percentiles
                showgrid=True,
                gridwidth=1, 
                gridcolor='rgba(211,211,211,0.3)',
                zeroline=False
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
        
        return fig
    
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