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
            # Feature selection controls at the top
            html.Div([
                html.Label("Feature Percentiles:", className="control-label mb-1", style={"fontSize": "14px", "fontWeight": "600"}),
                dcc.Dropdown(
                    id="percentile-timeseries-feature-dropdown",
                    options=self._get_feature_options(),
                    value=['all'],  # Default to all features
                    multi=True,
                    className="percentile-timeseries-feature-dropdown"
                ),
                # Add confidence interval toggle
                html.Div([
                    dcc.Checklist(
                        id="percentile-ci-toggle",
                        options=[
                            {'label': ' Show 95% Confidence Intervals', 'value': 'show_ci'}
                        ],
                        value=['show_ci'],  # Default to showing CI
                        className="ci-toggle-checklist",
                        style={"marginTop": "5px", "fontSize": "12px"}
                    )
                ], className="ci-controls")
            ], className="percentile-timeseries-controls mb-2"),
            
            # Main percentile timeseries plot - simplified container
                dcc.Graph(
                    id="percentile-timeseries-plot",
                    figure=self._create_empty_figure(),
                    config={
                        'displayModeBar': False,
                        'responsive': True
                    },
                className="percentile-timeseries-graph",
                style={'height': '550px'}  # Increased height since title removed
            ),
            
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
            margin=dict(l=40, r=20, t=40, b=40),  # Increased top margin for hover tooltips
            height=550,  # Increased to match raw timeseries plot
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
    
    def create_plot(self, subject_data, selected_features, highlighted_session=None, show_confidence_intervals=True):
        """
        Create the percentile timeseries plot using session-level percentiles with confidence intervals
        
        Parameters:
            subject_data: dict - Time series data from app_utils
            selected_features: list - Features to plot  
            highlighted_session: int - Session to highlight
            show_confidence_intervals: bool - Whether to show confidence interval bands
        """
        print(f"Creating percentile timeseries plot for subject with data keys: {list(subject_data.keys()) if subject_data else 'None'}")
        
        if not subject_data or 'sessions' not in subject_data:
            return self._create_empty_figure()
            
        sessions = subject_data['sessions']
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
        print(f"Show confidence intervals: {show_confidence_intervals}")
        
        # Create strata abbreviation mapping for hover info
        strata_sessions_map = {}
        if strata_data and len(strata_data) == len(sessions):
            for session, strata in zip(sessions, strata_data):
                strata_sessions_map[session] = self._get_strata_abbreviation(strata)
        
        # Plot each feature using percentile data with confidence intervals
        for i, feature in enumerate(features_to_plot):
            # Look for percentile data
            percentile_key = f"{feature}_percentiles"
            ci_lower_key = f"{feature}_percentile_ci_lower"
            ci_upper_key = f"{feature}_percentile_ci_upper"
            
            if percentile_key not in subject_data:
                print(f"No percentile data found for {feature}, skipping")
                continue
                
            percentile_data = subject_data[percentile_key]
            ci_lower_data = subject_data.get(ci_lower_key, [])
            ci_upper_data = subject_data.get(ci_upper_key, [])
            
            print(f"Using percentile data for {feature}: {len(percentile_data)} values")
            
            # Check if we have CI data
            has_ci_data = (len(ci_lower_data) == len(percentile_data) and 
                          len(ci_upper_data) == len(percentile_data) and
                          show_confidence_intervals)
            
            if has_ci_data:
                print(f"CI data available for {feature}")
            
            # Get bootstrap indicator data for this feature
            bootstrap_indicator_key = f"{feature}_bootstrap_enhanced"
            bootstrap_data = subject_data.get(bootstrap_indicator_key, [])
            
            # Filter out invalid percentile values (-1 represents missing data)
            valid_data = []
            valid_ci_lower = []
            valid_ci_upper = []
            valid_bootstrap_indicators = []
            
            for j, (session, percentile) in enumerate(zip(sessions, percentile_data)):
                if percentile is not None and not pd.isna(percentile) and percentile != -1:
                    valid_data.append((session, percentile))
                    
                    # Add CI data if available
                    if has_ci_data and j < len(ci_lower_data) and j < len(ci_upper_data):
                        ci_lower = ci_lower_data[j]
                        ci_upper = ci_upper_data[j]
                        
                        if (ci_lower is not None and not pd.isna(ci_lower) and ci_lower != -1 and
                            ci_upper is not None and not pd.isna(ci_upper) and ci_upper != -1):
                            valid_ci_lower.append(ci_lower)
                            valid_ci_upper.append(ci_upper)
                        else:
                            valid_ci_lower.append(None)
                            valid_ci_upper.append(None)
                    
                    # Add bootstrap indicator if available
                    if j < len(bootstrap_data):
                        valid_bootstrap_indicators.append(bootstrap_data[j])
                    else:
                        valid_bootstrap_indicators.append(False)
            
            if len(valid_data) < 2:
                print(f"Insufficient valid percentile data for {feature}: {len(valid_data)} points")
                continue
                
            valid_sessions, valid_percentiles = zip(*valid_data)
            valid_sessions = list(valid_sessions)
            valid_percentiles = list(valid_percentiles)
            
            print(f"Valid percentile data for {feature}: {len(valid_percentiles)} points, range: {min(valid_percentiles):.1f}% to {max(valid_percentiles):.1f}%")
            
            # Get color for this feature
            feature_color = self.feature_colors.get(feature, '#000000')
            
            # Add confidence interval bands if available
            if has_ci_data and len(valid_ci_lower) == len(valid_sessions):
                # Filter CI data to match valid sessions
                valid_ci_lower_filtered = [ci for ci in valid_ci_lower if ci is not None]
                valid_ci_upper_filtered = [ci for ci in valid_ci_upper if ci is not None]
                valid_sessions_ci = [session for session, ci_lower, ci_upper in zip(valid_sessions, valid_ci_lower, valid_ci_upper) 
                                   if ci_lower is not None and ci_upper is not None]
                
                if len(valid_ci_lower_filtered) > 0 and len(valid_sessions_ci) > 0:
                    # Add upper bound (invisible line)
                    fig.add_trace(go.Scatter(
                        x=valid_sessions_ci,
                        y=valid_ci_upper_filtered,
                        fill=None,
                        mode='lines',
                        line=dict(color='rgba(0,0,0,0)'),
                        showlegend=False,
                        hoverinfo='skip',
                        name=f'{feature}_ci_upper'
                    ))
                    
                    # Add lower bound with fill
                    fig.add_trace(go.Scatter(
                        x=valid_sessions_ci,
                        y=valid_ci_lower_filtered,
                        fill='tonexty',
                        mode='lines',
                        line=dict(color='rgba(0,0,0,0)'),
                        fillcolor=f'rgba({self._hex_to_rgb(feature_color)}, 0.2)',
                        showlegend=False,
                        hoverinfo='skip',
                        name=f'{feature}_ci_band'
                    ))
                    
                    print(f"Added CI bands for {feature}: {len(valid_sessions_ci)} sessions")

            # Prepare hover data
            if i == 0:  # First trace gets strata info
                strata_hover_info = [strata_sessions_map.get(session, 'Unknown') for session in valid_sessions]
                
                # Enhanced hover template with CI information
                if has_ci_data and len(valid_ci_lower) == len(valid_sessions):
                    # Determine CI method indicator for each session
                    ci_method_indicators = ['Bootstrap' if bootstrap else 'Wilson' for bootstrap in valid_bootstrap_indicators]
                    
                    hover_template = (f"<b>Strata: %{{customdata[1]}}</b><br><br>" +
                                    f"<b>{feature.replace('_', ' ').title()}</b><br>" +
                                    "Percentile: %{y:.1f}%<br>" +
                                    "95% CI: %{customdata[2]:.1f}% - %{customdata[3]:.1f}%<br>" +
                                    "CI Method: %{customdata[4]}<extra></extra>")
                    
                    # Create custom data with bootstrap/Wilson indicators
                    custom_data = list(zip(valid_percentiles, strata_hover_info, 
                                         [ci or 0 for ci in valid_ci_lower],
                                         [ci or 0 for ci in valid_ci_upper],
                                         ci_method_indicators))
                else:
                    hover_template = (f"<b>Strata: %{{customdata[1]}}</b><br><br>" +
                                    f"<b>{feature.replace('_', ' ').title()}</b><br>" +
                                    "Percentile: %{y:.1f}%<extra></extra>")
                    custom_data = list(zip(valid_percentiles, strata_hover_info))
            else:  # Subsequent traces don't include strata
                if has_ci_data and len(valid_ci_lower) == len(valid_sessions):
                    # Determine CI method indicator for each session
                    ci_method_indicators = ['Bootstrap' if bootstrap else 'Wilson' for bootstrap in valid_bootstrap_indicators]
                    
                    hover_template = (f"<b>{feature.replace('_', ' ').title()}</b><br>" +
                                    "Percentile: %{y:.1f}%<br>" +
                                    "95% CI: %{customdata[1]:.1f}% - %{customdata[2]:.1f}%<br>" +
                                    "CI Method: %{customdata[3]}<extra></extra>")
                    
                    custom_data = list(zip(valid_percentiles,
                                         [ci or 0 for ci in valid_ci_lower],
                                         [ci or 0 for ci in valid_ci_upper],
                                         ci_method_indicators))
                else:
                    hover_template = (f"<b>{feature.replace('_', ' ').title()}</b><br>" +
                                    "Percentile: %{y:.1f}%<extra></extra>")
                    custom_data = list(zip(valid_percentiles))
            
            # Create trace for feature percentiles (main line)
            fig.add_trace(go.Scatter(
                x=valid_sessions,
                y=valid_percentiles,
                mode='lines',
                name=feature.replace('_', ' ').replace('abs(', '|').replace(')', '|').title(),
                line=dict(
                    color=feature_color,
                    width=2,
                    shape='spline',
                    smoothing=1.0
                ),
                hovertemplate=hover_template,
                customdata=custom_data
            ))
        
        # Add overall percentile trace with CI (distinctive styling) - only if selected
        if show_overall_percentile and 'overall_percentiles' in subject_data:
            overall_percentiles = subject_data['overall_percentiles']
            overall_ci_lower = subject_data.get('overall_percentiles_ci_lower', [])
            overall_ci_upper = subject_data.get('overall_percentiles_ci_upper', [])
            
            print(f"Adding overall percentile trace: {len(overall_percentiles)} values")
            
            # Check if we have overall CI data
            has_overall_ci = (len(overall_ci_lower) == len(overall_percentiles) and 
                            len(overall_ci_upper) == len(overall_percentiles) and
                            show_confidence_intervals)
            
            # Get overall bootstrap indicator data
            overall_bootstrap_data = subject_data.get('overall_bootstrap_enhanced', [])
            
            # Filter out invalid values
            valid_overall_data = []
            valid_overall_ci_lower = []
            valid_overall_ci_upper = []
            valid_overall_bootstrap_indicators = []
            
            for j, (session, percentile) in enumerate(zip(sessions, overall_percentiles)):
                if percentile is not None and not pd.isna(percentile) and percentile != -1:
                    valid_overall_data.append((session, percentile))
                    
                    if has_overall_ci and j < len(overall_ci_lower) and j < len(overall_ci_upper):
                        ci_lower = overall_ci_lower[j]
                        ci_upper = overall_ci_upper[j]
                        
                        if (ci_lower is not None and not pd.isna(ci_lower) and ci_lower != -1 and
                            ci_upper is not None and not pd.isna(ci_upper) and ci_upper != -1):
                            valid_overall_ci_lower.append(ci_lower)
                            valid_overall_ci_upper.append(ci_upper)
                        else:
                            valid_overall_ci_lower.append(None)
                            valid_overall_ci_upper.append(None)
                    
                    # Add overall bootstrap indicator if available
                    if j < len(overall_bootstrap_data):
                        valid_overall_bootstrap_indicators.append(overall_bootstrap_data[j])
                    else:
                        valid_overall_bootstrap_indicators.append(False)
            
            if valid_overall_data:
                valid_sessions_overall, valid_percentiles_overall = zip(*valid_overall_data)
                valid_sessions_overall = list(valid_sessions_overall)
                valid_percentiles_overall = list(valid_percentiles_overall)
                
                print(f"Valid overall percentile data: {len(valid_percentiles_overall)} points, range: {min(valid_percentiles_overall):.1f}% to {max(valid_percentiles_overall):.1f}%")
                
                # Add overall CI bands if available
                if has_overall_ci and len(valid_overall_ci_lower) == len(valid_sessions_overall):
                    # Filter CI data to match valid sessions
                    valid_overall_ci_lower_filtered = [ci for ci in valid_overall_ci_lower if ci is not None]
                    valid_overall_ci_upper_filtered = [ci for ci in valid_overall_ci_upper if ci is not None]
                    valid_sessions_overall_ci = [session for session, ci_lower, ci_upper in zip(valid_sessions_overall, valid_overall_ci_lower, valid_overall_ci_upper) 
                                               if ci_lower is not None and ci_upper is not None]
                    
                    if len(valid_overall_ci_lower_filtered) > 0:
                        # Add upper bound (invisible line)
                        fig.add_trace(go.Scatter(
                            x=valid_sessions_overall_ci,
                            y=valid_overall_ci_upper_filtered,
                            fill=None,
                            mode='lines',
                            line=dict(color='rgba(0,0,0,0)'),
                            showlegend=False,
                            hoverinfo='skip',
                            name='overall_ci_upper'
                        ))
                        
                        # Add lower bound with fill (sea green with transparency)
                        fig.add_trace(go.Scatter(
                            x=valid_sessions_overall_ci,
                            y=valid_overall_ci_lower_filtered,
                            fill='tonexty',
                            mode='lines',
                            line=dict(color='rgba(0,0,0,0)'),
                            fillcolor='rgba(46, 139, 87, 0.2)',  # Sea green with transparency
                            showlegend=False,
                            hoverinfo='skip',
                            name='overall_ci_band'
                        ))
                        
                        print(f"Added overall CI bands: {len(valid_sessions_overall_ci)} sessions")
                
                # Create strata info for overall percentile hover
                strata_hover_info_overall = [strata_sessions_map.get(session, 'Unknown') for session in valid_sessions_overall]
                
                # Determine if this is the only trace being plotted
                is_only_trace = len(features_to_plot) == 0
                
                # Create enhanced hover template with CI info
                if is_only_trace or not features_to_plot:  # If overall percentile is the only thing selected
                    if has_overall_ci and len(valid_overall_ci_lower) == len(valid_sessions_overall):
                        # Determine CI method indicator for each session
                        overall_ci_method_indicators = ['Bootstrap' if bootstrap else 'Wilson' for bootstrap in valid_overall_bootstrap_indicators]
                        
                        hover_template = (f"<b>Strata: %{{customdata[1]}}</b><br><br>" +
                                        "<b>Overall Percentile</b><br>" +
                                        "Percentile: %{y:.1f}%<br>" +
                                        "95% CI: %{customdata[2]:.1f}% - %{customdata[3]:.1f}%<br>" +
                                        "CI Method: %{customdata[4]}<extra></extra>")
                        
                        custom_data = list(zip(valid_percentiles_overall, strata_hover_info_overall,
                                             [ci or 0 for ci in valid_overall_ci_lower],
                                             [ci or 0 for ci in valid_overall_ci_upper],
                                             overall_ci_method_indicators))
                    else:
                        hover_template = (f"<b>Strata: %{{customdata[1]}}</b><br><br>" +
                                        "<b>Overall Percentile</b><br>" +
                                        "Percentile: %{y:.1f}%<extra></extra>")
                        custom_data = list(zip(valid_percentiles_overall, strata_hover_info_overall))
                else:  # If there are other features, don't repeat strata info
                    if has_overall_ci and len(valid_overall_ci_lower) == len(valid_sessions_overall):
                        # Determine CI method indicator for each session
                        overall_ci_method_indicators = ['Bootstrap' if bootstrap else 'Wilson' for bootstrap in valid_overall_bootstrap_indicators]
                        
                        hover_template = ("<b>Overall Percentile</b><br>" +
                                        "Percentile: %{y:.1f}%<br>" +
                                        "95% CI: %{customdata[1]:.1f}% - %{customdata[2]:.1f}%<br>" +
                                        "CI Method: %{customdata[3]}<extra></extra>")
                        
                        custom_data = list(zip(valid_percentiles_overall,
                                             [ci or 0 for ci in valid_overall_ci_lower],
                                             [ci or 0 for ci in valid_overall_ci_upper],
                                             overall_ci_method_indicators))
                    else:
                        hover_template = ("<b>Overall Percentile</b><br>" +
                                        "Percentile: %{y:.1f}%<extra></extra>")
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
        
        # PHASE 2: Add outlier markers to percentile time series plot
        self._add_outlier_markers(fig, sessions, subject_data.get('is_outlier', []))
        
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
    
    def _hex_to_rgb(self, hex_color):
        """Convert hex color to RGB string for transparency"""
        hex_color = hex_color.lstrip('#')
        try:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return f"{r}, {g}, {b}"
        except:
            return "128, 128, 128"  # Default gray 

    def _add_outlier_markers(self, fig, sessions, outlier_data):
        """
        Add purple markers for outlier sessions on the percentile time series plot
        
        Parameters:
            fig: plotly.graph_objects.Figure - The percentile time series figure
            sessions: list - List of session numbers
            outlier_data: list - List of boolean outlier indicators for each session
        """
        if not sessions or not outlier_data or len(outlier_data) != len(sessions):
            print("No outlier data available or length mismatch for percentile time series markers")
            return
        
        outlier_sessions = []
        
        # Find outlier sessions
        for session, is_outlier in zip(sessions, outlier_data):
            if is_outlier:
                outlier_sessions.append(session)
        
        if not outlier_sessions:
            return
        
        # Add purple markers for outlier sessions at 95% on y-axis
        fig.add_trace(go.Scatter(
            x=outlier_sessions,
            y=[95] * len(outlier_sessions),  # Position near top of percentile range
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
        
        print(f"Added outlier markers to percentile plot for {len(outlier_sessions)} sessions: {outlier_sessions}") 