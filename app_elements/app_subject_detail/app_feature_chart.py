import plotly.graph_objects as go
from dash import dcc
import pandas as pd 

class AppFeatureChart:
    def __init__(self):
        pass
    
    def build(self, subject_id=None, app_utils=None):
        """
        Build subject-feature specific percentile chart using optimized data structure

        Parameters:
            subject_id (str): Subject ID to build chart for
            app_utils (AppUtils): App utilities instance for accessing optimized data

        Returns:
            dcc.Graph: Percentile chart
        """
        print(f"🎯 AppFeatureChart.build() called with subject_id: {subject_id}")
        
        if not subject_id or not app_utils:
            print("❌ Missing subject_id or app_utils - returning empty chart")
            return dcc.Graph(
                id='feature-percentile-chart',
                figure=go.Figure().add_annotation(
                    text="No subject selected",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=14, color="#666666")
                ),
                config={'displayModeBar': False},
                style={'height': '250px', 'width': '100%', 'minHeight': '250px', 'maxHeight': '250px'}
            )
        
        # CRITICAL FIX: Debug cache status
        print(f"📊 Checking cache status...")
        print(f"  session_level_data cached: {app_utils._cache.get('session_level_data') is not None}")
        print(f"  ui_structures cached: {app_utils._cache.get('ui_structures') is not None}")
        
        # CRITICAL FIX: Ensure unified pipeline has run
        if app_utils._cache.get('session_level_data') is None:
            print("🔄 Session data not cached - need to run unified pipeline first")
            try:
                raw_data = app_utils.get_session_data(use_cache=True)
                app_utils.process_data_pipeline(raw_data, use_cache=False)
                print("✅ Unified pipeline completed")
            except Exception as e:
                print(f"❌ Error running unified pipeline: {str(e)}")
                return self._create_error_chart("Data processing error")
        
        # Get optimized feature rank data for this subject
        print(f"📊 Getting feature rank data for subject {subject_id}...")
        feature_rank_data = app_utils.get_feature_rank_data(subject_id, use_cache=True)
        print(f"Feature rank data retrieved: {bool(feature_rank_data)}")
        
        if not feature_rank_data or not feature_rank_data.get('features'):
            print(f"❌ No feature rank data available for subject {subject_id}")
            # CRITICAL FIX: Check if subject exists in the data
            session_data = app_utils._cache.get('session_level_data')
            if session_data is not None:
                available_subjects = session_data['subject_id'].unique()
                if subject_id in available_subjects:
                    print(f"Subject {subject_id} exists in session data but has no feature data")
                    return self._create_error_chart("No feature data available for this subject")
                else:
                    print(f"Subject {subject_id} not found in session data")
                    return self._create_error_chart("Subject not found")
            
            return self._create_error_chart("No feature data available")
        
        print(f"✅ Feature rank data found with {len(feature_rank_data.get('features', {}))} features")
        
        # Extract feature data from optimized structure
        features = []
        percentiles = []
        colors = []
        
        # Color mapping for different alert categories
        color_map = {
            'B': '#FF8C40',   # Orange - Bad
            'SB': '#FF8C40',  # Orange - Severely Bad
            'G': '#4D94DA',   # Blue - Good
            'SG': '#4D94DA',  # Blue - Severely Good
            'N': '#CCCCCC',   # Grey - Normal
            'NS': '#CCCCCC'   # Grey - Not Scored
        }

        # Process feature data from optimized structure
        feature_data = feature_rank_data.get('features', {})
        valid_features_count = 0
        
        for feature_name, data in feature_data.items():
            percentile = data.get('percentile')
            category = data.get('category', 'NS')
            
            print(f"  Feature {feature_name}: percentile={percentile}, category={category}")
            
            # Only include features with valid percentile data
            if percentile is not None and not pd.isna(percentile):
                # Clean up feature name for display
                display_name = feature_name.replace('_', '').replace('abs(', '').replace(')', '').replace('bias', 'bias')
                features.append(display_name)
                percentiles.append(percentile)
                colors.append(color_map.get(category, '#CCCCCC'))
                valid_features_count += 1
                print(f"    ✅ Added to chart: {display_name} = {percentile:.1f}%")
            else:
                print(f"    ❌ Skipped (invalid percentile): {feature_name}")

        print(f"📈 Chart will include {valid_features_count} valid features")
        
        # If no valid features, return empty chart
        if not features:
            print("❌ No valid features found - returning empty chart with message")
            return self._create_error_chart("No valid feature percentiles calculated")

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
            marker_opacity=0.7,
            text=[f"{p:.1f}%" for p in percentiles],
            textposition='auto',
            hoverinfo='text',
            hovertext=[f"{f}: {p:.1f}%" for f, p in zip(features, percentiles)],
            width=0.6
        ))

        fig.update_layout(
            title=None,
            margin=dict(l=0, r=0, t=10, b=10),
            yaxis=dict(
                title='Percentile',
                range=[0, max_y],
                tickvals=[0, 25, 50, 75, 100][:int(max_y/25)+1],
                gridcolor='#EEEEEE',
                showgrid=True
            ),
            xaxis=dict(
                title=None,
                automargin=True,
                tickangle=-45
            ),
            plot_bgcolor='white',
            height=250,
            width=None,
            bargap=0.15
        )

        print(f"📈 Feature chart created successfully for {subject_id}")
        return dcc.Graph(
            id='feature-percentile-chart',
            figure=fig,
            config={'displayModeBar': False},
            style={'height': '250px', 'width': '100%', 'minHeight': '250px', 'maxHeight': '250px'}
        )
    
    def _create_error_chart(self, message):
        """Create an error chart with a message"""
        return dcc.Graph(
            id='feature-percentile-chart',
            figure=go.Figure().add_annotation(
                text=message,
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="#666666")
            ),
            config={'displayModeBar': False},
            style={'height': '250px', 'width': '100%', 'minHeight': '250px', 'maxHeight': '250px'}
        )

    def build_legacy(self, subject_data=None):
        """
        Legacy build method for backward compatibility
        
        Parameters:
            subject_data (dict): Subject data dictionary (old format)
            
        Returns:
            dcc.Graph: Percentile chart
        """
        if not subject_data:
            return dcc.Graph(
                id='feature-percentile-chart',
                figure=go.Figure(),
                config={'displayModeBar': False}
            )
        
        # Extract feature percentiles from subject data (old format)
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
            category_key = f"{feature}_category"
            
            if percentile_key in subject_data and not pd.isna(subject_data[percentile_key]):
                features.append(feature.replace('_', '').replace('(', '').replace(')', ''))
                percentile = subject_data[percentile_key]
                percentiles.append(percentile)

                # Determine color based on alert category 
                if category_key in subject_data:
                    category = subject_data[category_key]
                    colors.append(color_map.get(category, '#CCCCCC'))
                else:
                    colors.append('#CCCCCC')

        # Calculate max percentile for y-axis
        max_percentile = max(percentiles) if percentiles else 100
        # Round up to nearest 10
        max_y = min(100, ((max_percentile // 10) + 1) * 10)

        # Create figure with vertical bars showing actual percentiles
        fig = go.Figure()

        if percentiles:  # Only add trace if we have data
            fig.add_trace(go.Bar(
                y=percentiles,
                x=features,
                orientation='v',
                marker_color=colors,
                marker_opacity=0.7,
                text=[f"{p:.1f}%" for p in percentiles],
                textposition='auto',
                hoverinfo='text',
                hovertext=[f"{f}: {p:.1f}%" for f, p in zip(features, percentiles)],
                width=0.6
            ))

        fig.update_layout(
            title=None,
            margin=dict(l=0, r=0, t=10, b=10),
            yaxis=dict(
                title='Percentile',
                range=[0, max_y],
                tickvals=[0, 25, 50, 75, 100][:int(max_y/25)+1],
                gridcolor='#EEEEEE',
                showgrid=True
            ),
            xaxis=dict(
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
            style={'height': '250px', 'width': '100%', 'minHeight': '250px', 'maxHeight': '250px'}
        )

        
        
        