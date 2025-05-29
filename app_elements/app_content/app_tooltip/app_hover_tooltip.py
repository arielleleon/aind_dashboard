from dash import html, dcc
import pandas as pd
from typing import Dict, Any, List, Optional
import json

class AppHoverTooltip:
    """
    Hover tooltip component for displaying subject information
    Shows strata, overall percentile, active feature alerts, and threshold alerts
    """
    
    def __init__(self, app_utils=None):
        """
        Initialize tooltip component
        
        Parameters:
            app_utils: Shared AppUtils instance for data access
        """
        self.app_utils = app_utils
        
        # Feature display order for consistent comparison
        self.feature_display_order = [
            'finished_trials',
            'ignore_rate', 
            'total_trials',
            'foraging_performance',
            'abs(bias_naive)'
        ]
        
        # Feature display names (clean formatting)
        self.feature_display_names = {
            'finished_trials': 'Finished Trials',
            'ignore_rate': 'Ignore Rate',
            'total_trials': 'Total Trials', 
            'foraging_performance': 'Foraging Performance',
            'abs(bias_naive)': 'Bias'
        }
        
        # Color mapping matching DataTable alert schema
        self.alert_colors = {
            'SB': '#FF6B35',  # Dark orange (Severely Below)
            'B': '#FFB366',   # Light orange (Below)
            'G': '#4A90E2',   # Light blue (Good)
            'SG': '#2E5A87',  # Dark blue (Severely Good)
            'N': None,        # Normal (no color)
            'NS': None,       # Not Scored (no color)
            'T': '#795548'    # Brown (Threshold alerts)
        }

    def get_subject_tooltip_data(self, subject_id: str) -> Optional[Dict[str, Any]]:
        """
        Extract tooltip data for a specific subject from UI cache
        
        Parameters:
            subject_id: Subject ID to get data for
            
        Returns:
            Dict with tooltip data or None if subject not found
        """
        if not self.app_utils:
            return None
            
        # Get UI-optimized table display data (same as DataTable)
        table_data = self.app_utils.get_table_display_data(use_cache=True)
        
        if not table_data:
            return None
            
        # Find subject in table data
        subject_data = None
        for row in table_data:
            if row.get('subject_id') == subject_id:
                subject_data = row
                break
                
        if not subject_data:
            return None
            
        # Extract core information
        tooltip_data = {
            'subject_id': subject_id,
            'strata': subject_data.get('strata_abbr', 'N/A'),
            'overall_percentile': subject_data.get('session_overall_percentile'),
            'overall_category': subject_data.get('overall_percentile_category', 'NS'),
            'active_features': [],
            'threshold_alerts': []
        }
        
        # Extract active feature alerts (not N or NS)
        for feature in self.feature_display_order:
            category_col = f"{feature}_category"
            percentile_col = f"{feature}_session_percentile"
            
            category = subject_data.get(category_col)
            percentile = subject_data.get(percentile_col)
            
            # Only include if category is active (not N or NS)
            if category and category not in ['N', 'NS']:
                tooltip_data['active_features'].append({
                    'feature': feature,
                    'display_name': self.feature_display_names[feature],
                    'category': category,
                    'percentile': percentile,
                    'color': self.alert_colors.get(category)
                })
        
        # Extract active threshold alerts
        threshold_alert = subject_data.get('threshold_alert', 'N')
        if threshold_alert == 'T':
            # Check specific threshold alerts
            total_sessions_alert = subject_data.get('total_sessions_alert', '')
            stage_sessions_alert = subject_data.get('stage_sessions_alert', '')
            water_day_alert = subject_data.get('water_day_total_alert', '')
            
            if 'T |' in total_sessions_alert:
                value = total_sessions_alert.split('|')[1].strip()
                tooltip_data['threshold_alerts'].append({
                    'type': 'Total Sessions',
                    'value': f"{value} sessions",
                    'color': self.alert_colors['T']
                })
                
            if 'T |' in stage_sessions_alert:
                parts = stage_sessions_alert.split('|')
                stage = parts[1].strip() if len(parts) > 1 else ''
                sessions = parts[2].strip() if len(parts) > 2 else ''
                tooltip_data['threshold_alerts'].append({
                    'type': 'Stage Sessions',
                    'value': f"{stage}: {sessions}",
                    'color': self.alert_colors['T']
                })
                
            if 'T |' in water_day_alert:
                value = water_day_alert.split('|')[1].strip()
                tooltip_data['threshold_alerts'].append({
                    'type': 'Water Day Total', 
                    'value': f"{value} mL",
                    'color': self.alert_colors['T']
                })
        
        return tooltip_data

    def create_tooltip_content(self, tooltip_data: Dict[str, Any]) -> html.Div:
        """
        Create the HTML content for the tooltip
        
        Parameters:
            tooltip_data: Processed tooltip data from get_subject_tooltip_data
            
        Returns:
            html.Div: Tooltip content component
        """
        if not tooltip_data:
            return html.Div()
            
        content_elements = []
        
        # Header: Subject ID and Strata
        header = html.Div([
            html.Div(tooltip_data['subject_id'], className="tooltip-subject-id"),
            html.Div(f"Strata: {tooltip_data['strata']}", className="tooltip-strata")
        ], className="tooltip-header")
        content_elements.append(header)
        
        # Overall Percentile with color
        overall_percentile = tooltip_data.get('overall_percentile')
        overall_category = tooltip_data.get('overall_category', 'NS')
        
        if overall_percentile is not None and not pd.isna(overall_percentile):
            percentile_text = f"{overall_percentile:.1f}%"
            overall_color = self.alert_colors.get(overall_category)
            
            percentile_style = {}
            if overall_color:
                percentile_style = {
                    'color': overall_color,
                    'fontWeight': '600'
                }
            
            overall_div = html.Div([
                html.Span("Overall: ", className="tooltip-label"),
                html.Span(percentile_text, style=percentile_style, className="tooltip-value")
            ], className="tooltip-overall")
            content_elements.append(overall_div)
        else:
            # Not Scored case
            overall_div = html.Div([
                html.Span("Overall: ", className="tooltip-label"),
                html.Span("Not Scored", className="tooltip-value tooltip-ns")
            ], className="tooltip-overall")
            content_elements.append(overall_div)
        
        # Active Feature Alerts (only if any exist)
        active_features = tooltip_data.get('active_features', [])
        if active_features:
            feature_elements = []
            for feature_data in active_features:
                percentile = feature_data.get('percentile')
                if percentile is not None and not pd.isna(percentile):
                    percentile_text = f"{percentile:.1f}%"
                else:
                    percentile_text = "N/A"
                
                feature_style = {}
                if feature_data.get('color'):
                    feature_style = {
                        'color': feature_data['color'],
                        'fontWeight': '600'
                    }
                
                feature_div = html.Div([
                    html.Span(f"{feature_data['display_name']}: ", className="tooltip-feature-label"),
                    html.Span(percentile_text, style=feature_style, className="tooltip-feature-value")
                ], className="tooltip-feature-item")
                feature_elements.append(feature_div)
            
            features_section = html.Div(feature_elements, className="tooltip-features")
            content_elements.append(features_section)
        
        # Threshold Alerts (only if any exist)
        threshold_alerts = tooltip_data.get('threshold_alerts', [])
        if threshold_alerts:
            threshold_elements = []
            for alert_data in threshold_alerts:
                alert_style = {
                    'color': alert_data.get('color', '#795548'),
                    'fontWeight': '600'
                }
                
                alert_div = html.Div([
                    html.Span(f"{alert_data['type']}: ", className="tooltip-threshold-label"),
                    html.Span(alert_data['value'], style=alert_style, className="tooltip-threshold-value")
                ], className="tooltip-threshold-item")
                threshold_elements.append(alert_div)
            
            threshold_section = html.Div(threshold_elements, className="tooltip-thresholds")
            content_elements.append(threshold_section)
        
        return html.Div(content_elements, className="tooltip-content")

    def build_tooltip_container(self) -> html.Div:
        """
        Build the tooltip container that will be positioned dynamically
        
        Returns:
            html.Div: Empty tooltip container for dynamic content
        """
        return html.Div(
            id="subject-hover-tooltip",
            className="subject-tooltip hidden",
            style={
                'position': 'absolute',
                'zIndex': 9999,
                'pointerEvents': 'none',  # Don't interfere with mouse events
                'opacity': 0,
                'transition': 'opacity 0.15s ease-in-out'
            }
        )

    def get_empty_tooltip(self) -> html.Div:
        """
        Return empty tooltip content
        
        Returns:
            html.Div: Empty div for hiding tooltip
        """
        return html.Div() 