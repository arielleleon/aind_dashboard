from dash import html
import dash_bootstrap_components as dbc

class AppSubjectCompactInfo:
    def __init__(self):
        pass
    
    def build(self, subject_id=None, app_utils=None):
        """
        Build compact subject info display for above the bar plot
        
        Parameters:
            subject_id (str): Subject ID
            app_utils (AppUtils): App utilities instance for accessing cached data
            
        Returns:
            html.Div: Compact subject info
        """
        if not subject_id or not app_utils:
            return html.Div([
                html.Div("No subject selected", className="compact-subject-info")
            ], className="compact-info-section")
        
        # Get subject display data from UI cache
        subject_display_data = app_utils.get_subject_display_data(subject_id, use_cache=True)
        
        if not subject_display_data:
            return html.Div([
                html.Div(f"No data for {subject_id}", className="compact-subject-info")
            ], className="compact-info-section")
        
        # Extract data
        latest_data = subject_display_data.get('latest_session', {})
        strata = latest_data.get('strata', 'Unknown')
        
        # Get table display data for threshold alerts
        table_data = app_utils.get_table_display_data(use_cache=True)
        subject_table_data = next((row for row in table_data if row['subject_id'] == subject_id), {})
        
        # Build threshold alerts text
        threshold_alerts = []
        
        # Check for specific threshold alerts
        total_sessions_alert = subject_table_data.get('total_sessions_alert', '')
        stage_sessions_alert = subject_table_data.get('stage_sessions_alert', '')
        water_day_alert = subject_table_data.get('water_day_total_alert', '')
        
        if 'T |' in total_sessions_alert:
            value = total_sessions_alert.split('|')[1].strip()
            threshold_alerts.append(f"Total Sessions: {value}")
        
        if 'T |' in stage_sessions_alert:
            parts = stage_sessions_alert.split('|')
            stage_name = parts[1].strip() if len(parts) > 1 else ""
            sessions = parts[2].strip() if len(parts) > 2 else ""
            threshold_alerts.append(f"Stage {stage_name}: {sessions}")
        
        if 'T |' in water_day_alert:
            value = water_day_alert.split('|')[1].strip()
            threshold_alerts.append(f"Water: {value}mL")
        
        # Create individual info components
        info_components = [
            html.Div(f"Subject: {subject_id}", className="compact-subject-info"),
            html.Div(f"Strata: {strata}", className="compact-subject-info")
        ]
        
        # Add threshold alerts if present
        if threshold_alerts:
            alerts_text = f"Alerts: {', '.join(threshold_alerts)}"
            info_components.append(
                html.Div(
                    alerts_text, 
                    className="compact-subject-info",
                    style={"borderLeftColor": "#dc3545"}  # Red border for alerts
                )
            )
        
        return html.Div(
            info_components,
            className="compact-info-section"
        ) 