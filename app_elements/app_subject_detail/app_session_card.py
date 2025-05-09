from dash import html, dcc
import dash_bootstrap_components as dbc
from .app_subject_image_loader import AppSubjectImageLoader

class AppSessionCard:
    def __init__(self):
        """Initialize session card component"""
        self.image_loader = AppSubjectImageLoader()
        
    def build(self, session_data, is_active=False):
        """
        Build a card for a specific session
        
        Parameters:
        -----------
        session_data : dict
            Dictionary containing session information 
        is_active : bool
            Whether this card is currently active/selected
        
        Returns:
        --------
        dash component
            The session card component
        """
        # Extract session data
        session_num = session_data.get('session', 'N/A')
        subject_id = session_data.get('subject_id', 'Unknown')
        session_date = session_data.get('session_date', 'Unknown')
        trainer = session_data.get('trainer', 'Unknown')
        rig = session_data.get('rig', 'Unknown')
        notes = session_data.get('notes', '')
        
        # Format session date if needed
        if hasattr(session_date, 'strftime'):
            session_date = session_date.strftime('%Y-%m-%d')
            
        # Get nwb_suffix - default to 0 if not present
        nwb_suffix = session_data.get('nwb_suffix', 0)
        
        # Generate image URLs directly
        choice_url = self.image_loader.get_s3_public_url(
            subject_id=subject_id,
            session_date=session_date,
            nwb_suffix=nwb_suffix,
            figure_suffix="choice_history.png"
        )
        
        lick_url = self.image_loader.get_s3_public_url(
            subject_id=subject_id,
            session_date=session_date,
            nwb_suffix=nwb_suffix,
            figure_suffix="lick_analysis.png"
        )
        
        # Prepare card class based on active status
        card_class = "session-card active" if is_active else "session-card"
        
        # Build the card
        return html.Div([
            # Session header with session number and metadata
            html.Div([
                html.Div([
                    html.Span("Session ", className="session-label"),
                    html.Span(session_num, className="session-number")
                ], className="session-header"),
                
                # Session metadata in a grid
                html.Div([
                    html.Div([
                        html.Span("Date: ", className="metadata-label"),
                        html.Span(session_date, className="metadata-value")
                    ], className="metadata-item"),
                    
                    html.Div([
                        html.Span("Trainer: ", className="metadata-label"),
                        html.Span(trainer, className="metadata-value")
                    ], className="metadata-item"),
                    
                    html.Div([
                        html.Span("Rig: ", className="metadata-label"),
                        html.Span(rig, className="metadata-value")
                    ], className="metadata-item")
                ], className="session-metadata-grid"),
                
                # Notes (if any)
                html.Div([
                    html.Div("Notes:", className="notes-label"),
                    html.Div(notes, className="notes-content")
                ], className="session-notes", style={'display': 'block' if notes else 'none'})
            ], className="session-info-container"),
            
            # Image container with session plots from S3 - direct URLs
            html.Div([
                # Container for choice history plot
                html.Div([
                    html.Div("Choice History", className="image-title"),
                    html.Div(className="session-image-container", children=[
                        html.Img(src=choice_url, className="session-image")
                    ])
                ], className="image-section"),
                
                # Container for lick analysis plot
                html.Div([
                    html.Div("Lick Analysis", className="image-title"),
                    html.Div(className="session-image-container", children=[
                        html.Img(src=lick_url, className="session-image")
                    ])
                ], className="image-section")
            ], className="session-images-container"),
            
        ], id={"type": "session-card", "index": f"{subject_id}-{session_num}"}, 
           className=card_class,
           n_clicks=0)  # Make clickable for selection