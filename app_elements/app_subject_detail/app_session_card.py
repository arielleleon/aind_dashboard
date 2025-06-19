"""
Session card component for AIND Dashboard

This module creates individual session information cards with session metadata,
image integration, and hover details.
"""

import sys
import traceback

from dash import html

from app_utils.simple_logger import get_logger

from .app_subject_image_loader import AppSubjectImageLoader

logger = get_logger("session_card")


class AppSessionCard:
    def __init__(self):
        """Initialize session card component"""
        self.image_loader = AppSubjectImageLoader()

    def build(self, session_data, is_active=False):
        """
        Build a card for a specific session

        Parameters:
        session_data : dict
            Dictionary containing session information
        is_active : bool
            Whether this card is currently active/selected

        Returns:
        dash component
            The session card component
        """
        # Extract session data
        session_num = session_data.get("session", "N/A")
        subject_id = session_data.get("subject_id", "Unknown")
        session_date = session_data.get("session_date", "Unknown")
        trainer = session_data.get("trainer", "Unknown")
        rig = session_data.get("rig", "Unknown")
        notes = session_data.get("notes", "")

        # Format session date if needed
        if hasattr(session_date, "strftime"):
            session_date = session_date.strftime("%Y-%m-%d")

        # Get nwb_suffix
        nwb_suffix = session_data.get("nwb_suffix", 0)

        # Generate image URL directly
        try:
            choice_url = self.image_loader.get_s3_public_url(
                subject_id=subject_id,
                session_date=session_date,
                nwb_suffix=nwb_suffix,
                figure_suffix="choice_history.png",
            )
            logger.info(f"Generated image URL: {choice_url}")
        except Exception as e:
            logger.error(f"ERROR generating image URL: {str(e)}")
            choice_url = ""  # Fallback empty URL

        # Extract water total
        water_total = session_data.get("water_day_total", "N/A")
        if isinstance(water_total, (int, float)):
            water_total = f"{water_total:.2f} ml"

        # Prepare card class based on active status
        card_class = "session-card active" if is_active else "session-card"

        # Add debug wrapper to track click events
        debug_wrapper = {
            "data-debug": f"session-card-{subject_id}-{session_num}",
            "data-active": str(is_active).lower(),
            "data-session-num": str(session_num),
            "data-subject-id": subject_id,
        }

        # Build the card with a two-column layout
        try:
            card = html.Div(
                [
                    # Two-column container
                    html.Div(
                        [
                            # Left column - metadata
                            html.Div(
                                [
                                    # Session header
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Span(
                                                        "Session ",
                                                        className="session-label",
                                                    ),
                                                    html.Span(
                                                        session_num,
                                                        className="session-number",
                                                    ),
                                                ],
                                                className="session-header",
                                            ),
                                            html.Div(
                                                session_date, className="session-date"
                                            ),
                                        ],
                                        className="session-header-row",
                                    ),
                                    # Metadata items
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Span(
                                                        "Rig: ",
                                                        className="metadata-label",
                                                    ),
                                                    html.Span(
                                                        rig, className="metadata-value"
                                                    ),
                                                ],
                                                className="metadata-item",
                                            ),
                                            html.Div(
                                                [
                                                    html.Span(
                                                        "Trainer: ",
                                                        className="metadata-label",
                                                    ),
                                                    html.Span(
                                                        trainer,
                                                        className="metadata-value",
                                                    ),
                                                ],
                                                className="metadata-item",
                                            ),
                                            html.Div(
                                                [
                                                    html.Span(
                                                        "Water: ",
                                                        className="metadata-label",
                                                    ),
                                                    html.Span(
                                                        water_total,
                                                        className="metadata-value",
                                                    ),
                                                ],
                                                className="metadata-item",
                                            ),
                                        ],
                                        className="metadata-section",
                                    ),
                                    # Notes section
                                    html.Div(
                                        [
                                            html.Div(
                                                "Notes: ", className="notes-label"
                                            ),
                                            html.Div(notes, className="notes-content"),
                                        ],
                                        className="notes-section",
                                        style={"display": "block" if notes else "none"},
                                    ),
                                ],
                                className="session-metadata-column",
                            ),
                            # Right column - simple image
                            html.Div(
                                [
                                    html.Img(
                                        src=choice_url,
                                        className="session-image-large",
                                        id={
                                            "type": "session-image",
                                            "index": f"{subject_id}-{session_num}",
                                        },
                                    )
                                ],
                                className="session-image-column",
                            ),
                        ],
                        className="session-card-row",
                    )
                ],
                id={"type": "session-card", "index": f"{subject_id}-{session_num}"},
                className=card_class,
                n_clicks=0,
                style={"cursor": "pointer"},
                **debug_wrapper,
            )

            return card

        except Exception as e:
            logger.error(f"ERROR building session card: {str(e)}")
            print(traceback.format_exc(), file=sys.stderr)

            return html.Div(
                className="session-card-error",
                style={"color": "red", "padding": "10px"},
            )
