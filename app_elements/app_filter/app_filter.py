from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
from app_utils import AppUtils

class AppFilter:
    def __init__(self):
        # Initialize app utilities to get data for dropdowns
        app_utils = AppUtils()
        df = app_utils.get_session_data()
        
        # Get unique values for each filter dropdown
        self.stage_options = sorted(df['current_stage_actual'].dropna().unique().tolist())
        self.curriculum_options = sorted(df['curriculum_name'].dropna().unique().tolist())
        self.rig_options = sorted(df['rig'].dropna().unique().tolist())
        self.trainer_options = sorted(df['trainer'].dropna().unique().tolist())
        self.pi_options = sorted(df['PI'].dropna().unique().tolist())

        # Define time window options
        self.time_window_options = [
            {"label": "Last 7 days", "value": 7},
            {"label": "Last 30 days", "value": 30},
            {"label": "Last 60 days", "value": 60},
            {"label": "Last 90 days", "value": 90},
            {"label": "All time", "value": 365*10} 
        ]

        # Define sorting options 
        self.sort_options = [
            {"label": "Default", "value": "none"},
            {"label": "Overall Percentile (Ascending)", "value": "overall_percentile_asc"},
            {"label": "Overall Percentile (Descending)", "value": "overall_percentile_desc"}
        ]

        self.alert_category_options = [
            {"label": "All Alerts", "value": "all"},
            {"label": "SB", "value": "SB"},
            {"label": "B", "value": "B"},
            {"label": "N", "value": "N"},
            {"label": "G", "value": "G"},
            {"label": "SG", "value": "SG"},
            {"label": "NS", "value": "NS"},
            {"label": "Threshold", "value": "T"}
        ]


    def build(self):
        """
        Build filter component with a more compact layout for full width
        """
        return html.Div([
            # Filter header with title and clear button
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.I(className="fas fa-filter me-2"),
                        html.Span("Filters"),
                        html.Span(id="filter-count", className="filter-count ms-2")
                    ], className="filter-title"),
                    width="auto"
                ),
                dbc.Col(
                    html.Button("Clear all", id="clear-filters", className="clear-filters-btn"),
                    width="auto", className="ms-auto"
                )
            ], className="filter-header mb-2 align-items-center"),
            
            # Active filters display
            html.Div(id="active-filters", className="active-filters mb-2"),
            
            # Six column layout for filters to make it more compact
            dbc.Row([
                # Column 1: Sorting
                dbc.Col([
                    html.Label("Sort By", className="filter-label"),
                    dcc.Dropdown(
                        id="sort-option",
                        options=self.sort_options,
                        value="none",
                        clearable=False,
                        className="filter-dropdown"
                    )
                ], width=2, className="filter-column"),
                
                # Column 2: Alert filtering
                dbc.Col([
                    html.Label("Filter by Alert", className="filter-label"),
                    dcc.Dropdown(
                        id="alert-category-filter",
                        options=self.alert_category_options,
                        value="all",
                        clearable=False,
                        className="filter-dropdown"
                    )
                ], width=2, className="filter-column"),
                
                # Column 3: Time Window
                dbc.Col([
                    html.Label("Time Window", className="filter-label"),
                    dcc.Dropdown(
                        id="time-window-filter",
                        options=self.time_window_options,
                        value=30,
                        clearable=False,
                        className="filter-dropdown"
                    )
                ], width=2, className="filter-column"),
                
                # Column 4: Trainer/PI
                dbc.Col([
                    html.Label("Trainer/PI", className="filter-label"),
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id="trainer-filter",
                                options=[{"label": opt, "value": opt} for opt in self.trainer_options],
                                placeholder="Trainer",
                                clearable=True,
                                multi=True,
                                className="filter-dropdown-sm"
                            )
                        ], width=6, className="pe-1"),
                        dbc.Col([
                            dcc.Dropdown(
                                id="pi-filter",
                                options=[{"label": opt, "value": opt} for opt in self.pi_options],
                                placeholder="PI",
                                clearable=True,
                                multi=True,
                                className="filter-dropdown-sm"
                            )
                        ], width=6, className="ps-1")
                    ], className="g-0")
                ], width=2, className="filter-column"),
                
                # Column 5: Rig
                dbc.Col([
                    html.Label("Rig", className="filter-label"),
                    dcc.Dropdown(
                        id="rig-filter",
                        options=[{"label": opt, "value": opt} for opt in self.rig_options],
                        placeholder="Select rig",
                        clearable=True,
                        multi=True,
                        className="filter-dropdown"
                    )
                ], width=2, className="filter-column"),
                
                # Column 6: Stage/Curriculum
                dbc.Col([
                    html.Label("Stage/Curriculum", className="filter-label"),
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id="stage-filter",
                                options=[{"label": opt, "value": opt} for opt in self.stage_options],
                                placeholder="Stage",
                                clearable=True,
                                multi=True,
                                className="filter-dropdown-sm"
                            )
                        ], width=6, className="pe-1"),
                        dbc.Col([
                            dcc.Dropdown(
                                id="curriculum-filter",
                                options=[{"label": opt, "value": opt} for opt in self.curriculum_options],
                                placeholder="Curriculum",
                                clearable=True,
                                multi=True,
                                className="filter-dropdown-sm"
                            )
                        ], width=6, className="ps-1")
                    ], className="g-0")
                ], width=2, className="filter-column")
            ])
        ], className="filter-container mb-2", style={"height": "auto", "min-height": "120px"})