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
            {"label": "Overall Percentile (Ascending)", "value": "percentile_asc"},
            {"label": "Overall Percentile (Descending)", "value": "percentile_desc"},
            {"label": "Alert Category (Worst First)", "value": "alert_worst"},
            {"label": "Alert Category (Best First)", "value": "alert_best"},
        ]

        self.alert_category_options = [
            {"label": "All Alerts", "value": "all"},
            {"label": "SB", "value": "SB"},
            {"label": "B", "value": "B"},
            {"label": "N", "value": "N"},
            {"label": "G", "value": "G"},
            {"label": "SG", "value": "SG"}
        ]


    def build(self):
        """
        Build filter component with three columns of filters
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
            ], className="filter-header mb-3 align-items-center"),
            
            # Active filters display
            html.Div(id="active-filters", className="active-filters mb-3"),
            
            # Three column layout for filters
            dbc.Row([
                # Column 1: Sorting features
                dbc.Col([
                    html.Div("Sorting & Alert Filtering", className="section-header mb-2"),
                    
                    # Sort by dropdown
                    html.Div([
                        html.Label("Sort By", className="filter-label"),
                        dcc.Dropdown(
                            id="sort-option",
                            options=self.sort_options,
                            value="none",
                            clearable=False,
                            className="filter-dropdown"
                        )
                    ], className="mb-2"),
                    
                    # Alert category filter
                    html.Div([
                        html.Label("Filter by Alert", className="filter-label"),
                        dcc.Dropdown(
                            id="alert-category-filter",
                            options=self.alert_category_options,
                            value="all",
                            clearable=False,
                            className="filter-dropdown"
                        )
                    ], className="mb-2")
                ], width=4, className="filter-column"),
                
                # Column 2: Time Window, Trainer, PI
                dbc.Col([
                    html.Div("Data Filters", className="section-header mb-2"),
                    
                    # Time window filter
                    html.Div([
                        html.Label("Time Window", className="filter-label"),
                        dcc.Dropdown(
                            id="time-window-filter",
                            options=self.time_window_options,
                            value=30,  # Default to 30 days
                            clearable=False,
                            className="filter-dropdown"
                        )
                    ], className="mb-2"),
                    
                    # Trainer filter
                    html.Div([
                        html.Label("Trainer", className="filter-label"),
                        dcc.Dropdown(
                            id="trainer-filter",
                            options=[{"label": opt, "value": opt} for opt in self.trainer_options],
                            placeholder="Select trainer",
                            clearable=True,
                            className="filter-dropdown"
                        )
                    ], className="mb-2"),
                    
                    # PI filter
                    html.Div([
                        html.Label("PI", className="filter-label"),
                        dcc.Dropdown(
                            id="pi-filter",
                            options=[{"label": opt, "value": opt} for opt in self.pi_options],
                            placeholder="Select PI",
                            clearable=True,
                            className="filter-dropdown"
                        )
                    ], className="mb-2")
                ], width=4, className="filter-column"),
                
                # Column 3: Rig, Stage, Curriculum
                dbc.Col([
                    html.Div("\u00A0", className="section-header mb-2"),  # Invisible header for alignment
                    
                    # Rig filter
                    html.Div([
                        html.Label("Rig", className="filter-label"),
                        dcc.Dropdown(
                            id="rig-filter",
                            options=[{"label": opt, "value": opt} for opt in self.rig_options],
                            placeholder="Select rig",
                            clearable=True,
                            className="filter-dropdown"
                        )
                    ], className="mb-2"),
                    
                    # Stage filter
                    html.Div([
                        html.Label("Stage", className="filter-label"),
                        dcc.Dropdown(
                            id="stage-filter",
                            options=[{"label": opt, "value": opt} for opt in self.stage_options],
                            placeholder="Select stage",
                            clearable=True,
                            className="filter-dropdown"
                        )
                    ], className="mb-2"),
                    
                    # Curriculum filter
                    html.Div([
                        html.Label("Curriculum", className="filter-label"),
                        dcc.Dropdown(
                            id="curriculum-filter",
                            options=[{"label": opt, "value": opt} for opt in self.curriculum_options],
                            placeholder="Select curriculum",
                            clearable=True,
                            className="filter-dropdown"
                        )
                    ], className="mb-2")
                ], width=4, className="filter-column")
            ])
        ], className="filter-container mb-3")