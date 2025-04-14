import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from app_utils import AppUtils

class RankChangePlot:
    """
    Minimalist component for visualizing changes in subject performance categories over time
    """
    
    def __init__(self):
        """Initialize the rank change plot component"""
        # Initialize app utilities for data access
        self.app_utils = AppUtils()
        
        # Category colors
        self.category_colors = {
            'Below': '#FFC380',    # Orange for below average
            'Average': '#A6A6A6',  # Gray for average
            'Above': '#9FC5E8',    # Blue for above average
        }
        
    def build(self, window_days=None):
        """
        Build and return the rank change plot figure
        """
        # Default window days if not specified
        if window_days is None:
            window_days = 30
            
        try:
            # Get session data
            df = self.app_utils.get_session_data()
            
            # Create figure
            fig = go.Figure()
            
            # Check for empty dataframe
            if df.empty:
                return self._empty_figure("No data available")
                
            # Ensure session_date is datetime
            df['session_date'] = pd.to_datetime(df['session_date'])
            
            # Get reference date (most recent session date)
            reference_date = df['session_date'].max()
            start_date = reference_date - timedelta(days=window_days)
            
            # Filter data to analysis window
            window_df = df[(df['session_date'] >= start_date) & (df['session_date'] <= reference_date)].copy()
            
            if window_df.empty:
                return self._empty_figure("No data in selected time window")
                
            # Create time bins (weekly)
            time_bins = pd.date_range(start=start_date, end=reference_date, freq='W')
            if len(time_bins) < 2:
                time_bins = pd.date_range(start=start_date, end=reference_date, periods=5)
                
            # Format time bins for display
            time_bin_labels = [str(tb.date()) for tb in time_bins]
            
            # Get percentile category data
            df_with_categories = self._get_category_data(window_df, time_bins, reference_date)
            
            # If no categories data, return empty figure
            if df_with_categories is None or df_with_categories.empty:
                return self._empty_figure("No performance data available for analysis")
                
            # Add data for Below Average
            below_data = self._create_category_data(df_with_categories, ['SB', 'B'], time_bin_labels)
            if below_data is not None:
                fig.add_trace(
                    go.Scatter(
                        x=time_bin_labels,
                        y=below_data,
                        mode='lines+markers',
                        name='Below Average',
                        line=dict(color=self.category_colors['Below'], width=3),
                        marker=dict(size=8)
                    )
                )
            
            # Add data for Average
            average_data = self._create_category_data(df_with_categories, ['N'], time_bin_labels)
            if average_data is not None:
                fig.add_trace(
                    go.Scatter(
                        x=time_bin_labels,
                        y=average_data,
                        mode='lines+markers',
                        name='Average',
                        line=dict(color=self.category_colors['Average'], width=3),
                        marker=dict(size=8)
                    )
                )
            
            # Add data for Above Average
            above_data = self._create_category_data(df_with_categories, ['G', 'SG'], time_bin_labels)
            if above_data is not None:
                fig.add_trace(
                    go.Scatter(
                        x=time_bin_labels,
                        y=above_data,
                        mode='lines+markers',
                        name='Above Average',
                        line=dict(color=self.category_colors['Above'], width=3),
                        marker=dict(size=8)
                    )
                )
            
            # Update layout to ensure proper sizing in new layout
            fig.update_layout(
                title='Subject Performance Categories Over Time',
                xaxis=dict(title='Date', tickangle=-45),
                yaxis=dict(title='Proportion of Subjects', tickformat='.0%', range=[0, 1.0]),
                hovermode='closest',
                plot_bgcolor='white',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                margin=dict(l=40, r=20, t=60, b=40),  # Adjusted margins
                autosize=True  # Ensure the plot resizes with container
            )
            
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.1)'
            )
            
            return fig
            
        except Exception as e:
            import traceback
            print(f"Error creating rank change plot: {str(e)}")
            print(traceback.format_exc())
            return self._empty_figure(f"Error: {str(e)}")
    
    def _empty_figure(self, message):
        """Create an empty figure with a message"""
        fig = go.Figure()
        fig.update_layout(
            title='Subject Performance',
            annotations=[dict(
                text=message,
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )]
        )
        return fig
        
    def _get_category_data(self, window_df, time_bins, reference_date):
        """
        Get category data from the window dataframe
        """
        try:
            # Initialize new app_utils to avoid state issues
            app_utils = AppUtils()
            
            # Initialize reference processor
            reference_processor = app_utils.initialize_reference_processor(
                features_config={
                    'finished_trials': False,
                    'ignore_rate': True,
                    'total_trials': False,
                    'foraging_performance': False,
                    'abs(bias_naive)': True
                },
                window_days=30,
                min_sessions=1,
                min_days=1
            )
            
            # Process data
            stratified_data = app_utils.process_reference_data(
                df=window_df,
                reference_date=reference_date,
                remove_outliers=False
            )
            
            # Initialize quantile analyzer
            quantile_analyzer = app_utils.initialize_quantile_analyzer(stratified_data)
            
            # Get overall percentiles
            overall_percentiles = quantile_analyzer.calculate_overall_percentile()
            
            # Initialize alert service
            alert_service = app_utils.initialize_alert_service()
            
            # Map percentiles to categories
            overall_percentiles['category'] = overall_percentiles['overall_percentile'].apply(
                lambda p: alert_service.map_overall_percentile_to_category(p)
            )
            
            # Create a time bin column in the window data
            window_df['time_bin'] = None
            
            # Assign time bins (manually to avoid categorical issues)
            for i, bin_start in enumerate(time_bins):
                if i < len(time_bins) - 1:
                    bin_end = time_bins[i+1]
                else:
                    bin_end = reference_date + timedelta(days=1)
                    
                mask = (window_df['session_date'] >= bin_start) & (window_df['session_date'] < bin_end)
                window_df.loc[mask, 'time_bin'] = str(bin_start.date())
            
            # Get most recent session for each subject in each time bin
            window_df = window_df.sort_values('session_date', ascending=False)
            latest_sessions = window_df.drop_duplicates(subset=['subject_id', 'time_bin'], keep='first')
            
            # Join with percentile data
            result = pd.merge(
                latest_sessions[['subject_id', 'time_bin']],
                overall_percentiles[['subject_id', 'category']],
                on='subject_id',
                how='inner'
            )
            
            return result
            
        except Exception as e:
            print(f"Error getting category data: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None
    
    def _create_category_data(self, df, categories, time_bin_labels):
        """
        Create a list of proportions for a given set of categories across time bins
        """
        try:
            if df is None or df.empty:
                return None
                
            result = []
            
            for time_bin in time_bin_labels:
                # Get data for this time bin
                bin_data = df[df['time_bin'] == time_bin]
                
                if bin_data.empty:
                    result.append(0)
                    continue
                    
                # Count subjects in specified categories
                category_count = bin_data[bin_data['category'].isin(categories)].shape[0]
                total_count = bin_data.shape[0]
                
                # Calculate proportion
                proportion = category_count / total_count if total_count > 0 else 0
                
                result.append(proportion)
                
            return result
            
        except Exception as e:
            print(f"Error creating category data: {str(e)}")
            return None