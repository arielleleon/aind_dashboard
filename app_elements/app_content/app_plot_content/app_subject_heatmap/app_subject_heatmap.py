import plotly.graph_objects as go
import pandas as pd
import numpy as np
import math

class SubjectHeatmap:
    def __init__(self):
        """Initialize the subject heatmap component"""
        # Define the diverging color scale: dark orange (low) → grey (middle) → dark blue (high)
        self.colorscale = [
            [0.0, '#FFC380'],      # Dark orange for low percentiles
            [0.25, '#FFC380'],     # Medium orange
            [0.5, '#A6A6A6'],      # Grey for 50th percentile
            [0.75, '#9FC5E8'],     # Medium blue
            [1.0, '#9FC5E8']       # Dark blue for high percentiles
        ]
    
    def _calculate_grid_dimensions(self, n_subjects):
        """
        Calculate optimal grid dimensions for the given number of subjects
        
        Args:
            n_subjects: Number of subjects to display
            
        Returns:
            tuple: (rows, cols) for the grid
        """
        # Aim for a grid with aspect ratio close to 3:2 (typical plot shape)
        # This gives a reasonable layout for most numbers of subjects
        aspect_ratio = 1.5  # width:height
        cols = max(1, math.ceil(math.sqrt(n_subjects * aspect_ratio)))
        rows = max(1, math.ceil(n_subjects / cols))
        return rows, cols
    
    def create_figure(self, data):
        """
        Create heatmap figure for subject percentiles
        
        Args:
            data: DataFrame with subject_id and overall_percentile columns
            
        Returns:
            go.Figure: Plotly figure object
        """
        if data.empty:
            # Return empty figure if no data
            return go.Figure().update_layout(
                title="No subjects to display",
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=20, r=20, t=60, b=20)
            )
        
        # Extract relevant columns
        plot_data = data[['subject_id', 'overall_percentile', 'percentile_category']].copy()
        
        # Filter out NS subjects
        plot_data = plot_data[plot_data['percentile_category'] != 'NS']
        
        if plot_data.empty:
            # Return empty figure if no valid subjects
            return go.Figure().update_layout(
                title="No subjects with percentile scores to display",
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=20, r=20, t=60, b=20)
            )
        
        # Sort by percentile
        plot_data = plot_data.sort_values(by = 'overall_percentile')
        
        # Calculate grid dimensions
        rows, cols = self._calculate_grid_dimensions(len(plot_data))
        
        # Create z-values and text labels
        z = np.zeros((rows, cols))  # Initialize with zeros
        text = np.empty((rows, cols), dtype=object)  # For subject IDs
        mask = np.ones((rows, cols), dtype=bool) # Track which cells are empty
        
        # Fill grid with subject data
        for idx, row in enumerate(range(rows)):
            for col in range(cols):
                pos = row * cols + col
                if pos < len(plot_data):
                    subject = plot_data.iloc[pos]
                    text[row, col] = subject['subject_id']
                    z[row, col] = subject['overall_percentile']
                else:
                    # Empty cells beyond our data
                    text[row, col] = ""
                    mask[row, col] = False
        # Modify the color scale to handle empty cells
        modified_colorscale = [[0, 'white'], [0.001, self.colorscale[0][1]]]
        for i in range(1, len(self.colorscale)):
            # Scale colors to 0.001 to 1
            position = 0.001 + (self.colorscale[i][0] * 0.999)
            modified_colorscale.append([position, self.colorscale[i][1]])

        # Create heatmap figure
        fig = go.Figure(go.Heatmap(
            z=np.where(mask, z, -0.1),
            text=text,
            texttemplate="%{text}",
            hovertemplate='Subject: %{text}<br>Percentile: %{z:.1f}<extra></extra>',
            colorscale=modified_colorscale,
            showscale=False,
            zmin=-0.1,
            zmax=100
        ))
        
        # Update layout to remove axes and improve appearance
        fig.update_layout(
            title="Subject Performance Heatmap",
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis=dict(
                visible=False,
                showgrid=False,
                zeroline=False
            ),
            yaxis=dict(
                visible=False,
                showgrid=False,
                zeroline=False,
                autorange="reversed"  # So [0,0] is in the top-left
            )
        )
        
        return fig