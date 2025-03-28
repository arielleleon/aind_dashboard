import plotly.graph_objects as go
import pandas as pd
import numpy as np
import math

class SubjectHeatmap:
    def __init__(self):
        """Initialize the subject heatmap component"""
        # Define the diverging color scale: dark orange (low) → grey (middle) → dark blue (high)
        self.colorscale = [
            [0.0, '#993404'],      # Dark orange for low percentiles
            [0.25, '#fe9929'],     # Medium orange
            [0.5, '#cccccc'],      # Grey for 50th percentile
            [0.75, '#4292c6'],     # Medium blue
            [1.0, '#084594']       # Dark blue for high percentiles
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
        
        # Count valid subjects (those with non-NS percentile category)
        valid_subjects = plot_data[plot_data['percentile_category'] != 'NS']
        n_subjects = len(valid_subjects)
        
        if n_subjects == 0:
            # Return empty figure if no valid subjects
            return go.Figure().update_layout(
                title="No subjects with percentile scores to display",
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=20, r=20, t=60, b=20)
            )
        
        # Calculate grid dimensions
        rows, cols = self._calculate_grid_dimensions(len(plot_data))
        
        # Create z-values, text labels, and mask for NS values
        z = np.zeros((rows, cols))  # Initialize with zeros
        text = np.empty((rows, cols), dtype=object)  # For subject IDs
        mask = np.zeros((rows, cols), dtype=bool)  # For NS values
        
        # Fill grid with subject data
        for idx, row in enumerate(range(rows)):
            for col in range(cols):
                pos = row * cols + col
                if pos < len(plot_data):
                    subject = plot_data.iloc[pos]
                    text[row, col] = subject['subject_id']
                    
                    if subject['percentile_category'] == 'NS':
                        # Mark as masked (will be white) for NS
                        mask[row, col] = True
                        z[row, col] = 0  # Value doesn't matter as it will be masked
                    else:
                        z[row, col] = subject['overall_percentile']
                else:
                    # Empty cells beyond our data
                    mask[row, col] = True
                    text[row, col] = ""
        
        # Create a custom colorscale with white for masked values
        colorscale = self.colorscale.copy()
        
        # Create heatmap figure
        fig = go.Figure(go.Heatmap(
            z=z,
            text=text,
            texttemplate="%{text}",
            hovertemplate='Subject: %{text}<br>Percentile: %{z:.1f}<extra></extra>',
            colorscale=colorscale,
            showscale=False,
            zmin=0,
            zmax=100
        ))
        
        # Apply mask for NS values by adding white rectangles
        for row in range(rows):
            for col in range(cols):
                if mask[row, col]:
                    fig.add_shape(
                        type="rect",
                        x0=col - 0.5,
                        y0=row - 0.5,
                        x1=col + 0.5,
                        y1=row + 0.5,
                        fillcolor="white",
                        line=dict(width=1, color="lightgrey"),
                        layer="above"
                    )
                    # Add the subject ID text on top for NS values
                    if text[row, col]:
                        fig.add_annotation(
                            x=col,
                            y=row,
                            text=text[row, col],
                            showarrow=False,
                            font=dict(size=10)
                        )
        
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