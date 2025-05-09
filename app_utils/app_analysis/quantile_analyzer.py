import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns

class QuantileAnalyzer:
    """
    Analyzer for calculating and retrieving quantile-based metrics for subject performance
    """
    
    def __init__(self, stratified_data: Dict[str, pd.DataFrame], historical_data: Optional[pd.DataFrame] = None):
        """
        Initialize the QuantileAnalyzer with stratified subject data
        
        Parameters:
            stratified_data: Dict[str, pd.DataFrame]
                Dictionary of dataframes with subject averages, keyed by strata
            historical_data: Optional[pd.DataFrame]
                DataFrame containing historical strata data for all subjects
        """
        self.stratified_data = stratified_data
        self.historical_data = historical_data
        self.percentile_data = {}
        self.historical_percentile_data = None
        self.calculate_percentiles()
        
    def calculate_percentiles(self):
        """
        Calculate percentile ranks for each subject within each stratified group
        using combined current and historical data for more robust distributions
        """
        # Calculate percentiles for strata (now contains both current and historical data)
        for strata, df in self.stratified_data.items():
            # Log strata size for debugging
            print(f"Processing strata '{strata}' with {len(df)} subjects (current + historical)")
            
            # Skip strata with too few subjects (consider increasing minimum)
            if len(df) < 10:  # Minimum number for meaningful percentiles
                print(f"  Skipping strata '{strata}' - too few subjects ({len(df)} < 10)")
                continue
                
            # Get processed feature columns
            feature_cols = [col for col in df.columns if col.endswith('_processed')]
            
            # Create a copy to store percentiles
            percentile_df = df.copy()
            
            # Calculate percentile for each feature - ranking all subjects together
            for feature in feature_cols:
                percentile_df[f"{feature.replace('_processed', '_percentile')}"] = (
                    df[feature].rank(pct=True) * 100
                )
                
            # Store percentile data with an indicator of data source
            self.percentile_data[strata] = percentile_df
        
        # We don't need separate handling for historical data anymore since
        # it's already incorporated in the stratified_data from ReferenceProcessor

    def create_comprehensive_dataframe(self, include_history: bool = False) -> pd.DataFrame:
        """
        Create a comprehensive dataframe with all subjects, their strata, and feature percentile ranks
        
        Parameters:
            include_history: bool
                Whether to include historical strata data for subjects
        
        Returns:
            pd.DataFrame
                DataFrame containing subject_id, strata, and percentile ranks for all features
        """
        # Initialize an empty list to store data for each subject
        all_data = []
        
        # Process each strata for current data
        for strata, df in self.percentile_data.items():
            # For each subject in this strata
            for _, row in df.iterrows():
                # Get subject ID
                subject_id = row['subject_id']
                
                # Create a dictionary for this subject
                subject_data = {
                    'subject_id': subject_id,
                    'strata': strata,
                    'is_current': True
                }
                
                # Add all percentile values
                percentile_cols = [col for col in df.columns if col.endswith('_percentile')]
                for col in percentile_cols:
                    feature = col.replace('_percentile', '')
                    subject_data[f"{feature}_percentile"] = row[col]
                    
                    # Also add the processed feature value
                    processed_col = f"{feature}_processed"
                    if processed_col in df.columns:
                        subject_data[processed_col] = row[processed_col]
                
                # Add session count if available
                if 'session_count' in df.columns:
                    subject_data['session_count'] = row['session_count']
                
                # Add to our list
                all_data.append(subject_data)
        
        # Add historical data if requested and available
        if include_history and self.historical_percentile_data is not None:
            # For each historical subject-strata combination
            for _, row in self.historical_percentile_data.iterrows():
                # Skip if this is a current strata (already included above)
                if row.get('is_current', False):
                    continue
                
                # Get subject ID and strata
                subject_id = row['subject_id']
                strata = row['strata']
                
                # Create a dictionary for this historical entry
                historical_data = {
                    'subject_id': subject_id,
                    'strata': strata,
                    'is_current': False
                }
                
                # Add date information if available
                for date_col in ['first_date', 'last_date']:
                    if date_col in row:
                        historical_data[date_col] = row[date_col]
                
                # Add session count if available
                if 'session_count' in row:
                    historical_data['session_count'] = row['session_count']
                
                # Add all percentile values
                percentile_cols = [col for col in row.index if col.endswith('_percentile')]
                for col in percentile_cols:
                    feature = col.replace('_percentile', '')
                    historical_data[f"{feature}_percentile"] = row[col]
                    
                    # Also add the processed feature value
                    processed_col = f"{feature}_processed"
                    if processed_col in row:
                        historical_data[processed_col] = row[processed_col]
                
                # Add to our list
                all_data.append(historical_data)
        
        # Convert to DataFrame
        if all_data:
            return pd.DataFrame(all_data)
        else:
            # Return empty DataFrame with expected columns if no data
            return pd.DataFrame(columns=['subject_id', 'strata', 'is_current'])

    def get_subject_history(self, subject_id: str) -> pd.DataFrame:
        """
        Get the complete strata history for a specific subject with percentile ranks
        
        Parameters:
            subject_id: str
                The ID of the subject to retrieve history for
                
        Returns:
            pd.DataFrame
                DataFrame containing the subject's performance across all strata they've been in,
                ordered chronologically
        """
        # Create comprehensive dataframe with all subject-strata combinations
        all_data = self.create_comprehensive_dataframe(include_history=True)
        
        # Filter for the requested subject
        subject_data = all_data[all_data['subject_id'] == subject_id]
        
        if subject_data.empty:
            print(f"Subject {subject_id} not found in the data")
            return pd.DataFrame()
        
        # Sort by date to get chronological progression if date columns exist
        if 'first_date' in subject_data.columns:
            subject_data = subject_data.sort_values('first_date')
        
        return subject_data
    
    def calculate_overall_percentile(self, subject_ids: Optional[List[str]] = None,
                                     feature_weights: Optional[Dict[str, float]] = None,
                                     include_history: bool = False) -> pd.DataFrame:
        """
        Calculate overall percentile performance for subjects via a simple average across features

        Parameters:
            subject_ids: Optional[List[str]]
                List of subject IDs to calculate overall percentiles for
                If None: all subjects in dataframe will be used
            feature_weights: Optional[Dict[str, float]]
                Not used in this implementation as we're using a simple average
            include_history: bool
                Whether to include historical strata in averaging
        
        Returns:
            pd.DataFrame
                DataFrame containing overall percentile scores for each subject in each strata
        """
        # Get comprehensive dataframe with all subject data
        all_data = self.create_comprehensive_dataframe(include_history=include_history)

        # Filter for specific subject if provided
        if subject_ids is not None:
            all_data = all_data[all_data['subject_id'].isin(subject_ids)]

        if all_data.empty:
            print("No subjects found in the data")
            return pd.DataFrame()
        
        # Get all percentile columns
        percentile_cols = [col for col in all_data.columns if col.endswith('_percentile')]

        # Calculate simple average for each subject in each strata
        results = []

        # Group by subject and strata to handle historical data
        for (subject_id, strata), group in all_data.groupby(['subject_id', 'strata']):
            # Get first row for this subject-strata combination
            row = group.iloc[0]
            
            # Extract all percentile values that are not NaN
            percentile_values = []
            for col in percentile_cols:
                if col in row and not pd.isna(row[col]):
                    percentile_values.append(row[col])
            
            # Calculate simple average if we have any valid percentiles
            if percentile_values:
                overall_percentile = sum(percentile_values) / len(percentile_values)
            else:
                overall_percentile = np.nan

            # Create record
            result = {
                'subject_id': subject_id,
                'strata': strata,
                'overall_percentile': overall_percentile,
                'is_current': row.get('is_current', False)
            }

            # Add additional information if available
            for col in ['session_count', 'first_date', 'last_date']:
                if col in row:
                    result[col] = row[col]

            results.append(result)

        # Convert to DataFrame
        result_df = pd.DataFrame(results)

        # Sort by subject_id and is_current (current strata first)
        result_df = result_df.sort_values(['subject_id', 'is_current'], ascending=[True, False])

        return result_df
