import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ThresholdAnalyzer:
    """
    Analyzer for identifying when specific features cross defined thresholds
    in subject session data.
    """
    
    def __init__(self, session_data: pd.DataFrame, feature_thresholds: Dict[str, Dict[str, float]]):
        """
        Initialize the ThresholdAnalyzer with raw session data and feature thresholds
        
        Parameters:
            session_data: pd.DataFrame
                DataFrame containing raw session data with at minimum:
                - subject_id: identifier for each subject
                - session_date: date of the session
                - feature columns: all features that may be analyzed
            
            feature_thresholds: Dict[str, Dict[str, float]]
                Dictionary mapping feature names to their threshold settings:
                {
                    'feature_name': {
                        'lower': 0.0,  # Optional, default 0
                        'upper': 10.0  # Optional
                    }
                }
        """
        self.session_data = session_data.copy()
        self.feature_thresholds = feature_thresholds
        
        # Validate session data has required columns
        required_cols = ["subject_id", "session_date"]
        missing_cols = [col for col in required_cols if col not in self.session_data.columns]
        if missing_cols:
            raise ValueError(f"session_data missing required columns: {missing_cols}")
        
        # Validate all features in thresholds exist in the data
        missing_features = [feat for feat in feature_thresholds if feat not in self.session_data.columns]
        if missing_features:
            raise ValueError(f"Features not found in session data: {missing_features}")
        
        # Ensure session_date is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(self.session_data['session_date']):
            try:
                self.session_data['session_date'] = pd.to_datetime(self.session_data['session_date'])
            except:
                raise ValueError("Could not convert session_date to datetime format")
        
        # Process threshold crossings
        self.threshold_results = self._process_thresholds()
    
    def _process_thresholds(self) -> pd.DataFrame:
        """
        Process the session data to determine threshold crossings
        
        Returns:
            pd.DataFrame: DataFrame with threshold crossing results
        """
        # Create a results dataframe with subject_id and session_date
        results = self.session_data[['subject_id', 'session_date']].copy()
        
        # Apply threshold logic for each feature
        for feature, thresholds in self.feature_thresholds.items():
            # Get feature values
            feature_values = self.session_data[feature]
            
            # Lower bound check (default to 0 if not specified)
            lower_bound = thresholds.get('lower', 0)
            if lower_bound is not None:
                results[f"{feature}_above_lower"] = feature_values >= lower_bound
            
            # Upper bound check (if specified)
            upper_bound = thresholds.get('upper', None)
            if upper_bound is not None:
                results[f"{feature}_below_upper"] = feature_values <= upper_bound
                
            # Combined check (if both bounds specified)
            if lower_bound is not None and upper_bound is not None:
                results[f"{feature}_within_range"] = (feature_values >= lower_bound) & (feature_values <= upper_bound)
                
        return results
    
    def get_threshold_crossings(self, 
                                subject_ids: Optional[List[str]] = None, 
                                start_date: Optional[Union[str, datetime]] = None,
                                end_date: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Get threshold crossing data, optionally filtered by subjects and date range
        
        Parameters:
            subject_ids: Optional[List[str]]
                List of subject IDs to include (if None, all subjects included)
            start_date: Optional[Union[str, datetime]]
                Start date for filtering (inclusive)
            end_date: Optional[Union[str, datetime]]
                End date for filtering (inclusive)
        
        Returns:
            pd.DataFrame: DataFrame with threshold crossing results
        """
        results = self.threshold_results.copy()
        
        # Filter by subject if specified
        if subject_ids is not None:
            results = results[results['subject_id'].isin(subject_ids)]
        
        # Filter by date range if specified
        if start_date is not None:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            results = results[results['session_date'] >= start_date]
            
        if end_date is not None:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            results = results[results['session_date'] <= end_date]
        
        return results
    
    def get_subject_crossing_summary(self, 
                                    subject_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get a summary of threshold crossings by subject
        
        Parameters:
            subject_ids: Optional[List[str]]
                List of subject IDs to include (if None, all subjects included)
        
        Returns:
            pd.DataFrame: DataFrame with summary statistics for each subject
        """
        # Get threshold data filtered by subjects if specified
        threshold_data = self.get_threshold_crossings(subject_ids=subject_ids)
        
        # Get feature crossing columns (all boolean columns)
        crossing_cols = [col for col in threshold_data.columns 
                        if col not in ['subject_id', 'session_date'] and 
                        pd.api.types.is_bool_dtype(threshold_data[col])]
        
        # Initialize results storage
        summary_data = []
        
        # Analyze each subject
        for subject_id, subject_df in threshold_data.groupby('subject_id'):
            subject_summary = {'subject_id': subject_id}
            
            # Count total sessions
            subject_summary['total_sessions'] = len(subject_df)
            
            # Calculate statistics for each threshold check
            for col in crossing_cols:
                # Count and percentage of sessions where threshold criteria was met
                passed_count = subject_df[col].sum()
                passed_percent = (passed_count / len(subject_df)) * 100 if len(subject_df) > 0 else 0
                
                subject_summary[f"{col}_count"] = passed_count
                subject_summary[f"{col}_percent"] = passed_percent
                
                # First date criteria met (if any)
                if passed_count > 0:
                    first_passed = subject_df[subject_df[col]]['session_date'].min()
                    subject_summary[f"{col}_first_date"] = first_passed
                else:
                    subject_summary[f"{col}_first_date"] = None
            
            summary_data.append(subject_summary)
        
        # Convert to DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        return summary_df
    
    def add_feature_to_session_data(self, feature_name: str, column_suffix: str = "crossed") -> pd.DataFrame:
        """
        Add threshold crossing data back to the original session data for a specific feature
        
        Parameters:
            feature_name: str
                Name of the feature to add threshold data for
            column_suffix: str
                Suffix to add to the feature name for the new column
        
        Returns:
            pd.DataFrame: Original session data with added threshold columns
        """
        # Create a copy of the original data
        enhanced_data = self.session_data.copy()
        
        # Get all threshold columns for this feature
        feature_cols = [col for col in self.threshold_results.columns 
                      if col.startswith(feature_name) and 
                      col not in ['subject_id', 'session_date']]
        
        # Add each column to the enhanced data
        for col in feature_cols:
            enhanced_data[f"{col}_{column_suffix}"] = self.threshold_results[col].values
            
        return enhanced_data
        
        