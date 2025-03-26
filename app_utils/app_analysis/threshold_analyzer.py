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
        
        # Add the actual feature values to the results for reference
        for feature in self.feature_thresholds.keys():
            if feature in self.session_data.columns:
                results[feature] = self.session_data[feature]
        
        # Apply threshold logic for each feature
        for feature, thresholds in self.feature_thresholds.items():
            # Get feature values
            feature_values = self.session_data[feature]
            
            # Get bounds
            lower_bound = thresholds.get('lower', None)
            upper_bound = thresholds.get('upper', None)
            
            # Create violation flags (True means threshold was violated)
            if lower_bound is not None:
                results[f"{feature}_below_lower"] = feature_values < lower_bound
            
            if upper_bound is not None:
                results[f"{feature}_above_upper"] = feature_values > upper_bound
            
            # Combined violation flag (either bound was violated)
            if lower_bound is not None and upper_bound is not None:
                results[f"{feature}_outside_range"] = (
                    (feature_values < lower_bound) | (feature_values > upper_bound)
                )
            
            # Also keep the compliance flags for compatibility (False means violation)
            if lower_bound is not None:
                results[f"{feature}_above_lower"] = feature_values >= lower_bound
            
            if upper_bound is not None:
                results[f"{feature}_below_upper"] = feature_values <= upper_bound
            
            if lower_bound is not None and upper_bound is not None:
                results[f"{feature}_within_range"] = (
                    (feature_values >= lower_bound) & (feature_values <= upper_bound)
                )
            
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
        
        # Get violation columns (all boolean columns that indicate violations)
        violation_cols = [col for col in threshold_data.columns 
                        if col not in ['subject_id', 'session_date'] and 
                        pd.api.types.is_bool_dtype(threshold_data[col]) and
                        any(x in col for x in ['below_lower', 'above_upper', 'outside_range'])]
        
        # Get compliance columns for backward compatibility
        compliance_cols = [col for col in threshold_data.columns 
                         if col not in ['subject_id', 'session_date'] and 
                         pd.api.types.is_bool_dtype(threshold_data[col]) and
                         any(x in col for x in ['above_lower', 'below_upper', 'within_range'])]
        
        # Initialize results storage
        summary_data = []
        
        # Analyze each subject
        for subject_id, subject_df in threshold_data.groupby('subject_id'):
            subject_summary = {'subject_id': subject_id}
            
            # Count total sessions
            subject_summary['total_sessions'] = len(subject_df)
            
            # Calculate statistics for violation columns
            for col in violation_cols:
                feature = col.split('_')[0]  # Extract feature name
                
                # Count and percentage of sessions where threshold was violated
                violation_count = subject_df[col].sum()
                violation_percent = (violation_count / len(subject_df)) * 100 if len(subject_df) > 0 else 0
                
                subject_summary[f"{col}_count"] = violation_count
                subject_summary[f"{col}_percent"] = violation_percent
                
                # Flag if any violations occurred
                subject_summary[f"{feature}_has_violations"] = violation_count > 0
                
                # First date violation occurred (if any)
                if violation_count > 0:
                    first_violation = subject_df[subject_df[col]]['session_date'].min()
                    subject_summary[f"{col}_first_date"] = first_violation
                else:
                    subject_summary[f"{col}_first_date"] = None
            
            # Calculate statistics for compliance columns (for backward compatibility)
            for col in compliance_cols:
                # Count and percentage of sessions that met the compliance criteria
                passed_count = subject_df[col].sum()
                passed_percent = (passed_count / len(subject_df)) * 100 if len(subject_df) > 0 else 0
                
                subject_summary[f"{col}_count"] = passed_count
                subject_summary[f"{col}_percent"] = passed_percent
            
            summary_data.append(subject_summary)
        
        # Convert to DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Add a total violations column
        if violation_cols:
            violation_count_cols = [col + "_count" for col in violation_cols]
            if all(col in summary_df.columns for col in violation_count_cols):
                summary_df['total_violations'] = summary_df[violation_count_cols].sum(axis=1)
        
        return summary_df
    
    def add_feature_to_session_data(self, feature_name: str, violation_suffix: str = "violation") -> pd.DataFrame:
        """
        Add threshold violation data back to the original session data for a specific feature
        
        Parameters:
            feature_name: str
                Name of the feature to add threshold data for
            violation_suffix: str
                Suffix to add to the feature name for the new column
        
        Returns:
            pd.DataFrame: Original session data with added threshold columns
        """
        # Create a copy of the original data
        enhanced_data = self.session_data.copy()
        
        # Get all threshold columns for this feature
        violation_cols = [col for col in self.threshold_results.columns 
                       if col.startswith(feature_name) and 
                       col not in ['subject_id', 'session_date'] and
                       any(x in col for x in ['below_lower', 'above_upper', 'outside_range'])]
        
        # Add violation columns to the enhanced data
        for col in violation_cols:
            enhanced_data[f"{col}_{violation_suffix}"] = self.threshold_results[col].values
        
        # Also add a combined violation flag
        if violation_cols:
            enhanced_data[f"{feature_name}_any_{violation_suffix}"] = self.threshold_results[violation_cols].any(axis=1)
        
        return enhanced_data
    
    def get_threshold_violations(self, 
                               subject_ids: Optional[List[str]] = None,
                               start_date: Optional[Union[str, datetime]] = None,
                               end_date: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Get threshold violation data, focusing only on sessions that violate thresholds
        
        Parameters:
            subject_ids: Optional[List[str]]
                List of subject IDs to include (if None, all subjects included)
            start_date: Optional[Union[str, datetime]]
                Start date for filtering (inclusive)
            end_date: Optional[Union[str, datetime]]
                End date for filtering (inclusive)
        
        Returns:
            pd.DataFrame: DataFrame with threshold violation results
        """
        # Get all threshold data
        all_data = self.get_threshold_crossings(
            subject_ids=subject_ids,
            start_date=start_date,
            end_date=end_date
        )
        
        # Get violation columns
        violation_cols = [col for col in all_data.columns 
                        if col not in ['subject_id', 'session_date'] and 
                        pd.api.types.is_bool_dtype(all_data[col]) and
                        any(x in col for x in ['below_lower', 'above_upper', 'outside_range'])]
        
        if not violation_cols:
            return pd.DataFrame()  # No violation columns found
        
        # Create a mask for sessions with any violations
        has_violation = all_data[violation_cols].any(axis=1)
        
        # Return only sessions with violations
        violations_df = all_data[has_violation].copy()
        
        # Add a column that indicates the total number of thresholds violated in each session
        violations_df['violation_count'] = violations_df[violation_cols].sum(axis=1)
        
        return violations_df
        
        