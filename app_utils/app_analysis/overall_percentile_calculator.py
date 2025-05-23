import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any

class OverallPercentileCalculator:
    """
    Dedicated class for calculating overall percentiles across features.
    This centralizes percentile calculation logic used across the application.
    """
    
    def __init__(self):
        """Initialize the calculator with empty cache"""
        self._cache = {
            'overall_percentiles': None,
            'last_update_time': None,
            'session_overall_percentiles': None
        }
    
    def calculate_overall_percentile(self, 
                                    comprehensive_df: pd.DataFrame,
                                    subject_ids: Optional[List[str]] = None,
                                    feature_weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Calculate overall percentile performance for subjects via a simple average across features

        Parameters:
            comprehensive_df: pd.DataFrame
                DataFrame containing subject_id, strata, and percentile values for features
            subject_ids: Optional[List[str]]
                List of subject IDs to calculate overall percentiles for
                If None: all subjects in dataframe will be used
            feature_weights: Optional[Dict[str, float]]
                Optional dictionary mapping feature names to their weights
                If None: equal weights are used for all features
        
        Returns:
            pd.DataFrame
                DataFrame containing overall percentile scores for each subject in each strata
        """

        all_data = comprehensive_df.copy()

        # Filter for specific subject if provided
        if subject_ids is not None:
            all_data = all_data[all_data['subject_id'].isin(subject_ids)]

        if all_data.empty:
            print("No subjects found in the data")
            return pd.DataFrame()
        
        # Get all percentile columns
        percentile_cols = [col for col in all_data.columns if col.endswith('_percentile')]
        
        if not percentile_cols:
            print("No percentile columns found in data")
            return pd.DataFrame()

        # Calculate simple or weighted average for each subject in each strata
        results = []

        # Group by subject and strata to handle historical data
        for (subject_id, strata), group in all_data.groupby(['subject_id', 'strata']):
            # Get first row for this subject-strata combination
            row = group.iloc[0]
            
            # Extract all percentile values that are not NaN
            percentile_values = []
            feature_names = []
            
            for col in percentile_cols:
                if col in row and not pd.isna(row[col]):
                    feature_name = col.replace('_percentile', '')
                    feature_names.append(feature_name)
                    percentile_values.append(row[col])
            
            # Apply weights if provided
            if feature_weights is not None and percentile_values:
                weighted_values = []
                total_weight = 0
                
                for i, feature_name in enumerate(feature_names):
                    weight = feature_weights.get(feature_name, 1.0)
                    weighted_values.append(percentile_values[i] * weight)
                    total_weight += weight
                
                # Calculate weighted average
                overall_percentile = sum(weighted_values) / total_weight if total_weight > 0 else np.nan
            elif percentile_values:
                # Calculate simple average
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
        if not result_df.empty:
            result_df = result_df.sort_values(['subject_id', 'is_current'], ascending=[True, False])

        return result_df
    
    def calculate_session_overall_percentile(self,
                                           session_data: pd.DataFrame,
                                           feature_weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Calculate overall percentile performance for each session via a simple average across feature percentiles.

        Parameters:
            session_data: pd.DataFrame
                DataFrame containing session-level data with percentile values
            feature_weights: Optional[Dict[str, float]]
                Optional dictionary mapping feature names to their weights
                If None: equal weights are used for all features
        
        Returns:
            pd.DataFrame
                DataFrame with added session_overall_percentile column
        """
        # Create a copy to avoid modifying the input
        result_df = session_data.copy()
        
        # Get all session-level percentile columns
        percentile_cols = [col for col in result_df.columns if col.endswith('_session_percentile')]
        
        if not percentile_cols:
            print("No session-level percentile columns found in data")
            return result_df
        
        # Process each session
        for idx, row in result_df.iterrows():
            # Extract all percentile values that are not NaN
            percentile_values = []
            feature_names = []
            
            for col in percentile_cols:
                if not pd.isna(row[col]):
                    feature_name = col.replace('_session_percentile', '')
                    feature_names.append(feature_name)
                    percentile_values.append(row[col])
            
            # Skip if no valid percentile values
            if not percentile_values:
                continue
                
            # Apply weights if provided
            if feature_weights is not None:
                weighted_values = []
                total_weight = 0
                
                for i, feature_name in enumerate(feature_names):
                    weight = feature_weights.get(feature_name, 1.0)
                    weighted_values.append(percentile_values[i] * weight)
                    total_weight += weight
                
                # Calculate weighted average
                overall_percentile = sum(weighted_values) / total_weight if total_weight > 0 else np.nan
            else:
                # Calculate simple average
                overall_percentile = sum(percentile_values) / len(percentile_values)
            
            # Store in result dataframe
            result_df.loc[idx, 'session_overall_percentile'] = overall_percentile
        
        # Cache the result
        self._cache['session_overall_percentiles'] = result_df
        
        return result_df
    
    def set_cache(self, session_percentiles: pd.DataFrame) -> None:
        """
        Update the cache with unified session percentiles
        
        Parameters:
            session_percentiles: pd.DataFrame
                DataFrame containing session-level percentiles
        """
        self._cache['session_percentiles'] = session_percentiles
        self._cache['last_update_time'] = pd.Timestamp.now()
    
    def set_session_cache(self, session_overall_percentiles: pd.DataFrame) -> None:
        """
        Update the cache with pre-calculated session-level overall percentiles
        
        Parameters:
            session_overall_percentiles: pd.DataFrame
                DataFrame containing pre-calculated session-level overall percentiles
        """
        self._cache['session_overall_percentiles'] = session_overall_percentiles
        self._cache['last_update_time'] = pd.Timestamp.now()
    
    def get_cached_percentiles(self) -> Optional[pd.DataFrame]:
        """
        Get cached session percentiles if available
        
        Returns:
            Optional[pd.DataFrame]: Cached session percentiles or None if not available
        """
        return self._cache['session_percentiles']
    
    def get_cached_session_percentiles(self) -> Optional[pd.DataFrame]:
        """
        Get cached session-level overall percentiles if available
        
        Returns:
            Optional[pd.DataFrame]: Cached session-level overall percentiles or None if not available
        """
        return self._cache['session_overall_percentiles']
    
    def clear_cache(self) -> None:
        """Clear the cache"""
        self._cache['overall_percentiles'] = None
        self._cache['session_overall_percentiles'] = None
        self._cache['last_update_time'] = None 

    def calculate_unified_percentiles(self,
                               session_data: pd.DataFrame,
                               feature_weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Unified method to calculate percentiles across features for any granularity level.
        This replaces both calculate_overall_percentile and calculate_session_overall_percentile.

        Parameters:
            session_data: pd.DataFrame
                DataFrame containing data with percentile values
            feature_weights: Optional[Dict[str, float]]
                Optional dictionary mapping feature names to their weights
                If None: equal weights are used for all features
        
        Returns:
            pd.DataFrame
                DataFrame with overall percentile scores added
        """
        result_df = session_data.copy()
        
        # Get all percentile columns
        percentile_cols = [col for col in result_df.columns if col.endswith('_percentile')]
        
        if not percentile_cols:
            print("No percentile columns found in data")
            return result_df
        
        # Process each row
        for idx, row in result_df.iterrows():
            # Extract all percentile values that are not NaN
            percentile_values = []
            feature_names = []
            
            for col in percentile_cols:
                if not pd.isna(row[col]):
                    feature_name = col.replace('_percentile', '')
                    feature_names.append(feature_name)
                    percentile_values.append(row[col])
            
            # Skip if no percentile values
            if not percentile_values:
                result_df.loc[idx, 'overall_percentile'] = np.nan
                continue
            
            # Apply weights if provided
            if feature_weights is not None and percentile_values:
                weighted_values = []
                total_weight = 0
                
                for i, feature_name in enumerate(feature_names):
                    weight = feature_weights.get(feature_name, 1.0)
                    weighted_values.append(percentile_values[i] * weight)
                    total_weight += weight
                
                # Calculate weighted average
                overall_percentile = sum(weighted_values) / total_weight if total_weight > 0 else np.nan
            elif percentile_values:
                # Calculate simple average
                overall_percentile = sum(percentile_values) / len(percentile_values)
            else:
                overall_percentile = np.nan
            
            # Store the overall percentile
            result_df.loc[idx, 'overall_percentile'] = overall_percentile
        
        return result_df
    
    def calculate_session_overall_rolling_average(self,
                                                session_data: pd.DataFrame,
                                                feature_weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Calculate overall rolling average for each session by averaging the feature rolling averages.
        This provides the "rolling average value" for overall percentile hover information.

        Parameters:
            session_data: pd.DataFrame
                DataFrame containing session-level data with rolling average values
            feature_weights: Optional[Dict[str, float]]
                Optional dictionary mapping feature names to their weights
                If None: equal weights are used for all features
        
        Returns:
            pd.DataFrame
                DataFrame with added session_overall_rolling_avg column
        """
        # Create a copy to avoid modifying the input
        result_df = session_data.copy()
        
        # Get all session-level rolling average columns
        rolling_avg_cols = [col for col in result_df.columns if col.endswith('_processed_rolling_avg')]
        
        if not rolling_avg_cols:
            print("No session-level rolling average columns found in data")
            return result_df
        
        # Process each session
        for idx, row in result_df.iterrows():
            # Extract all rolling average values that are not NaN
            rolling_avg_values = []
            feature_names = []
            
            for col in rolling_avg_cols:
                if not pd.isna(row[col]):
                    # Extract feature name from column (remove _processed_rolling_avg suffix)
                    feature_name = col.replace('_processed_rolling_avg', '')
                    feature_names.append(feature_name)
                    rolling_avg_values.append(row[col])
            
            # Skip if no valid rolling average values
            if not rolling_avg_values:
                result_df.loc[idx, 'session_overall_rolling_avg'] = np.nan
                continue
                
            # Apply weights if provided
            if feature_weights is not None:
                weighted_values = []
                total_weight = 0
                
                for i, feature_name in enumerate(feature_names):
                    weight = feature_weights.get(feature_name, 1.0)
                    weighted_values.append(rolling_avg_values[i] * weight)
                    total_weight += weight
                
                # Calculate weighted average
                overall_rolling_avg = sum(weighted_values) / total_weight if total_weight > 0 else np.nan
            else:
                # Calculate simple average
                overall_rolling_avg = sum(rolling_avg_values) / len(rolling_avg_values)
            
            # Store in result dataframe
            result_df.loc[idx, 'session_overall_rolling_avg'] = overall_rolling_avg
        
        return result_df