import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
from .overall_percentile_calculator import OverallPercentileCalculator

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
        self.percentile_calculator = OverallPercentileCalculator()
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
    
    def calculate_session_level_percentiles(self, session_data: pd.DataFrame) -> pd.DataFrame:
        """ 
        Calculate session-level percentile ranks for each subject in each strata

        Parameters:
            session_data: pd.DataFrame
                Dataframe containing session-level data with rolling averages
        
        Returns:
            pd.DataFrame
                DataFrame with calculated percentiles for each session
        """
        # Create a copy to avoid modifying the input
        result_df = session_data.copy()
        
        # Get all rolling average columns
        rolling_avg_cols = [col for col in result_df.columns if col.endswith('_rolling_avg')]
        
        if not rolling_avg_cols:
            print("No rolling average columns found in data")
            return result_df
            
        # Track how many session-level percentiles we create
        created_columns = 0
            
        # Process each strata separately to maintain consistent reference distributions
        for strata, strata_df in session_data.groupby('strata'):
            # Check if we have percentile data for this strata
            if strata not in self.percentile_data:
                print(f"No reference distribution found for strata '{strata}'")
                continue
                
            # Get reference distribution for this strata
            reference_df = self.percentile_data[strata]
            
            # For each rolling average feature, calculate percentile using reference distribution
            for rolling_col in rolling_avg_cols:
                # Extract the base feature name from the rolling_avg column
                feature_name = rolling_col.replace('_rolling_avg', '')
                
                # The feature name in reference data should already include '_processed'
                # so don't add it again
                processed_col = feature_name
                
                # Get reference column (processed feature values)
                if processed_col not in reference_df.columns:
                    print(f"No reference data found for {processed_col} in strata '{strata}'")
                    continue
                    
                # Get reference values for this feature
                reference_values = reference_df[processed_col].values
                
                # For each session in this strata
                for idx, row in strata_df.iterrows():
                    # Get rolling average value
                    rolling_value = row[rolling_col]
                    
                    if pd.isna(rolling_value):
                        continue
                    
                    temp_values = pd.Series(list(reference_values) + [rolling_value])
                    temp_values = temp_values[~temp_values.isna()]  # Remove NaN values
                    
                    # Calculate percentile using rank method for consistency
                    ranks = temp_values.rank(pct=True)
                    percentile = ranks.iloc[-1] * 100  # Get percentile of the last value (our rolling value)
                    
                    # Store in result dataframe with correct column name
                    # Extract clean feature name for percentile column
                    clean_feature_name = feature_name.replace('_processed', '')
                    result_df.loc[idx, f"{clean_feature_name}_session_percentile"] = percentile
                    created_columns += 1
        
        print(f"Created {created_columns} session-level percentile columns")
        return result_df
    
    def calculate_session_overall_percentile(self, session_data: pd.DataFrame, 
                                           feature_weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Calculate overall percentile for each session by taking the arithmetic mean of feature percentiles.
        
        Parameters:
            session_data: pd.DataFrame
                DataFrame with session-level percentiles
            feature_weights: Optional[Dict[str, float]]
                Optional weights for features
                
        Returns:
            pd.DataFrame
                DataFrame with calculated overall percentiles for each session
        """
        # Create a copy to avoid modifying the input
        result_df = session_data.copy()
        
        # Get all session percentile columns
        session_percentile_cols = [col for col in result_df.columns if col.endswith('_session_percentile')]
        
        if not session_percentile_cols:
            print("No session percentile columns found")
            return result_df
            
        # Process each session
        for idx, row in result_df.iterrows():
            # Extract percentile values for this session
            percentile_values = []
            feature_names = []
            
            for col in session_percentile_cols:
                if pd.notna(row[col]):
                    # FIX: Use clean feature name without _processed suffix
                    feature_name = col.replace('_session_percentile', '')
                    percentile_values.append(row[col])
                    feature_names.append(feature_name)
            
            # Skip if no percentile values
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
        
        return result_df
    
    def validate_session_strata_consistency(self, comprehensive_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that the final session in each strata has metrics that match the strata-level metrics.
        
        Parameters:
            comprehensive_df: pd.DataFrame
                Comprehensive DataFrame with session and strata-level metrics
        
        Returns:
            Dict[str, Any]: Validation results
        """
        # Get all subject-strata combinations
        subject_strata = comprehensive_df[['subject_id', 'strata']].drop_duplicates()
        
        # Initialize validation results
        validation = {
            'total_subjects': len(subject_strata),
            'matching_subjects': 0,
            'mismatches': [],
            'match_percentage': 0.0,
            'column_counts': {
                'session_percentile': len([col for col in comprehensive_df.columns if col.endswith('_session_percentile')]),
                'strata_percentile': len([col for col in comprehensive_df.columns if col.endswith('_strata_percentile')]),
                'rolling_avg': len([col for col in comprehensive_df.columns if col.endswith('_rolling_avg')]),
                'processed': len([col for col in comprehensive_df.columns if col.endswith('_processed')])
            }
        }
        
        # Print column information
        print(f"Column counts in validation: {validation['column_counts']}")
        
        # Check for overall percentile columns
        has_session_overall = 'session_overall_percentile' in comprehensive_df.columns
        has_strata_overall = 'strata_overall_percentile' in comprehensive_df.columns
        print(f"Has session_overall_percentile: {has_session_overall}")
        print(f"Has strata_overall_percentile: {has_strata_overall}")
        
        # If session_overall_percentile doesn't exist, early return
        if not has_session_overall:
            print("WARNING: session_overall_percentile column missing - can't validate consistency")
            return validation
        
        # Process each subject-strata combination
        for _, row in subject_strata.iterrows():
            subject_id = row['subject_id']
            strata = row['strata']
            
            # Get sessions for this subject-strata
            subject_sessions = comprehensive_df[(comprehensive_df['subject_id'] == subject_id) & 
                                              (comprehensive_df['strata'] == strata)]
            
            # Skip if no sessions
            if subject_sessions.empty:
                continue
            
            # Get the final session (most recent)
            if 'session_date' in subject_sessions.columns:
                final_session = subject_sessions.sort_values('session_date').iloc[-1]
            else:
                final_session = subject_sessions.iloc[-1]
            
            # Check if overall percentile values match
            session_overall = final_session.get('session_overall_percentile')
            strata_overall = final_session.get('strata_overall_percentile')
            
            # If both values exist and are close, consider them matching
            if pd.notna(session_overall) and pd.notna(strata_overall):
                # Check if values are within small tolerance
                if abs(session_overall - strata_overall) < 0.01:
                    validation['matching_subjects'] += 1
                else:
                    validation['mismatches'].append({
                        'subject_id': subject_id,
                        'strata': strata,
                        'session_overall': session_overall,
                        'strata_overall': strata_overall,
                        'difference': abs(session_overall - strata_overall)
                    })
        
        # Calculate match percentage
        if validation['total_subjects'] > 0:
            validation['match_percentage'] = (validation['matching_subjects'] / validation['total_subjects']) * 100
            
        # Print detailed mismatches (first 5)
        if validation['mismatches']:
            print(f"\nFirst 5 mismatches out of {len(validation['mismatches'])}:")
            for i, mismatch in enumerate(validation['mismatches'][:5]):
                print(f"{i+1}. Subject {mismatch['subject_id']} in {mismatch['strata']}: " +
                      f"session={mismatch['session_overall']:.2f}, strata={mismatch['strata_overall']:.2f}, " +
                      f"diff={mismatch['difference']:.2f}")
        
        return validation

    def debug_strata_reference_data(self, strata: str) -> None:
        """
        Debug helper to analyze reference data for a specific strata
        
        Parameters:
            strata: str
                Strata to debug
        """
        if strata not in self.percentile_data:
            print(f"No percentile data found for strata '{strata}'")
            return
            
        reference_df = self.percentile_data[strata]
        
        print(f"\nDebug information for strata '{strata}':")
        print(f"Number of subjects: {len(reference_df)}")
        
        # Check for processed feature columns
        processed_cols = [col for col in reference_df.columns if col.endswith('_processed')]
        print(f"Processed feature columns ({len(processed_cols)}): {processed_cols}")
        
        # Check for percentile columns
        percentile_cols = [col for col in reference_df.columns if col.endswith('_percentile')]
        print(f"Percentile columns ({len(percentile_cols)}): {percentile_cols}")
        
        # Show first few rows of reference data
        if not reference_df.empty:
            print("\nSample reference data (first 3 rows):")
            sample_cols = ['subject_id', 'strata']
            sample_cols.extend(processed_cols[:2] if len(processed_cols) > 2 else processed_cols)
            print(reference_df[sample_cols].head(3))
