"""
Quantile analyzer for session-level percentile calculations and statistical analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from .overall_percentile_calculator import OverallPercentileCalculator
from .statistical_utils import StatisticalUtils
from app_utils.simple_logger import get_logger

logger = get_logger('quantile_analyzer')

class QuantileAnalyzer:
    """
    Analyzer for calculating and retrieving quantile-based metrics for subject performance
    ENHANCED IN PHASE 2: Now supports weighted percentile ranking for robust outlier handling
    Uses Wilson confidence intervals for statistical robustness
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
        self.statistical_utils = StatisticalUtils()
        self.calculate_percentiles()
        
    def calculate_percentiles(self):
        """
        Calculate percentile ranks for each subject within each stratified group
        using combined current and historical data for more robust distributions
        
        ENHANCED IN PHASE 2: Now supports weighted percentile ranking when outlier weights are available
        """
        # Calculate percentiles for strata (now contains both current and historical data)
        for strata, df in self.stratified_data.items():
            # Log strata size for debugging
            logger.info(f"Processing strata '{strata}' with {len(df)} subjects (current + historical)")
            
            # Skip strata with too few subjects (consider increasing minimum)
            if len(df) < 10:  # Minimum number for meaningful percentiles
                logger.info(f"  Skipping strata '{strata}' - too few subjects ({len(df)} < 10)")
                continue
                
            # Get processed feature columns
            feature_cols = [col for col in df.columns if col.endswith('_processed')]
            
            # Create a copy to store percentiles
            percentile_df = df.copy()
            
            # Check if we have outlier weights for weighted percentile calculation
            has_outlier_weights = 'outlier_weight' in df.columns
            
            if has_outlier_weights:
                logger.info(f"  Using weighted percentile ranking (outlier weights detected)")
                outlier_count = (df['outlier_weight'] < 1.0).sum()
                logger.info(f"    Found {outlier_count} sessions with outlier weights")
            
            # Calculate percentile for each feature - ranking all subjects together
            for feature in feature_cols:
                feature_values = df[feature].values
                
                if has_outlier_weights:
                    # PHASE 2: Use weighted percentile ranking
                    outlier_weights = df['outlier_weight'].values
                    
                    # Calculate weighted percentiles for each subject
                    percentiles = []
                    for i, target_value in enumerate(feature_values):
                        if pd.isna(target_value):
                            percentiles.append(np.nan)
                        else:
                            # Use weighted percentile ranking
                            percentile = self.statistical_utils.calculate_weighted_percentile_rank(
                                reference_values=feature_values,
                                reference_weights=outlier_weights,
                                target_value=target_value
                            )
                            percentiles.append(percentile)
                    
                    percentile_df[f"{feature.replace('_processed', '_percentile')}"] = percentiles
                    
                else:
                    # Traditional percentile ranking (for backward compatibility)
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
            logger.info(f"Subject {subject_id} not found in the data")
            return pd.DataFrame()
        
        # Sort by date to get chronological progression if date columns exist
        if 'first_date' in subject_data.columns:
            subject_data = subject_data.sort_values('first_date')
        
        return subject_data
    
    def calculate_session_level_percentiles(self, session_data: pd.DataFrame) -> pd.DataFrame:
        """ 
        Calculate session-level percentile ranks for each subject in each strata
        Enhanced with 95% confidence interval calculations and weighted percentile ranking
        
        ENHANCED IN PHASE 2: Now supports weighted percentile ranking when outlier weights are available
        """
        # Create a copy to avoid modifying the input
        result_df = session_data.copy()
        
        # Get all rolling average columns
        rolling_avg_cols = [col for col in result_df.columns if col.endswith('_rolling_avg')]
        
        if not rolling_avg_cols:
            logger.info("No rolling average columns found in data")
            return result_df
            
        # Track how many session-level percentiles we create
        created_columns = 0
        created_ci_columns = 0
        
        # Check if session data has outlier weights
        session_has_weights = 'outlier_weight' in session_data.columns
        if session_has_weights:
            logger.info("Session data contains outlier weights - will use weighted percentiles where available")
        
        # Process each strata separately to maintain consistent reference distributions
        for strata, strata_df in session_data.groupby('strata'):
            # Check if we have percentile data for this strata
            if strata not in self.percentile_data:
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
                    logger.info(f"No reference data found for {processed_col} in strata '{strata}'")
                    continue
                    
                # Get reference values for this feature
                reference_values = reference_df[processed_col].values
                
                # Get reference weights if available
                if 'outlier_weight' in reference_df.columns:
                    reference_weights = reference_df['outlier_weight'].values
                else:
                    reference_weights = np.ones(len(reference_values))  # Equal weights
                
                # Remove NaN values from reference for CI calculation
                valid_mask = ~np.isnan(reference_values)
                clean_reference_values = reference_values[valid_mask]
                clean_reference_weights = reference_weights[valid_mask]
                
                if len(clean_reference_values) < 3:
                    logger.info(f"Insufficient reference data for CI calculation in {processed_col}, strata '{strata}'")
                    continue
                
                # For each session in this strata
                for idx, row in strata_df.iterrows():
                    # Get rolling average value
                    rolling_value = row[rolling_col]
                    
                    if pd.isna(rolling_value):
                        continue
                    
                    if 'outlier_weight' in reference_df.columns:
                        # PHASE 2: Use weighted percentile ranking
                        percentile = self.statistical_utils.calculate_weighted_percentile_rank(
                            reference_values=clean_reference_values,
                            reference_weights=clean_reference_weights,
                            target_value=rolling_value
                        )
                    else:
                        # Traditional percentile calculation
                        temp_values = pd.Series(list(clean_reference_values) + [rolling_value])
                        temp_values = temp_values[~temp_values.isna()]  # Remove NaN values
                        
                        # Calculate percentile using rank method for consistency
                        ranks = temp_values.rank(pct=True)
                        percentile = ranks.iloc[-1] * 100  # Get percentile of the last value (our rolling value)
                    
                    # Calculate confidence interval using Wilson Score method ONLY
                    # Wilson Score CIs are appropriate for percentile ranking uncertainty
                    ci_lower, ci_upper = self.statistical_utils.calculate_percentile_confidence_interval(
                        clean_reference_values, 
                        percentile, 
                        confidence_level=0.95
                    )
                    
                    # Store in result dataframe with correct column name
                    # Extract clean feature name for percentile column
                    clean_feature_name = feature_name.replace('_processed', '')
                    result_df.loc[idx, f"{clean_feature_name}_session_percentile"] = percentile
                    result_df.loc[idx, f"{clean_feature_name}_session_percentile_ci_lower"] = ci_lower
                    result_df.loc[idx, f"{clean_feature_name}_session_percentile_ci_upper"] = ci_upper
                    
                    created_columns += 1
                    created_ci_columns += 2  # Lower and upper bounds
        
        logger.info(f"Created {created_columns} session-level percentile columns")
        logger.info(f"Created {created_ci_columns} confidence interval columns")
        if session_has_weights or any('outlier_weight' in self.percentile_data[strata].columns for strata in self.percentile_data):
            logger.info(f"Used weighted percentile ranking where outlier weights were available")
        
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
            logger.info("No session percentile columns found")
            return result_df
            
        # Process each session
        for idx, row in result_df.iterrows():
            # Extract percentile values for this session
            percentile_values = []
            feature_names = []
            
            for col in session_percentile_cols:
                if pd.notna(row[col]):
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