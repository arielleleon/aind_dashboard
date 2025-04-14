import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any

class ThresholdAnalyzer:
    """
    Analyzer for calculating threshold-based alerts for subject performance
    """
    
    def __init__(self, threshold_config: Dict[str, Any] = None):
        """
        Initialize the ThresholdAnalyzer with threshold configuration
        
        Parameters:
            threshold_config: Dict[str, Any]
                Dictionary of feature thresholds and conditions
                Format: {
                    'feature_name': {
                        'condition': 'gt' or 'lt' or 'eq',
                        'value': threshold_value,
                        'context': Optional context like stage filter
                    }
                }
        """
        self.threshold_config = threshold_config or {}
        
    def set_threshold_config(self, threshold_config: Dict[str, Any]):
        """
        Update threshold configuration
        
        Parameters:
            threshold_config: Dict[str, Any]
                Dictionary of feature thresholds and conditions
        """
        self.threshold_config = threshold_config
        
    def evaluate_condition(self, value: Any, condition: str, threshold: Any) -> bool:
        """
        Evaluate if a value meets a threshold condition
        
        Parameters:
            value: Any
                The value to evaluate
            condition: str
                The condition type ('gt', 'lt', 'eq', 'gte', 'lte')
            threshold: Any
                The threshold value to compare against
                
        Returns:
            bool: True if condition is met, False otherwise
        """
        if pd.isna(value):
            return False
            
        if condition == 'gt':
            return value > threshold
        elif condition == 'lt':
            return value < threshold
        elif condition == 'eq':
            return value == threshold
        elif condition == 'gte':
            return value >= threshold
        elif condition == 'lte':
            return value <= threshold
        else:
            raise ValueError(f"Unknown condition: {condition}")
            
    def analyze_thresholds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze dataframe for threshold violations and generate alerts
        
        Parameters:
            df: pd.DataFrame
                Input dataframe with session data
                
        Returns:
            pd.DataFrame
                DataFrame with threshold_alert column added
        """
        if not self.threshold_config:
            # No thresholds configured, return original dataframe with empty alerts
            df_result = df.copy()
            df_result['threshold_alert'] = 'N'
            return df_result
            
        # Create a copy of the dataframe to avoid modifying the original
        df_result = df.copy()
        
        # Add threshold_alert column with default value
        df_result['threshold_alert'] = 'N'  # Default to Normal
        
        # Process each threshold configuration
        for feature, config in self.threshold_config.items():
            # Skip if feature not in dataframe
            if feature not in df_result.columns:
                continue
                
            # Get condition and threshold value
            condition = config.get('condition', 'gt')
            threshold = config.get('value')
            
            if threshold is None:
                continue
                
            # Context filter if provided (e.g., specific stage)
            context = config.get('context')
            
            # Apply condition check for each row
            if context:
                # Get context column and value
                context_col = context.get('column')
                context_values = context.get('values', [])
                
                if context_col and context_col in df_result.columns and context_values:
                    # Only check rows matching the context
                    for idx, row in df_result.iterrows():
                        if row[context_col] in context_values:
                            # Check threshold
                            if self.evaluate_condition(row[feature], condition, threshold):
                                df_result.at[idx, 'threshold_alert'] = 'T'
            else:
                # Check all rows without context filtering
                for idx, row in df_result.iterrows():
                    if self.evaluate_condition(row[feature], condition, threshold):
                        df_result.at[idx, 'threshold_alert'] = 'T'
                        
        return df_result
    
    def get_stage_based_thresholds(self, stage_thresholds: Dict[str, int]) -> Dict[str, Any]:
        """
        Generate threshold configuration for sessions by stage
        
        Parameters:
            stage_thresholds: Dict[str, int]
                Dictionary mapping stage names to their session thresholds
                
        Returns:
            Dict[str, Any]: Properly formatted threshold config for analyze_thresholds
        """
        config = {}
        
        for stage, threshold in stage_thresholds.items():
            config[f"stage_{stage}_sessions"] = {
                'condition': 'gt',
                'value': threshold,
                'context': {
                    'column': 'current_stage_actual',
                    'values': [stage]
                }
            }
            
        return config
    
    def apply_standard_thresholds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply standard session thresholds based on stages as specified:
        - Total sessions: > 40 sessions -> T
        - Stage 1: > 5 sessions -> T
        - Stage 2: > 5 sessions -> T
        - Stage 3: > 6 sessions -> T
        - Stage 4/Final > 10 sessions -> T
        - GRADUATED: > 20 sessions -> T
        - Water day total: > 3.5 ml -> T
        
        Parameters:
            df: pd.DataFrame
                Input dataframe with session data
                
        Returns:
            pd.DataFrame: DataFrame with threshold_alert column added
        """
        # Define the standard thresholds
        stage_thresholds = {
            'STAGE_1': 5,
            'STAGE_2': 5,
            'STAGE_3': 6,
            'STAGE_4': 10,
            'STAGE_FINAL': 10,
            'GRADUATED': 20
        }
        
        # Create combined threshold config
        threshold_config = {
            'session': {
                'condition': 'gt',
                'value': 40  # Total sessions threshold
            },
            'water_day_total': {
                'condition': 'gt',
                'value': 3.5  # Water day total threshold (ml)
            }
        }
        
        # Process each row to check the stage and apply the appropriate threshold
        df_result = df.copy()
        df_result['threshold_alert'] = 'N'  # Default to Normal
        
        for idx, row in df_result.iterrows():
            # Check total sessions threshold
            if 'session' in row and not pd.isna(row['session']) and row['session'] > 40:
                df_result.at[idx, 'threshold_alert'] = 'T'
                continue
            
            # Check water_day_total threshold
            if 'water_day_total' in row and not pd.isna(row['water_day_total']) and row['water_day_total'] > 3.5:
                df_result.at[idx, 'threshold_alert'] = 'T'
                continue
                
            # Check stage-specific thresholds
            stage = row.get('current_stage_actual')
            if stage in stage_thresholds:
                threshold = stage_thresholds[stage]
                if 'session' in row and not pd.isna(row['session']) and row['session'] > threshold:
                    df_result.at[idx, 'threshold_alert'] = 'T'
                    
        return df_result
