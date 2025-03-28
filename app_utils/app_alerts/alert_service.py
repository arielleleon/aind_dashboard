import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple

class AlertService:
    """
    Service for getting / setting alerts based on percentile rankings
    """

    # Default percentile category boundaries
    DEFAULT_PERCENTILE_CATEGORIES = {
            "SB": 6.5,  # Significantly Bad: < 6.5% ( < -2.75 std dev)
            "B": 28,    # Bad: < 28% ( < -0.25 std dev)
            "N": 72,    # Normal: 28% - 72% ( -0.25 std dev to +0.25 std dev)
            "G": 93.5,  # Good: 72% - 93.5% ( > +0.25 std dev)
            "SG": 100   # Significantly Good: > 93.5% ( > +2.75 std dev)
        }

    def __init__(self, app_utils=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AlertService

        Parameters:
            app_utils (AppUtils): The AppUtils instance (optional)
            config (Dict[str, Any]): Optional configuration for the AlertService
        """
        self.app_utils = app_utils

        # Initialize configuation with default
        self.config = {
            'percentile_categories': self.DEFAULT_PERCENTILE_CATEGORIES.copy(),
            "feature_config": {} # Feature specific configuration
        }

        # Override defaults if provided with config
        if config:
            self._update_config(config)

        # Initialize alert caches
        self._quantile_alerts = {}
        self._last_update_time = None

    def _update_config(self, config: Dict[str, Any]) -> None:
        """
        Update the configuration with new values

        Parameters:
            config (Dict[str, Any]): The new configuration
        """
        if "percentile_categories" in config:
            # Merge with defaults
            self.config["percentile_categories"].update(config["percentile_categories"])

        if "feature_config" in config:
            self.config["feature_config"].update(config["feature_config"])

    def set_app_utils(self, app_utils) -> None:
        """
        Set the AppUtils instance

        Parameters:
            app_utils (AppUtils): The AppUtils instance
        """
        self.app_utils = app_utils

    def map_percentile_to_category(self, percentile: float) -> str:
        """
        Map a percentile value to its corresponding category

        Parameters: 
            percentile (float): The percentile value to map

        Returns:
            str: Category abbreviation (SB, B, N, G, SG)
        """
        if percentile is None or np.isnan(percentile):
            return "Unknown"
        
        categories = self.config["percentile_categories"]

        if percentile < categories["SB"]:
            return "SB"
        elif percentile < categories["B"]:
            return "B"
        elif percentile < categories["N"]:
            return "N"
        elif percentile < categories["G"]:
            return "G"
        else:
            return "SG"
        
    def get_category_description(self, category: str) -> str:
        """
        Get the description for a given category

        Parameters:
            category (str): The category abbreviation (SB, B, N, G, SG)

        Returns:
            str: Description of the category
        """
        descriptions = {
            "SB": "Significantly Below Average",
            "B": "Below Average",
            "N": "Average",
            "G": "Above Average",
            "SG": "Significantly Above Average"
        }
        return descriptions.get(category, "Unknown")
    
    def _validate_analyzer(self) -> bool:
        """
        Validate that required analyzers are available

        Returns: 
            bool: True if all required analyzers are available
        """
        if self.app_utils is None:
            return False
        
        # Check quantile analyzer
        has_quantile = hasattr(self.app_utils, 'quantile_analyzer') and self.app_utils.quantile_analyzer is not None

        return has_quantile

    def calculate_quantile_alerts(self, subject_ids: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Calculate quantile alerts for given subjects based on percentile rankings

        Parameters:
            subject_ids (Optional[List[str]]): List of subject IDs to calculate alerts for

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping subject IDs to their alerts
        """
        # Validate analyzer
        if not self._validate_analyzer():
            raise ValueError("QuantileAnalyzer not available. Initialize with AppUtils instance.")
        
        # Get comprehensive dataframe with all subject percentile rankings
        all_data = self.app_utils.quantile_analyzer.create_comprehensive_dataframe(include_history = True)

        # If empty, return empty results
        if all_data.empty: 
            return {}
        
        # Filter to specified subjects if provided
        if subject_ids is not None:
            all_data = all_data[all_data['subject_id'].isin(subject_ids)]

        # Initialize alerts
        alerts = {}

        # Get all percentile columns
        percentile_cols = [col for col in all_data.columns if col.endswith('_percentile')]

        # Process each subject
        for subject_id in all_data['subject_id'].unique():
            # Get all data for this subject
            subject_data = all_data[all_data['subject_id'] == subject_id]

            # Get the current strata row(s)
            current_strata = subject_data[subject_data['is_current'] == True]

            # Skip if no current strata
            if current_strata.empty:
                continue

            # Get historical strata
            historical_strata = subject_data[subject_data['is_current'] == False]

            # Initialize subject alerts
            subject_alerts = {
                'current': {},
                'historical': {}
            }

            # Process current strata percentiles
            for _, row in current_strata.iterrows():
                strata = row['strata']
                strata_alerts = {}

                # Process each percentile column
                for col in percentile_cols:
                    # Extract feature names
                    feature = col.replace('_percentile', '')

                    # Get percentile and map to category
                    percentile = row[col]
                    if not pd.isna(percentile):
                        category = self.map_percentile_to_category(percentile)

                        # Create alert entry with details
                        strata_alerts[feature] = {
                            'percentile': percentile,
                            'category': category,
                            'description': self.get_category_description(category),
                            'strata': strata
                        }

                        # Add processed value if available
                        processed_col = f"{feature}_processed"
                        if processed_col in row and not pd.isna(row[processed_col]):
                            strata_alerts[feature]['processed_value'] = row[processed_col]

                # Add strata alerts to subject's current alerts
                subject_alerts['current'][strata] = strata_alerts

            # Process historical strata in chronological order
            if not historical_strata.empty and 'first_date' in historical_strata.columns:
                # Sort by date
                historical_strata = historical_strata.sort_values('first_date')

                # Process each historical strata
                for _, row in historical_strata.iterrows():
                    strata = row['strata']
                    strata_alerts = {}

                    # Process each percentile column
                    for col in percentile_cols:
                        # Extract feature names
                        feature = col.replace('_percentile', '')

                        # Get percentile and map to category
                        percentile = row[col]
                        if not pd.isna(percentile):
                            category = self.map_percentile_to_category(percentile)

                            # Create alert entry with details
                            strata_alerts[feature] = {
                                'percentile': percentile,
                                'category': category,
                                'description': self.get_category_description(category),
                                'strata': strata,
                                'first_date': row.get('first_date'),
                                'last_date': row.get('last_date')
                            }

                            # Add processed value if available
                            processed_col = f"{feature}_processed"
                            if processed_col in row and not pd.isna(row[processed_col]):
                                strata_alerts[feature]['processed_value'] = row[processed_col]

                    # Add strata alerts to subject's historical alerts
                    if strata_alerts:
                        subject_alerts['historical'][strata] = strata_alerts

            # Only add subjects if there are alerts
            if subject_alerts['current'] or subject_alerts['historical']:
                alerts[subject_id] = subject_alerts

        # Update the cache
        self._quantile_alerts = alerts
        self._last_update_time = pd.Timestamp.now()

        # Make sure to return the alerts dictionary
        return alerts
    
    def get_quantile_alerts(self, subject_ids: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get quantile alerts for given subjects

        Parameters:
            subject_ids (Optional[List[str]]): List of subject IDs to get alerts for

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping subject IDs to their alerts
        """
        # Calculate alerts if not already calculated
        if not self._quantile_alerts or subject_ids is None:
            self.calculate_quantile_alerts(subject_ids)

        # If subject_ids specified, filter to those subjects
        if subject_ids is not None:
            return {sid: alerts for sid, alerts in self._quantile_alerts.items() if sid in subject_ids}
        
        return self._quantile_alerts
    
    def get_subjects_with_quantile_alerts(self, features: Optional[List[str]] = None, 
                                        categories: Optional[List[str]] = None) -> List[str]:
        """
        Get a list of subjects that have quantile alerts matching specified criteria
        
        Parameters:
            features: Optional list of features to check for alerts
                    If None, checks for alerts on any feature
            categories: Optional list of categories to filter by (SB, B, N, G, SG)
                    If None, includes all categories
        
        Returns:
            List of subject IDs with matching quantile alerts
        """
        # Ensure alerts are calculated
        if not self._quantile_alerts:
            self.calculate_quantile_alerts()
        
        # Initialize results
        subjects_with_alerts = []
        
        # Process each subject
        for subject_id, subject_alerts in self._quantile_alerts.items():
            # Check only current strata
            current_strata_alerts = subject_alerts.get('current', {})
            
            # Flag to track if this subject matches criteria
            subject_matches = False
            
            # Check each strata in current alerts
            for strata, strata_alerts in current_strata_alerts.items():
                # Skip if no matching strata alerts
                if not strata_alerts:
                    continue
                
                # Process each feature alert
                for feature, alert in strata_alerts.items():
                    # Skip if filtering by features and this one isn't included
                    if features is not None and feature not in features:
                        continue
                    
                    # Skip if filtering by categories and this one isn't included
                    if categories is not None and alert['category'] not in categories:
                        continue
                    
                    # If we get here, we have a match
                    subject_matches = True
                    break
                
                # Break strata loop if we found a match
                if subject_matches:
                    break
            
            # Add subject if it matches criteria
            if subject_matches:
                subjects_with_alerts.append(subject_id)
        
        return subjects_with_alerts

    def get_quantile_alert_summary(self, subject_id: str, current_only: bool = True) -> str:
        """
        Get a summary of quantile alerts for a given subject

        Parameters:
            subject_id (str): The subject ID to get alerts for
            current_only (bool): Whether to only include current strata alerts

        Returns:
            str: Summary of quantile alerts
        """
        # Ensure alerts are calculated
        if not self._quantile_alerts:
            self.calculate_quantile_alerts()

        # Check if subject has alerts
        if subject_id not in self._quantile_alerts:
            return "No quantile alerts"
        
        # Get subject's alerts
        subject_alerts = self._quantile_alerts[subject_id]

        # Current strata first
        current_strata_alerts = subject_alerts.get('current', {})
        if not current_strata_alerts:
            return "No current strata alerts"
        
        # Count alerts by category
        category_counts = {'SB': 0, 'B': 0, 'N': 0, 'G': 0, 'SG': 0}

        # Track features by category
        features_by_category = {'SB': [], 'B': [], 'N': [], 'G': [], 'SG': []}

        # Process current strata
        for strata, strata_alerts in current_strata_alerts.items():
            for feature, alert in strata_alerts.items():
                category = alert['category']
                category_counts[category] += 1
                features_by_category[category].append(feature)

        # Build summary
        summary_parts = []

        # Report non-normal categories first
        for category in ['SB', 'B', 'G', 'SG']:
            count = category_counts[category]
            if count > 0:
                features = features_by_category[category]
                category_text = self.get_category_description(category)
                summary_parts.append(f"{count} {category_text}: {', '.join(features)}")

        # Only include normal if it's the only category or we have fewwer features in other categories
        if category_counts['N'] > 0 and (len(summary_parts) == 0 or category_counts['N'] > 3):
            normal_count = category_counts['N']
            summary_parts.append(f"{normal_count} Average features")

        # Add current strata information
        strata_names = list(current_strata_alerts.keys())
        if len(strata_names) == 1:
            summary_parts.append(f"in strata: {strata_names[0]}")

        # Combine all parts
        if summary_parts:
            return " | ".join(summary_parts)
        else:
            return "No notable quantile alerts"
        
    def get_alert_counts(self) -> Dict[str, Any]:
        """
        Get counts of alerts across all subjects

        Returns:
            Dict[str, Any]: Dictionary with alert counts and statistics
        """
        # Get all alerts
        quantile_alerts = self.get_quantile_alerts()

        # Calculate counts
        results = {
            'quantile': {
                'subjects_with_alerts': len(quantile_alerts),
                'category_counts': {
                    'SB': 0,
                    'B': 0,
                    'N': 0,
                    'G': 0,
                    'SG': 0
                }
            }
        }
        
        # Count quantile categories
        for subject_id, subject_alerts in quantile_alerts.items():
            current_strata = subject_alerts.get('current', {})

            for strata, features in current_strata.items():
                for feature, details in features.items():
                    category = details['category']
                    if category in results['quantile']['category_counts']:
                        results['quantile']['category_counts'][category] += 1
        
        return results
    
    def map_overall_percentile_to_category(self, overall_percentile):
        """
        Map an overall percentile to its corresponding category

        Parameters:
            overall_percentile (float): The overall percentile value to map

        Returns:
            str: Category abbreviation (SB, B, N, G, SG)
        """
        if overall_percentile is None or np.isnan(overall_percentile):
            return "NS" # Not scored
        
        categories = self.config["percentile_categories"]

        if overall_percentile < categories["SB"]:
            return "SB"
        elif overall_percentile < categories["B"]:
            return "B"
        elif overall_percentile < categories["N"]:
            return "N"
        elif overall_percentile < categories["G"]:
            return "G"
        else:
            return "SG"