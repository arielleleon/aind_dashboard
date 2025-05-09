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

    # Default minimum sessions for eligibility
    DEFAULT_MIN_SESSIONS = 1

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
        Calculate quantile-based alerts for subjects
        
        Parameters:
            subject_ids: Optional[List[str]]
                List of subject IDs to calculate alerts for
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping subject IDs to their quantile alerts
        """
        # Make sure we have the quantile analyzer
        if not self._validate_analyzer():
            return {}
        
        # Get analyzer reference from app utils
        analyzer = self.app_utils.quantile_analyzer
        
        # Calculate overall percentiles for all subjects using the simple average method
        overall_percentiles = analyzer.calculate_overall_percentile(subject_ids=subject_ids)
        
        # Get raw data to determine the most recent strata for each subject
        raw_data = None
        if hasattr(self.app_utils, 'get_session_data'):
            raw_data = self.app_utils.get_session_data()
        
        # Create a dictionary to map subjects to their latest strata
        subject_latest_strata = {}
        if raw_data is not None and not raw_data.empty:
            # Get the most recent session for each subject
            latest_sessions = raw_data.sort_values('session_date').groupby('subject_id').last().reset_index()
            
            # Map subject_id to their current strata, reconstructing it from session data
            for _, row in latest_sessions.iterrows():
                subject_id = row['subject_id']
                
                # Extract task and stage
                task = row.get('task', '')
                stage = row.get('current_stage_actual', '')
                
                # Determine version 
                version = row.get('curriculum_version', '')
                if "2.3" in version:
                    ver_group = "v3"
                elif "1.0" in version:
                    ver_group = "v1"
                else:
                    ver_group = "v2"
                
                # Simplify stage
                if 'STAGE_FINAL' in stage or 'GRADUATED' in stage:
                    simplified_stage = 'ADVANCED'
                elif any(s in stage for s in ['STAGE_4', 'STAGE_3']):
                    simplified_stage = 'INTERMEDIATE'
                elif any(s in stage for s in ['STAGE_2', 'STAGE_1', 'STAGE_1_WARMUP']):
                    simplified_stage = 'BEGINNER'
                else:
                    simplified_stage = 'UNKNOWN'
                
                # Create the strata string
                strata = f"{task}_{simplified_stage}_{ver_group}"
                
                # Add to the mapping
                subject_latest_strata[subject_id] = strata
        
        # Create a dictionary to store alerts by subject ID
        alerts = {}
        
        # Process each subject
        for subject_id, subject_group in overall_percentiles.groupby('subject_id'):
            # Get the actual latest strata if available, otherwise use logic from overall_percentiles
            if subject_id in subject_latest_strata:
                # Find the entry with matching strata
                latest_strata = subject_latest_strata[subject_id]
                matching_rows = subject_group[subject_group['strata'] == latest_strata]
                
                # Use matching row if found, otherwise fallback to date-based or other methods
                if not matching_rows.empty:
                    subject_data = matching_rows.iloc[0]
                elif 'last_date' in subject_group.columns:
                    subject_data = subject_group.sort_values('last_date', ascending=False).iloc[0]
                else:
                    subject_data = subject_group.iloc[0]
            elif 'last_date' in subject_group.columns:
                # Sort by date if available
                subject_data = subject_group.sort_values('last_date', ascending=False).iloc[0]
            else:
                # Fallback to first row if no better method
                subject_data = subject_group.iloc[0]
        
            # Extract overall percentile
            overall_percentile = subject_data.get('overall_percentile')
            
            # Skip subjects with no percentile (not scored)
            if pd.isna(overall_percentile):
                # Get reason for not scored
                ns_reason = self.get_not_scored_reason(subject_id)
                
                # Create alert with not scored status
                alerts[subject_id] = {
                    'subject_id': subject_id,
                    'overall_percentile': None,
                    'alert_category': 'NS',
                    'ns_reason': ns_reason,
                    'strata': subject_data.get('strata')
                }
                continue
            
            # Map percentile to category using configured thresholds
            alert_category = self.map_percentile_to_category(overall_percentile)
            
            # Create alert
            alerts[subject_id] = {
                'subject_id': subject_id,
                'overall_percentile': overall_percentile,
                'alert_category': alert_category,
                'strata': subject_data.get('strata')
            }
            
            # Add feature-specific percentiles if available
            feature_percentiles = {}
            for feature in self.config["feature_config"].keys():
                percentile_col = f"{feature}_percentile"
                
                # Check if feature percentile is in the data
                if percentile_col in subject_data:
                    feature_percentile = subject_data[percentile_col]
                    
                    # Map to category if percentile is valid
                    if not pd.isna(feature_percentile):
                        feature_category = self.map_percentile_to_category(feature_percentile)
                    else:
                        feature_category = 'NS'
                    
                    # Add to feature percentiles
                    feature_percentiles[feature] = {
                        'percentile': feature_percentile,
                        'category': feature_category
                    }
            
            # Add feature percentiles to alert
            if feature_percentiles:
                alerts[subject_id]['feature_percentiles'] = feature_percentiles
        
        # Store alerts for later retrieval
        self._quantile_alerts = alerts
        
        # Return alerts dictionary
        return alerts
    
    def get_quantile_alerts(self, subject_ids: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get precomputed quantile alerts for specified subjects
        
        Parameters:
            subject_ids: Optional[List[str]]
                List of subject IDs to get alerts for
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping subject IDs to their quantile alerts
        """
        # Validate that we have an analyzer
        if not self._validate_analyzer():
            return {}
        
        # Calculate alerts if not already calculated
        if not hasattr(self, '_quantile_alerts') or self._quantile_alerts is None:
            self.calculate_quantile_alerts()
        
        # Return all alerts if no subject IDs specified
        if subject_ids is None:
            return self._quantile_alerts
        
        # Otherwise, filter alerts for specified subjects
        result = {}
        for subject_id in subject_ids:
            if subject_id in self._quantile_alerts:
                result[subject_id] = self._quantile_alerts[subject_id]
        
        return result
    
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
        
    def get_not_scored_reason(self, subject_id: str) -> str:
        """
        Get the reason why a subject is not scored
        """
        # First check if subject has off-curriculum sessions
        if hasattr(self.app_utils, 'off_curriculum_subjects') and subject_id in self.app_utils.off_curriculum_subjects:
            info = self.app_utils.off_curriculum_subjects[subject_id]
            percent = (info['count'] / info['total_sessions']) * 100
            return f"Off-curriculum session: ({info['count']}, {percent:.0f}% of total)"
        
        # Rest of the checks...
        if not self._validate_analyzer():
            return "Analyzer not initialized"
        
        # Get min_sessions from config
        min_sessions = self.DEFAULT_MIN_SESSIONS
        if hasattr(self, 'config') and self.config and 'min_sessions' in self.config:
            min_sessions = self.config['min_sessions']
        
        # Check if subject exists in any strata
        analyzer = self.app_utils.quantile_analyzer
        for strata, strata_df in analyzer.stratified_data.items():
            if subject_id in strata_df['subject_id'].values:
                # Check session count
                row = strata_df[strata_df['subject_id'] == subject_id]
                if row['session_count'].values[0] < min_sessions:
                    return f"Insufficient sessions (< {min_sessions})"
                
                # Check overall percentiles
                overall_df = analyzer.calculate_overall_percentile()
                subject_row = overall_df[overall_df['subject_id'] == subject_id]
                if subject_row.empty or pd.isna(subject_row['overall_percentile'].values[0]):
                    return "No percentile data (strata may be too small)"
                
                return "Unknown reason"
        
        return "No eligible sessions in analysis window"
        
    def get_unified_alerts(self, subject_ids=None):
        """
        Get unified alert structure combining both quantile and threshold alerts
        
        Parameters:
            subject_ids (Optional[List[str]]): List of subject IDs to get alerts for
            
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping subject IDs to their unified alerts
        """
        # Validate analyzer requirements
        if not self._validate_analyzer():
            raise ValueError("Required analyzers not available. Initialize with AppUtils instance.")
        
        # Get quantile alerts
        quantile_alerts = self.get_quantile_alerts(subject_ids)
        
        # Get threshold alerts if threshold analyzer is available
        threshold_alerts = {}
        if hasattr(self.app_utils, 'threshold_analyzer') and self.app_utils.threshold_analyzer is not None:
            # Analyze most recent data with threshold analyzer
            if hasattr(self.app_utils, 'get_session_data'):
                df = self.app_utils.get_session_data()
                threshold_df = self.app_utils.threshold_analyzer.analyze_thresholds(df)
                
                # Get most recent session for each subject
                most_recent = threshold_df.sort_values('session_date').groupby('subject_id').last().reset_index()
                
                # Stage-specific session thresholds
                stage_thresholds = {
                    'STAGE_1': 5,
                    'STAGE_2': 5,
                    'STAGE_3': 6,
                    'STAGE_4': 10,
                    'STAGE_FINAL': 10,
                    'GRADUATED': 20
                }
                
                # Extract threshold alerts
                for _, row in most_recent.iterrows():
                    subject_id = row['subject_id']
                    current_stage = row.get('current_stage_actual', '')
                    session_count = row.get('session', 0)
                    water_day_total = row.get('water_day_total', 0)
                    
                    # Initialize threshold alerts structure
                    subject_threshold_alerts = {
                        'threshold_alert': 'N',  # Overall threshold alert (N/T)
                        'session_count': session_count,
                        'water_day_total': water_day_total,
                        'stage': current_stage,
                        'session_date': row.get('session_date'),
                        'specific_alerts': {
                            'total_sessions': {
                                'value': session_count,
                                'threshold': 40,
                                'alert': 'T' if session_count > 40 else 'N',
                                'description': f"Total sessions: {session_count} > 40" if session_count > 40 else ''
                            },
                            'water_day_total': {
                                'value': water_day_total,
                                'threshold': 3.5,
                                'alert': 'T' if water_day_total > 3.5 else 'N',
                                'description': f"Water day total: {water_day_total} > 3.5ml" if water_day_total > 3.5 else ''
                            }
                        }
                    }
                    
                    # Add stage-specific threshold
                    if current_stage in stage_thresholds:
                        stage_threshold = stage_thresholds[current_stage]
                        # Calculate number of sessions in this stage for this subject
                        if hasattr(self.app_utils, 'get_session_data'):
                            all_sessions = self.app_utils.get_session_data()
                            stage_sessions_count = len(all_sessions[(all_sessions['subject_id'] == subject_id) & (all_sessions['current_stage_actual'] == current_stage)])
                        else:
                            stage_sessions_count = 0
                        subject_threshold_alerts['specific_alerts']['stage_sessions'] = {
                            'value': stage_sessions_count,
                            'threshold': stage_threshold,
                            'alert': 'T' if stage_sessions_count > stage_threshold else 'N',
                            'description': f"{current_stage}: {stage_sessions_count} > {stage_threshold}" if stage_sessions_count > stage_threshold else '',
                            'stage': current_stage  # Add stage name to the alert data
                        }
                    
                    # Set overall threshold alert to 'T' if any specific alert is 'T'
                    if any(alert['alert'] == 'T' for alert in subject_threshold_alerts['specific_alerts'].values()):
                        subject_threshold_alerts['threshold_alert'] = 'T'
                    
                    threshold_alerts[subject_id] = subject_threshold_alerts
        
        # Combine alerts into unified structure
        unified_alerts = {}
        
        # Get all subjects to process
        all_subjects = set()
        if subject_ids is not None:
            all_subjects.update(subject_ids)
        else:
            # Include all subjects from quantile and threshold alerts
            all_subjects.update(quantile_alerts.keys())
            all_subjects.update(threshold_alerts.keys())
            
            # If no subject_ids specified, also include all subjects from session data
            if hasattr(self.app_utils, 'get_session_data'):
                df = self.app_utils.get_session_data()
                if df is not None and not df.empty:
                    all_subjects.update(df['subject_id'].unique())
        
        # Handle off-curriculum subjects first
        for subject_id in all_subjects:
            # Check if this is an off-curriculum subject
            if hasattr(self.app_utils, 'off_curriculum_subjects') and subject_id in self.app_utils.off_curriculum_subjects:
                # Create NS alert for this subject
                ns_reason = self.get_not_scored_reason(subject_id)
                unified_alerts[subject_id] = {
                    'alert_category': 'NS',
                    'overall_percentile': None,
                    'ns_reason': ns_reason,
                    'threshold': {
                        'threshold_alert': 'N',
                        'specific_alerts': {}
                    }
                }
        
        # Start with all subjects in quantile alerts, except off-curriculum ones
        off_curriculum_subjects = set()
        if hasattr(self.app_utils, 'off_curriculum_subjects'):
            off_curriculum_subjects = set(self.app_utils.off_curriculum_subjects.keys())
        
        for subject_id, alerts in quantile_alerts.items():
            # Skip off-curriculum subjects as they're already handled above
            if subject_id in off_curriculum_subjects:
                continue
            
            unified_alerts[subject_id] = {
                'quantile': alerts,
                'threshold': threshold_alerts.get(subject_id, {
                    'threshold_alert': 'N',
                    'specific_alerts': {}
                })
            }
        
        # Add any subjects that only have threshold alerts
        for subject_id, alerts in threshold_alerts.items():
            if subject_id not in unified_alerts:
                unified_alerts[subject_id] = {
                    'quantile': {'current': {}, 'historical': {}},
                    'threshold': alerts
                }
        
        # Add feature-specific percentiles and categories
        if hasattr(self.app_utils, 'quantile_analyzer') and self.app_utils.quantile_analyzer is not None:
            # Get comprehensive dataframe with all subject percentile data
            all_data = self.app_utils.quantile_analyzer.create_comprehensive_dataframe(include_history=False)
            
            if not all_data.empty:
                # Get all percentile columns
                percentile_cols = [col for col in all_data.columns if col.endswith('_percentile')]
                
                # Filter to specified subjects if provided
                if subject_ids is not None:
                    all_data = all_data[all_data['subject_id'].isin(subject_ids)]
                
                # Process each subject
                for subject_id in all_data['subject_id'].unique():
                    # Skip if subject not in unified alerts (shouldn't happen)
                    if subject_id not in unified_alerts:
                        continue
                        
                    # Get current strata data for this subject
                    subject_data = all_data[(all_data['subject_id'] == subject_id) & 
                                           (all_data['is_current'] == True)]
                    
                    if subject_data.empty:
                        continue
                    
                    # Get first row (most recent strata)
                    row = subject_data.iloc[0]
                    
                    # Add feature-specific percentiles and categories
                    feature_percentiles = {}
                    percentile_values = []
                    
                    for col in percentile_cols:
                        # Get feature name
                        feature = col.replace('_percentile', '')
                        
                        # Get percentile value
                        percentile = row[col] if col in row and not pd.isna(row[col]) else None
                        
                        # Skip features with no percentile data
                        if percentile is None:
                            continue
                            
                        # Store for overall percentile calculation
                        percentile_values.append(percentile)
                        
                        # Map to category
                        category = self.map_percentile_to_category(percentile)
                        
                        # Add to feature percentiles
                        feature_percentiles[feature] = {
                            'percentile': percentile,
                            'category': category,
                            'description': self.get_category_description(category)
                        }
                        
                        # Add processed value if available
                        processed_col = f"{feature}_processed"
                        if processed_col in row and not pd.isna(row[processed_col]):
                            feature_percentiles[feature]['processed_value'] = row[processed_col]
                    
                    # Add feature percentiles to unified alerts
                    unified_alerts[subject_id]['feature_percentiles'] = feature_percentiles
                    
                    # Calculate overall percentile directly as simple average of feature percentiles
                    if percentile_values:
                        calculated_overall = sum(percentile_values) / len(percentile_values)
                        # Add the calculated overall percentile
                        unified_alerts[subject_id]['calculated_overall_percentile'] = calculated_overall
                    
                    # Add strata information
                    if 'strata' in row:
                        unified_alerts[subject_id]['strata'] = row['strata']
        
        # Calculate overall percentiles for all subjects using the app_utils method 
        overall_percentiles = {}
        try:
            # Get overall percentiles for all subjects using app_utils method 
            overall_df = self.app_utils.calculate_overall_percentile(list(all_subjects))
            if not overall_df.empty:
                # Create mapping of subject_id to overall_percentile
                overall_percentiles = dict(
                    zip(overall_df['subject_id'], overall_df['overall_percentile'])
                )
        except Exception as e:
            print(f"Error calculating overall percentiles: {e}")
        
        # Add subjects that don't have alerts yet and get NS reasons
        subjects_without_alerts = all_subjects - set(unified_alerts.keys())
        for subject_id in subjects_without_alerts:
            # Get NS reason for this subject
            ns_reason = self.get_not_scored_reason(subject_id)
            
            # Add to unified alerts with NS category
            unified_alerts[subject_id] = {
                'quantile': {'current': {}, 'historical': {}},
                'threshold': {
                    'threshold_alert': 'N',
                    'specific_alerts': {}
                },
                'overall_percentile': None,
                'alert_category': 'NS',
                'ns_reason': ns_reason
            }
        
        # Add overall percentiles and categories to all subjects
        for subject_id in unified_alerts:
            # Use the directly calculated percentile if available, otherwise fall back to the app_utils one
            if 'calculated_overall_percentile' in unified_alerts[subject_id]:
                overall_percentile = unified_alerts[subject_id]['calculated_overall_percentile']
                # Clean up temporary variable
                unified_alerts[subject_id].pop('calculated_overall_percentile', None)
            else:
                # Fall back to app_utils calculation if direct calculation not available
                overall_percentile = overall_percentiles.get(subject_id)
            
            # Add to unified alerts
            unified_alerts[subject_id]['overall_percentile'] = overall_percentile
            
            # Calculate alert category from overall percentile
            if overall_percentile is not None and not np.isnan(overall_percentile):
                alert_category = self.map_overall_percentile_to_category(overall_percentile)
            else:
                alert_category = 'NS'
                # Add NS reason if not already present
                if 'ns_reason' not in unified_alerts[subject_id]:
                    unified_alerts[subject_id]['ns_reason'] = self.get_not_scored_reason(subject_id)
            
            unified_alerts[subject_id]['alert_category'] = alert_category
        
        return unified_alerts
