from .app_data_load import AppLoadData
from .app_analysis import ReferenceProcessor, QuantileAnalyzer, ThresholdAnalyzer
from .app_alerts import AlertService
from typing import Dict, List, Optional, Union, Any
import pandas as pd

class AppUtils:
    """
    Central utility class for app
    Access to data loading and computation functions
    """

    def __init__(self):
        """
        Initialize app utils
        """
        self.data_loader = AppLoadData()
        self.reference_processor = None
        self.quantile_analyzer = None
        self.threshold_analyzer = None
        self.alert_service = None
        
    def get_session_data(self, load_bpod = False):
        """
        Get session data from the data loader
        """
        if load_bpod:
            return self.data_loader.load(load_bpod = True)
        return self.data_loader.get_data()
    
    def reload_data(self, load_bpod = False):
        """
        Force reload session data
        """
        return self.data_loader.load(load_bpod = load_bpod)
    
    def initialize_reference_processor(self, features_config, window_days = 49, min_sessions = 5, min_days = 7):
        """
        Initialize reference processor

        Parameters:
            features_config (Dict[str, bool]): Configuration of features (feature_name: higher or lower better)
            window_days (int): Number of days to include in sliding window
            min_sessions (int): Minimum number of sessions required for eligibility
            min_days (int): Minimum number of days required for eligibility

        Returns: 
            ReferenceProcessor: Initialized reference processor
        """
        self.reference_processor = ReferenceProcessor(
            features_config = features_config,
            window_days = window_days,
            min_sessions = min_sessions,
            min_days = min_days
        )
        return self.reference_processor
    
    def process_reference_data(self, df, reference_date = None, remove_outliers = False):
        """
        Process data through reference processor pipeline

        Parameters:
            df (pd.DataFrame): Input dataframe with raw session data
            reference_date (datetime, optional): Reference date for sliding window
            remove_outliers (bool, optional): Whether to remove outliers

        Returns:
            Dict[str, pd.DataFrame]: Dictionary of stratified data for quantile analysis
        """
        if self.reference_processor is None:
            raise ValueError("Reference processor not initialized. Call initialize_reference_processor first.")
        
        # Apply sliding window
        window_df = self.reference_processor.apply_sliding_window(df, reference_date)

        # Get eligible subjects
        eligible_subjects = self.reference_processor.get_eligible_subjects(window_df)
        eligible_df = window_df[window_df['subject_id'].isin(eligible_subjects)]

        # Preprocess data
        processed_df = self.reference_processor.preprocess_data(eligible_df, remove_outliers)

        # Prepare for quantile analysis
        stratified_data = self.reference_processor.prepare_for_quantile_analysis(processed_df)

        # Store historical data for later use
        self.historical_data = self.reference_processor.subject_history

        return stratified_data
    
    def initialize_quantile_analyzer(self, stratified_data):
        """
        Initialize quantile analyzer

        Parameters:
            stratified_data (Dict[str, pd.DataFrame]): Dictionary of stratified data

        Returns:
            QuantileAnalyzer: Initialized quantile analyzer
        """
        self.quantile_analyzer = QuantileAnalyzer(
            stratified_data = stratified_data,
            historical_data = getattr(self, 'historical_data', None)
        )
        return self.quantile_analyzer
    
    def get_subject_percentiles(self, subject_id):
        """
        Get percentile data for specific subject

        Parameters:
            subject_id (str): subject ID to get percentile data for

        Returns:
            pd.DataFrame: Dataframe with percentile data for the subject
        """
        if self.quantile_analyzer is None:
            raise ValueError("Quantile analyzer not initialized. Process data first.")
        
        return self.quantile_analyzer.get_subject_history(subject_id)
    
    def calculate_overall_percentile(self, subject_ids = None, feature_weights = None):
        """
        Calculate overall percentile scores for subjects

        Parameters:
            subject_ids (List[str], optional): List of specific subjects to calculate for
            feature_weights (Dict[str, float], optional): Weights for different features in calculation

        Returns:
            pd.DataFrame: Dataframe with overall percentile scores
        """
        if self.quantile_analyzer is None:
            raise ValueError("Quantile analyzer not initialized. Process data first.")
        
        return self.quantile_analyzer.calculate_overall_percentile(
            subject_ids = subject_ids,
            feature_weights = feature_weights
        )
    
    def initialize_threshold_analyzer(self, feature_thresholds):
        """
        Initialize threshold analyzer

        Parameters:
            feature_thresholds (Dict[str, Dict[str, float]]): Dictionary mapping feature names to threshold settings
                Example: {
                    'feature_name': {
                        'lower': 0.0,  # Optional, default 0
                        'upper': 10.0  # Optional
                    }
                }

        Returns:
            ThresholdAnalyzer: Initialized threshold analyzer
        """
        session_data = self.get_session_data()
        self.threshold_analyzer = ThresholdAnalyzer(
            session_data=session_data,
            feature_thresholds=feature_thresholds
        )
        return self.threshold_analyzer
    
    def get_threshold_crossings(self, subject_ids=None, start_date=None, end_date=None):
        """
        Get threshold crossing data, optionally filtered by subjects and date range

        Parameters:
            subject_ids (List[str], optional): List of subject IDs to include
            start_date (str or datetime, optional): Start date for filtering (inclusive)
            end_date (str or datetime, optional): End date for filtering (inclusive)

        Returns:
            pd.DataFrame: DataFrame with threshold crossing results
        """
        if self.threshold_analyzer is None:
            raise ValueError("Threshold analyzer not initialized. Call initialize_threshold_analyzer first.")
        
        return self.threshold_analyzer.get_threshold_crossings(
            subject_ids=subject_ids,
            start_date=start_date,
            end_date=end_date
        )
    
    def get_subject_threshold_summary(self, subject_ids=None):
        """
        Get a summary of threshold crossings by subject

        Parameters:
            subject_ids (List[str], optional): List of subject IDs to include

        Returns:
            pd.DataFrame: DataFrame with summary statistics for each subject
        """
        if self.threshold_analyzer is None:
            raise ValueError("Threshold analyzer not initialized. Call initialize_threshold_analyzer first.")
        
        return self.threshold_analyzer.get_subject_crossing_summary(subject_ids=subject_ids)
    
    def add_threshold_to_session_data(self, feature_name, column_suffix="crossed"):
        """
        Add threshold crossing data back to the original session data for a specific feature

        Parameters:
            feature_name (str): Name of the feature to add threshold data for
            column_suffix (str, optional): Suffix to add to the feature name for the new column

        Returns:
            pd.DataFrame: Original session data with added threshold columns
        """
        if self.threshold_analyzer is None:
            raise ValueError("Threshold analyzer not initialized. Call initialize_threshold_analyzer first.")
        
        return self.threshold_analyzer.add_feature_to_session_data(
            feature_name=feature_name,
            column_suffix=column_suffix
        )
    
    def initialize_alert_service(self, config: Optional[Dict[str, Any]] = None) -> AlertService:
        """
        Initialize alert service for monitoring and reporting issues
        
        Parameters:
            config (Optional[Dict[str, Any]]): Configuration for alert service
        
        Returns:
            AlertService: Initialized alert service
        """
        # Create alert service with access to this AppUtils instance
        self.alert_service = AlertService(app_utils=self, config=config)
        
        # Force reset caches for a clean start
        if hasattr(self.alert_service, 'force_reset'):
            self.alert_service.force_reset()
        
        return self.alert_service
    
    def get_alerts(self, subject_ids: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get combined alerts for specified subjects
        
        Parameters:
            subject_ids (Optional[List[str]]): List of subject IDs to get alerts for
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of alerts by subject
        """
        self._ensure_alert_service()
        return self.alert_service.get_alerts(subject_ids)
    
    def get_subjects_with_alerts(self, 
                               threshold_features: Optional[List[str]] = None,
                               quantile_features: Optional[List[str]] = None,
                               quantile_categories: Optional[List[str]] = None) -> List[str]:
        """
        Get list of subjects with alerts matching specified criteria
        
        Parameters:
            threshold_features (Optional[List[str]]): Threshold features to check
            quantile_features (Optional[List[str]]): Quantile features to check
            quantile_categories (Optional[List[str]]): Quantile categories to check
        
        Returns:
            List[str]: List of subject IDs with matching alerts
        """
        self._ensure_alert_service()
        return self.alert_service.get_subjects_with_alerts(
            threshold_features=threshold_features,
            quantile_features=quantile_features,
            quantile_categories=quantile_categories
        )
    
    def get_alert_summary(self, subject_id: str) -> Dict[str, str]:
        """
        Get summary of alerts for a subject
        
        Parameters:
            subject_id (str): Subject ID to get summary for
        
        Returns:
            Dict[str, str]: Dictionary with alert summaries
        """
        self._ensure_alert_service()
        return self.alert_service.get_alert_summary(subject_id)
    
    def has_critical_alerts(self, subject_id: str) -> bool:
        """
        Check if a subject has critical alerts
        
        Parameters:
            subject_id (str): Subject ID to check
        
        Returns:
            bool: Whether the subject has critical alerts
        """
        self._ensure_alert_service()
        return self.alert_service.has_critical_alerts(subject_id)
    
    def get_alert_counts(self) -> Dict[str, Any]:
        """
        Get counts of different types of alerts across all subjects
        
        Returns:
            Dict[str, Any]: Dictionary with alert counts
        """
        self._ensure_alert_service()
        return self.alert_service.get_alert_counts()
    
    def _ensure_alert_service(self) -> None:
        """
        Ensure alert service is initialized
        
        Raises:
            ValueError: If alert service is not initialized
        """
        if self.alert_service is None:
            # Auto-initialize with default settings if not already initialized
            self.initialize_alert_service()
            
        # Check that required analyzers are available
        if self.quantile_analyzer is None or self.threshold_analyzer is None:
            raise ValueError(
                "Alert service requires initialized quantile and threshold analyzers. "
                "Call initialize_threshold_analyzer and initialize_quantile_analyzer first."
            )

    def get_threshold_violations(self, subject_ids=None, start_date=None, end_date=None):
        """
        Get sessions that violate thresholds, focusing only on actual violations
        
        Parameters:
            subject_ids (List[str], optional): List of subject IDs to include
            start_date (str or datetime, optional): Start date for filtering (inclusive)
            end_date (str or datetime, optional): End date for filtering (inclusive)
        
        Returns:
            pd.DataFrame: DataFrame with sessions that violate thresholds
        """
        if self.threshold_analyzer is None:
            raise ValueError("Threshold analyzer not initialized. Call initialize_threshold_analyzer first.")
        
        return self.threshold_analyzer.get_threshold_violations(
            subject_ids=subject_ids,
            start_date=start_date,
            end_date=end_date
        )

    def get_subjects_with_violations(self, features=None):
        """
        Get list of subjects that have threshold violations
        
        Parameters:
            features (List[str], optional): List of specific features to check for violations
                If None, checks for violations on any feature
        
        Returns:
            List[str]: List of subject IDs with threshold violations
        """
        if self.threshold_analyzer is None:
            raise ValueError("Threshold analyzer not initialized. Call initialize_threshold_analyzer first.")
        
        # Get threshold summary
        summary = self.get_subject_threshold_summary()
        
        # Look for violation columns based on the new naming convention
        if features is None:
            # Look for any feature with violations
            violation_cols = [col for col in summary.columns 
                             if '_has_violations' in col or 
                             any(x in col for x in ['above_upper_count', 'below_lower_count', 'outside_range_count'])]
        else:
            # Look for violations only in specified features
            violation_cols = []
            for feature in features:
                feature_cols = [col for col in summary.columns 
                               if col.startswith(feature) and 
                               (col.endswith('_has_violations') or 
                                any(x in col for x in ['above_upper_count', 'below_lower_count', 'outside_range_count']))]
                violation_cols.extend(feature_cols)
        
        # No violation columns found
        if not violation_cols:
            return []
        
        # Check each subject for violations
        subjects_with_violations = []
        
        for _, row in summary.iterrows():
            has_violation = False
            
            # Check each violation column
            for col in violation_cols:
                if col.endswith('_has_violations'):
                    # Direct flag column
                    if row[col]:
                        has_violation = True
                        break
                elif col.endswith('_count'):
                    # Count column - has violation if count > 0
                    if row[col] > 0:
                        has_violation = True
                        break
            
            if has_violation:
                subjects_with_violations.append(row['subject_id'])
        
        return subjects_with_violations

    def get_violation_summary(self, subject_id):
        """
        Get a summary of threshold violations for a specific subject
        
        Parameters:
            subject_id (str): Subject ID to get summary for
        
        Returns:
            Dict[str, Any]: Dictionary with violation summary information
        """
        if self.threshold_analyzer is None:
            raise ValueError("Threshold analyzer not initialized. Call initialize_threshold_analyzer first.")
        
        # Get threshold summary for this subject
        summary = self.get_subject_threshold_summary([subject_id])
        
        # If subject not found, return empty result
        if summary.empty:
            return {"subject_id": subject_id, "has_violations": False}
        
        # Get the row for this subject
        subject_row = summary.iloc[0]
        
        # Initialize result
        result = {
            "subject_id": subject_id,
            "total_sessions": subject_row.get('total_sessions', 0),
            "violations": {}
        }
        
        # Get violation columns based on the new naming convention
        violation_cols = [col for col in summary.columns 
                         if col.endswith('_count') and 
                         any(x in col for x in ['above_upper', 'below_lower', 'outside_range'])]
        
        # Get feature names from columns
        feature_names = set()
        for col in violation_cols:
            parts = col.split('_')
            if len(parts) >= 3:
                feature = '_'.join(parts[:-2])  # Everything before the last two parts
                feature_names.add(feature)
        
        # Extract violation info for each feature
        has_any_violation = False
        
        for feature in feature_names:
            # Get violation columns for this feature
            feature_cols = [col for col in violation_cols if col.startswith(feature)]
            
            # Skip if no violation columns found
            if not feature_cols:
                continue
            
            # Check if any violations exist
            has_violations = False
            for col in feature_cols:
                if subject_row[col] > 0:
                    has_violations = True
                    has_any_violation = True
                    break
            
            # Add details if violations exist
            if has_violations:
                feature_details = {}
                
                # Add counts for each type of violation
                for col in feature_cols:
                    violation_type = col.replace(f"{feature}_", "").replace("_count", "")
                    count = subject_row[col]
                    if count > 0:
                        feature_details[violation_type] = {
                            "count": count,
                            "percent": subject_row.get(col.replace("_count", "_percent"), 0)
                        }
                        
                        # Add first date if available
                        date_col = col.replace("_count", "_first_date")
                        if date_col in subject_row and not pd.isna(subject_row[date_col]):
                            feature_details[violation_type]["first_date"] = subject_row[date_col]
                
                result["violations"][feature] = feature_details
        
        result["has_violations"] = has_any_violation
        
        return result

    def add_violations_to_alerts(self, violation_features=None):
        """
        Add threshold violation data to the alert service
        
        Parameters:
            violation_features (List[str], optional): List of specific features to check for violations
                If None, checks all features with thresholds
        
        Returns:
            Dict[str, Dict[str, Any]]: Updated alerts dictionary
        """
        if self.threshold_analyzer is None:
            raise ValueError("Threshold analyzer not initialized. Call initialize_threshold_analyzer first.")
        
        if self.alert_service is None:
            self.initialize_alert_service()
        
        # Get subjects with violations
        subjects_with_violations = self.get_subjects_with_violations(violation_features)
        
        # Get current alerts
        all_alerts = self.get_alerts(subjects_with_violations)
        
        # For each subject with violations, add detailed violation data
        for subject_id in subjects_with_violations:
            # Get violation summary
            violation_summary = self.get_violation_summary(subject_id)
            
            # Add to alerts if subject exists in alerts dict
            if subject_id in all_alerts:
                all_alerts[subject_id]['violations'] = violation_summary
        
        return all_alerts

    def get_threshold_alerts(self, subject_ids=None):
        """
        Get threshold alerts for given subjects
        
        Parameters:
            subject_ids (List[str], optional): List of subject IDs to get alerts for
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping subject IDs to their threshold alerts
        """
        self._ensure_alert_service()
        return self.alert_service.get_threshold_alerts(subject_ids)

    def get_quantile_alerts(self, subject_ids=None):
        """
        Get quantile alerts for given subjects
        
        Parameters:
            subject_ids (List[str], optional): List of subject IDs to get alerts for
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping subject IDs to their quantile alerts
        """
        self._ensure_alert_service()
        return self.alert_service.get_quantile_alerts(subject_ids)

