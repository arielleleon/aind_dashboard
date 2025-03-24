from .app_data_load import AppLoadData
from .app_analysis import ReferenceProcessor, QuantileAnalyzer, ThresholdAnalyzer
from .app_alerts import AlertService
from typing import Dict, List, Optional, Union, Any

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

