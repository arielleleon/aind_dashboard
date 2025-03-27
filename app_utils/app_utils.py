from .app_data_load import AppLoadData
from .app_analysis import ReferenceProcessor, QuantileAnalyzer
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
    

    def get_quantile_alerts(self, subject_ids=None):
        """
        Get quantile alerts for given subjects
        
        Parameters:
            subject_ids (List[str], optional): List of subject IDs to get alerts for
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping subject IDs to their quantile alerts
        """
        return self.alert_service.get_quantile_alerts(subject_ids)

    def get_alerts(self, subject_ids=None):
        """
        Alias for get_quantile_alerts for backward compatibility
        
        Parameters:
            subject_ids (List[str], optional): List of subject IDs to get alerts for
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping subject IDs to their quantile alerts
        """
        return self.get_quantile_alerts(subject_ids)
