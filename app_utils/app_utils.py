from .app_data_load import AppLoadData
from .app_analysis import ReferenceProcessor, QuantileAnalyzer, ThresholdAnalyzer
from .app_alerts import AlertService
from typing import Dict, List, Optional, Union, Any
import pandas as pd
from datetime import datetime, timedelta

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
        self.threshold_analyzer = None
        
        # Simplified cache for processed data
        self._cache = {
            'raw_data': None,
            'processed_data': None,  # No longer keyed by window_days
            'formatted_data': None,  # No longer keyed by window_days
            'stratified_data': None,  # No longer keyed by window_days
            'overall_percentiles': None,
            'unified_alerts': None,
            'last_process_time': None,
            'data_hash': None
        }

    def get_session_data(self, load_bpod = False, use_cache = True):
        """
        Get session data from the data loader
        
        Parameters:
            load_bpod (bool): Whether to load bpod data
            use_cache (bool): Whether to use cached data if available
            
        Returns:
            pd.DataFrame: Session data
        """
        if use_cache and self._cache['raw_data'] is not None:
            return self._cache['raw_data']
            
        if load_bpod:
            data = self.data_loader.load(load_bpod = True)
        else:
            data = self.data_loader.get_data()
            
        # Cache the raw data
        self._cache['raw_data'] = data
        
        # Reset other caches since raw data has changed
        self._invalidate_derived_caches()
        
        return data
    
    def _invalidate_derived_caches(self):
        """Reset all derived data caches when raw data changes"""
        self._cache['processed_data'] = None
        self._cache['formatted_data'] = None
        self._cache['stratified_data'] = None
        self._cache['overall_percentiles'] = None
        self._cache['unified_alerts'] = None
        self._cache['last_process_time'] = None
        self._cache['data_hash'] = None
    
    def reload_data(self, load_bpod = False):
        """
        Force reload session data
        """
        data = self.data_loader.load(load_bpod = load_bpod)
        
        # Update cache and invalidate derived caches
        self._cache['raw_data'] = data
        self._invalidate_derived_caches()
        
        return data
    
    def initialize_reference_processor(self, features_config, min_sessions = 5, min_days = 7):
        """
        Initialize reference processor

        Parameters:
            features_config (Dict[str, bool]): Configuration of features (feature_name: higher or lower better)
            min_sessions (int): Minimum number of sessions required for eligibility
            min_days (int): Minimum number of days required for eligibility

        Returns: 
            ReferenceProcessor: Initialized reference processor
        """
        self.reference_processor = ReferenceProcessor(
            features_config = features_config,
            min_sessions = min_sessions,
            min_days = min_days
        )
        return self.reference_processor
    
    def process_reference_data(self, df, reference_date = None, remove_outliers = False, use_cache = True):
        """
        Process data through reference pipeline with enhanced historical data handling
        
        Parameters:
            df (pd.DataFrame): Data to process
            reference_date (datetime, optional): Reference date for sliding window
            remove_outliers (bool): Whether to remove outliers
            use_cache (bool): Whether to use cached data if available
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of stratified data
        """
        # Check if we have cached results
        if use_cache and self._cache['stratified_data'] is not None:
            print(f"Using cached stratified data")
            return self._cache['stratified_data']
            
        if self.reference_processor is None:
            raise ValueError("Reference processor not initialized. Call initialize_reference_processor first.")
        
        # Filter out off-curriculum sessions FIRST
        original_count = len(df)
        
        # Track subjects with off-curriculum sessions
        self.off_curriculum_subjects = {}
        
        # Create a boolean mask for off-curriculum sessions
        off_curriculum_mask = (
            df['curriculum_name'].isna() | 
            (df['curriculum_name'] == "None") |
            df['current_stage_actual'].isna() | 
            (df['current_stage_actual'] == "None") |
            df['curriculum_version'].isna() |
            (df['curriculum_version'] == "None")
        )
        
        # Identify subjects with off-curriculum sessions
        for subject_id in df['subject_id'].unique():
            subject_sessions = df[df['subject_id'] == subject_id]
            off_curriculum_sessions = subject_sessions[off_curriculum_mask.loc[subject_sessions.index]]
            
            if not off_curriculum_sessions.empty:
                # Store information about this subject's off-curriculum sessions
                self.off_curriculum_subjects[subject_id] = {
                    'count': len(off_curriculum_sessions),
                    'total_sessions': len(subject_sessions),
                    'latest_date': off_curriculum_sessions['session_date'].max()
                }
        
        # Remove all off-curriculum sessions from the analysis pipeline
        df_filtered = df[~off_curriculum_mask].copy()
        off_curriculum_count = original_count - len(df_filtered)
        
        print(f"Filtered out {off_curriculum_count} off-curriculum sessions ({off_curriculum_count/original_count:.1%} of total)")
        print(f"Identified {len(self.off_curriculum_subjects)} subjects with off-curriculum sessions")
        
        # Get eligible subjects - no longer using sliding window
        eligible_subjects = self.reference_processor.get_eligible_subjects(df_filtered)
        eligible_df = df_filtered[df_filtered['subject_id'].isin(eligible_subjects)]
        print(f"Got {len(eligible_subjects)} eligible subjects")

        # Preprocess data
        processed_df = self.reference_processor.preprocess_data(eligible_df, remove_outliers)
        print(f"Preprocessed data: {len(processed_df)} sessions")

        # Prepare for quantile analysis
        stratified_data = self.reference_processor.prepare_for_quantile_analysis(
            processed_df, 
            include_history=True
        )
        print(f"Created {len(stratified_data)} strata including historical data")

        # Store historical data for later use
        self.historical_data = self.reference_processor.subject_history
        print(f"Stored historical data for {len(self.historical_data) if self.historical_data is not None else 0} subject-strata combinations")

        # Cache the results
        self._cache['processed_data'] = processed_df
        self._cache['stratified_data'] = stratified_data
        self._cache['last_process_time'] = datetime.now()

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
    
    def calculate_overall_percentile(self, subject_ids = None, use_cache = True):
        """
        Calculate overall percentile scores for subjects using a simple average of feature percentiles

        Parameters:
            subject_ids (List[str], optional): List of specific subjects to calculate for
            use_cache (bool): Whether to use cached results if available

        Returns:
            pd.DataFrame: Dataframe with overall percentile scores
        """
        # Return cached results if available and no specific subjects requested
        if use_cache and subject_ids is None and self._cache['overall_percentiles'] is not None:
            print("Using cached overall percentiles")
            return self._cache['overall_percentiles']
            
        if self.quantile_analyzer is None:
            raise ValueError("Quantile analyzer not initialized. Process data first.")
        
        percentiles = self.quantile_analyzer.calculate_overall_percentile(
            subject_ids = subject_ids
        )
        
        # Cache results if calculating for all subjects
        if subject_ids is None:
            self._cache['overall_percentiles'] = percentiles
            
        return percentiles
    
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

    def initialize_threshold_analyzer(self, threshold_config: Optional[Dict[str, Any]] = None) -> ThresholdAnalyzer:
        """
        Initialize threshold analyzer
        
        Parameters:
            threshold_config: Optional[Dict[str, Any]]
                Configuration for threshold alerts
                
        Returns:
            ThresholdAnalyzer: Initialized threshold analyzer
        """
        
        self.threshold_analyzer = ThresholdAnalyzer(threshold_config)
        return self.threshold_analyzer

    def get_unified_alerts(self, subject_ids = None, use_cache = True):
        """ 
        Get unified alerts
        
        Parameters:
            subject_ids (List[str], optional): List of subject IDs to get alerts for
            use_cache (bool): Whether to use cached alerts if available
            
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping subject IDs to their unified alerts
        """
        # Return cached alerts if available and not requesting specific subjects
        if use_cache and subject_ids is None and self._cache['unified_alerts'] is not None:
            print("Using cached unified alerts")
            return self._cache['unified_alerts']
            
        if self.alert_service is None:
            self.initialize_alert_service()

        alerts = self.alert_service.get_unified_alerts(subject_ids)
        
        # Cache results if getting alerts for all subjects
        if subject_ids is None:
            self._cache['unified_alerts'] = alerts
            
        return alerts
        
    def get_formatted_data(self, window_days=30, reference_date=None, use_cache=True):
        """
        Get formatted data for display, using cache if available
        
        Parameters:
            window_days (int): Number of days to include in sliding window
            reference_date (datetime, optional): Reference date for sliding window
            use_cache (bool): Whether to use cached data if available
            
        Returns:
            pd.DataFrame: Formatted data for display
        """
        # Check if we have cached formatted data
        if use_cache and self._cache['formatted_data'] is not None:
            print(f"Using cached formatted data")
            # Apply time window filter directly to cached data
            formatted_df = self._cache['formatted_data'].copy()
            
            # Apply time window filter directly to the session_date column
            if reference_date is None:
                reference_date = formatted_df['session_date'].max()
            
            start_date = reference_date - timedelta(days=window_days)
            time_filtered_df = formatted_df[formatted_df['session_date'] >= start_date]
            
            # Get most recent session for each subject in the window
            time_filtered_df = time_filtered_df.sort_values('session_date', ascending=False)
            result_df = time_filtered_df.drop_duplicates(subset=['subject_id'], keep='first')
            
            print(f"Applied {window_days} day window filter: {len(result_df)} subjects")
            return result_df
            
        # If not cached, we need to format the data
        from app_elements.app_content.app_dataframe.app_dataframe import AppDataFrame
        
        # Get raw data
        df = self.get_session_data(use_cache=use_cache)
        
        # Create formatter
        formatter = AppDataFrame()
        
        # Format data
        formatted_df = formatter.format_dataframe(df, reference_date=reference_date)
        
        # Cache the full formatted data
        self._cache['formatted_data'] = formatted_df
        
        # Apply time window filter
        if reference_date is None:
            reference_date = formatted_df['session_date'].max()
        
        start_date = reference_date - timedelta(days=window_days)
        time_filtered_df = formatted_df[formatted_df['session_date'] >= start_date]
        
        # Get most recent session for each subject in the window
        time_filtered_df = time_filtered_df.sort_values('session_date', ascending=False)
        result_df = time_filtered_df.drop_duplicates(subset=['subject_id'], keep='first')
        
        print(f"Applied {window_days} day window filter: {len(result_df)} subjects")
        return result_df
    
    def get_subject_sessions(self, subject_id):
        """
        Get all sessions for a specific subject
        
        Parameters:
        -----------
        subject_id : str
            The subject ID to retrieve sessions for
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing all sessions for the subject
        """
        try:
            # Get all data
            all_data = self.data_loader.get_data()
            
            # Filter for specific subject
            subject_data = all_data[all_data['subject_id'] == subject_id].copy()
            
            if subject_data.empty:
                print(f"No sessions found for subject {subject_id}")
                return None
                
            # Sort by session date (most recent first)
            subject_data = subject_data.sort_values('session_date', ascending=False)
            
            return subject_data
        except Exception as e:
            print(f"Error getting sessions for subject {subject_id}: {str(e)}")
            return None