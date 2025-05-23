from .app_data_load import AppLoadData
from .app_analysis import ReferenceProcessor, QuantileAnalyzer, ThresholdAnalyzer
from .app_analysis.overall_percentile_calculator import OverallPercentileCalculator
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
        self.percentile_calculator = OverallPercentileCalculator()
        
        # Simplified cache for processed data
        self._cache = {
            'raw_data': None,
            'processed_data': None,  
            'formatted_data': None,
            'stratified_data': None,
            'overall_percentiles': None,
            'unified_alerts': None,
            'last_process_time': None,
            'data_hash': None,
            'session_level_data': None
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
        self._cache['formatted_data_hash'] = None
        self._cache['stratified_data'] = None
        self._cache['overall_percentiles'] = None
        self._cache['unified_alerts'] = None
        self._cache['last_process_time'] = None
        self._cache['data_hash'] = None
        self._cache['session_level_data'] = None
        self._cache['optimized_storage'] = None  # Optimized storage cache
        self._cache['ui_structures'] = None      # UI-optimized structures cache
        
        # Also clear percentile calculator cache
        if hasattr(self, 'percentile_calculator'):
            self.percentile_calculator.clear_cache()
    
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
    
    def calculate_overall_percentile(self, subject_ids = None, use_cache = True, feature_weights = None):
        """
        Calculate overall percentile scores for subjects using the OverallPercentileCalculator

        Parameters:
            subject_ids (List[str], optional): List of specific subjects to calculate for
            use_cache (bool): Whether to use cached results if available
            feature_weights (Dict[str, float], optional): Optional weights for features

        Returns:
            pd.DataFrame: Dataframe with overall percentile scores
        """
        # Return cached results if available and no specific subjects requested
        if use_cache and subject_ids is None:
            cached_percentiles = self.percentile_calculator.get_cached_percentiles()
            if cached_percentiles is not None:
                print("Using cached overall percentiles")
                return cached_percentiles
            
            # Fall back to app_utils cache if calculator cache is empty
            if self._cache['overall_percentiles'] is not None:
                print("Using cached overall percentiles from app_utils")
                return self._cache['overall_percentiles']
            
        if self.quantile_analyzer is None:
            raise ValueError("Quantile analyzer not initialized. Process data first.")
        
        # Get comprehensive dataframe from quantile analyzer
        comprehensive_df = self.quantile_analyzer.create_comprehensive_dataframe(include_history=False)
        
        # Use dedicated calculator to compute overall percentiles
        percentiles = self.percentile_calculator.calculate_overall_percentile(
            comprehensive_df=comprehensive_df,
            subject_ids=subject_ids,
            feature_weights=feature_weights
        )
        
        # Cache results if calculating for all subjects
        if subject_ids is None:
            self._cache['overall_percentiles'] = percentiles
            self.percentile_calculator.set_cache(percentiles)
            
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
    
    def analyze_session_level_percentiles(self, 
                                         subject_ids: Optional[List[str]] = None, 
                                         feature_weights: Optional[Dict[str, float]] = None,
                                         use_cache: bool = True) -> pd.DataFrame:
        """
        DEPRECATED: Use process_data_pipeline() instead
        
        This method is kept for backward compatibility but delegates to the unified pipeline.
        """
        print("DEPRECATED: analyze_session_level_percentiles() - Use process_data_pipeline() instead")
        
        # Get raw data and process through unified pipeline
        raw_data = self.get_session_data(use_cache=True)
        
        # Filter for specific subjects if requested
        if subject_ids is not None:
            raw_data = raw_data[raw_data['subject_id'].isin(subject_ids)]
        
        # Use the unified pipeline
        return self.process_data_pipeline(raw_data, use_cache=use_cache)

    def process_data_pipeline(self, df: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
        """
        NEW UNIFIED PIPELINE: Process raw session data through the complete session-level pipeline
        
        This replaces the previous dual approach (strata-level + session-level) with a unified
        session-level approach that provides all necessary metrics for the application.
        
        Parameters:
            df: pd.DataFrame
                Raw session data from the data loader
            use_cache: bool
                Whether to use cached results if available
                
        Returns:
            pd.DataFrame
                Complete session-level data with percentiles, rolling averages, and metadata
        """
        print(f"Starting unified data processing pipeline with {len(df)} sessions")
        
        # Check cache first
        if use_cache and self._cache['session_level_data'] is not None:
            print("Using cached session-level data")
            return self._cache['session_level_data']
        
        # Step 1: Initialize processors if needed
        if self.reference_processor is None:
            features_config = {
                'finished_trials': False,  # Higher is better
                'ignore_rate': True,     # Lower is better
                'total_trials': False,   # Higher is better
                'foraging_performance': False,   # Higher is better
                'abs(bias_naive)': True  # Lower is better 
            }
            self.initialize_reference_processor(features_config, min_sessions=1, min_days=1)
        
        # Step 2: Get eligible subjects and preprocess data
        eligible_subjects = self.reference_processor.get_eligible_subjects(df)
        eligible_df = df[df['subject_id'].isin(eligible_subjects)]
        print(f"Got {len(eligible_subjects)} eligible subjects")
        
        processed_df = self.reference_processor.preprocess_data(eligible_df, remove_outliers=False)
        print(f"Preprocessed data: {len(processed_df)} sessions")
        
        # Step 3: Prepare session-level data with rolling averages and strata assignments
        session_level_data = self.reference_processor.prepare_session_level_data(processed_df)
        print(f"Prepared session-level data: {len(session_level_data)} sessions")
        
        # Step 4: Calculate reference distributions for percentile calculation
        # We still need reference distributions, but now they're used for session-level percentiles
        if self.quantile_analyzer is None:
            # Create reference distributions using the current approach but for session percentiles
            stratified_data = self.reference_processor.prepare_for_quantile_analysis(
                processed_df, include_history=True
            )
            self.initialize_quantile_analyzer(stratified_data)
            print(f"Initialized quantile analyzer with {len(stratified_data)} strata")
        
        # Step 5: Calculate session-level percentiles using reference distributions
        session_with_percentiles = self.quantile_analyzer.calculate_session_level_percentiles(session_level_data)
        print(f"Calculated session-level percentiles")
        
        # Step 6: Calculate overall percentiles for each session
        comprehensive_data = self.percentile_calculator.calculate_session_overall_percentile(
            session_with_percentiles
        )
        print(f"Calculated overall session percentiles")
        
        # Step 6.5: Calculate overall rolling averages for hover information
        comprehensive_data = self.percentile_calculator.calculate_session_overall_rolling_average(
            comprehensive_data
        )
        print(f"Calculated overall session rolling averages")
        
        # Step 7: Add alerts and metadata
        comprehensive_data = self._add_session_metadata(comprehensive_data)
        
        # Cache the results
        self._cache['session_level_data'] = comprehensive_data
        self._cache['last_process_time'] = datetime.now()
        
        # Create optimized storage structure for fast lookups
        optimized_storage = self.optimize_session_data_storage(comprehensive_data)
        self._cache['optimized_storage'] = optimized_storage
        
        # Create UI-optimized structures for fast component rendering
        ui_structures = self.create_ui_optimized_structures(comprehensive_data)
        self._cache['ui_structures'] = ui_structures
        
        print(f"Unified pipeline complete: {len(comprehensive_data)} sessions processed")
        print(f"Optimized storage created with {len(optimized_storage['subjects'])} subjects")
        print(f"UI structures created for {len(ui_structures['feature_rank_data'])} subjects")
        return comprehensive_data
    
    def _add_session_metadata(self, session_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add metadata to session-level data for easier consumption by UI components
        
        Parameters:
            session_data: pd.DataFrame
                Session data with percentiles
                
        Returns:
            pd.DataFrame
                Enhanced session data with metadata
        """
        result_df = session_data.copy()
        
        # Add percentile categories for each feature
        feature_list = list(self.reference_processor.features_config.keys())
        
        for feature in feature_list:
            percentile_col = f"{feature}_session_percentile"
            category_col = f"{feature}_category"
            
            if percentile_col in result_df.columns:
                # Map percentiles to categories using the same logic as alerts
                result_df[category_col] = result_df[percentile_col].apply(
                    lambda x: self._map_percentile_to_category(x) if not pd.isna(x) else 'NS'
                )
        
        # Add overall percentile category
        if 'session_overall_percentile' in result_df.columns:
            result_df['overall_percentile_category'] = result_df['session_overall_percentile'].apply(
                lambda x: self._map_percentile_to_category(x) if not pd.isna(x) else 'NS'
            )
        
        return result_df
    
    def _map_percentile_to_category(self, percentile: float) -> str:
        """
        Map percentile value to alert category
        
        Parameters:
            percentile: float
                Percentile value (0-100)
                
        Returns:
            str: Alert category (SB, B, N, G, SG)
        """
        if pd.isna(percentile):
            return 'NS'
        
        # Use the correct thresholds from the alert service
        if percentile < 6.5:
            return 'SB'  # Severely Below: < 6.5%
        elif percentile < 28:
            return 'B'   # Below: < 28%
        elif percentile <= 72:
            return 'N'   # Normal: 28% - 72%
        elif percentile <= 93.5:
            return 'G'   # Good: 72% - 93.5%
        else:
            return 'SG'  # Severely Good: > 93.5%

    def get_subject_session_level_percentiles(self, subject_id: str) -> pd.DataFrame:
        """
        Get session-level percentiles for a specific subject using the unified pipeline
        
        Parameters:
            subject_id: str
                The subject ID to get session-level percentiles for
                
        Returns:
            pd.DataFrame
                DataFrame containing session-level percentiles for the subject
        """
        # Get all session data from the unified pipeline
        if self._cache['session_level_data'] is not None:
            session_data = self._cache['session_level_data']
        else:
            # Process all data to get session-level data
            raw_data = self.get_session_data(use_cache=True)
            session_data = self.process_data_pipeline(raw_data, use_cache=True)
        
        # Filter for the requested subject
        subject_data = session_data[session_data['subject_id'] == subject_id]
        
        if subject_data.empty:
            print(f"No session-level data found for subject {subject_id}")
            return pd.DataFrame()
        
        # Sort by session date or index
        if 'session_date' in subject_data.columns:
            subject_data = subject_data.sort_values('session_date')
        elif 'session_index' in subject_data.columns:
            subject_data = subject_data.sort_values('session_index')
        
        return subject_data

    def get_most_recent_subject_sessions(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Get the most recent session for each subject with all session-level metrics
        
        Parameters:
            use_cache: bool
                Whether to use cached data if available
                
        Returns:
            pd.DataFrame
                DataFrame with most recent session for each subject
        """
        # Get all session data from the unified pipeline
        if use_cache and self._cache['session_level_data'] is not None:
            session_data = self._cache['session_level_data']
        else:
            # Process all data to get session-level data
            raw_data = self.get_session_data(use_cache=True)
            session_data = self.process_data_pipeline(raw_data, use_cache=True)
        
        # Sort by subject ID and session date (descending)
        session_data = session_data.sort_values(['subject_id', 'session_date'], ascending=[True, False])
        
        # Get most recent session for each subject
        most_recent = session_data.groupby('subject_id').first().reset_index()
        
        return most_recent

    def optimize_session_data_storage(self, session_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Optimize session-level data storage for efficient lookup and memory usage
        
        This creates optimized data structures for:
        1. Subject-indexed session data
        2. Strata-indexed reference distributions  
        3. Compressed historical data
        
        Parameters:
            session_data: pd.DataFrame
                Complete session-level data from unified pipeline
                
        Returns:
            Dict[str, Any]: Optimized storage structure
        """
        print("Optimizing session data storage...")
        
        # Create subject-indexed storage for fast subject lookups
        subject_data = {}
        strata_reference = {}
        
        # Group by subject for efficient subject-based operations
        for subject_id, subject_sessions in session_data.groupby('subject_id'):
            # Sort sessions by date
            subject_sessions = subject_sessions.sort_values('session_date')
            
            # Store only essential columns to save memory
            essential_columns = [
                'subject_id', 'session_date', 'session', 'strata', 'session_index',
                'session_overall_percentile', 'overall_percentile_category',
                'session_overall_rolling_avg',  # Add overall rolling average for hover info
                'is_current_strata', 'is_last_session',
                # CRITICAL FIX: Add essential metadata columns
                'PI', 'trainer', 'rig', 'current_stage_actual', 'curriculum_name',
                'water_day_total', 'base_weight', 'target_weight', 'weight_after',
                'total_trials', 'finished_trials', 'ignore_rate', 'foraging_performance',
                'abs(bias_naive)', 'finished_rate'
            ]
            
            # Add feature-specific columns
            feature_columns = [col for col in subject_sessions.columns 
                             if col.endswith(('_session_percentile', '_category', '_processed_rolling_avg'))]
            essential_columns.extend(feature_columns)
            
            # Filter to available columns and ensure uniqueness
            available_columns = [col for col in essential_columns if col in subject_sessions.columns]
            # Remove duplicates while preserving order
            unique_columns = []
            seen = set()
            for col in available_columns:
                if col not in seen:
                    unique_columns.append(col)
                    seen.add(col)
            
            # Store compressed subject data
            subject_data[subject_id] = {
                'sessions': subject_sessions[unique_columns].to_dict('records'),
                'current_strata': subject_sessions['strata'].iloc[-1],
                'total_sessions': len(subject_sessions),
                'first_session_date': subject_sessions['session_date'].min(),
                'last_session_date': subject_sessions['session_date'].max(),
                'strata_history': subject_sessions[['strata', 'session_date']].drop_duplicates('strata').to_dict('records')
            }
        
        # Create strata-indexed reference distributions for percentile calculations
        for strata, strata_sessions in session_data.groupby('strata'):
            # Store only the reference distribution data needed for percentile calculations
            processed_features = [col for col in strata_sessions.columns if col.endswith('_processed_rolling_avg')]
            
            if processed_features:
                reference_data = strata_sessions[processed_features + ['subject_id']].dropna()
                
                strata_reference[strata] = {
                    'subject_count': len(reference_data),
                    'session_count': len(strata_sessions),
                    'reference_distributions': {
                        feature: reference_data[feature].values.tolist() 
                        for feature in processed_features
                        if not reference_data[feature].isna().all()
                    }
                }
        
        # Create optimized storage structure
        optimized_storage = {
            'subjects': subject_data,
            'strata_reference': strata_reference,
            'metadata': {
                'total_subjects': len(subject_data),
                'total_sessions': len(session_data),
                'total_strata': len(strata_reference),
                'storage_timestamp': pd.Timestamp.now(),
                'data_hash': self._calculate_data_hash(session_data)
            }
        }
        
        print(f"Optimized storage created:")
        print(f"  - {len(subject_data)} subjects")
        print(f"  - {len(strata_reference)} strata references")
        print(f"  - {len(session_data)} total sessions")
        
        return optimized_storage
    
    def _calculate_data_hash(self, df: pd.DataFrame) -> str:
        """Calculate a hash for data validation"""
        import hashlib
        data_str = f"{len(df)}_{df['subject_id'].nunique()}_{df['session_date'].max()}"
        return hashlib.md5(data_str.encode()).hexdigest()[:8]
    
    def get_subject_optimized_data(self, subject_id: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get optimized subject data using the new storage format
        
        Parameters:
            subject_id: str
                Subject ID to retrieve data for
            use_cache: bool
                Whether to use cached optimized storage
                
        Returns:
            Dict[str, Any]: Optimized subject data
        """
        # Check if we have optimized storage cached
        if use_cache and 'optimized_storage' in self._cache and self._cache['optimized_storage'] is not None:
            optimized_storage = self._cache['optimized_storage']
            return optimized_storage['subjects'].get(subject_id, {})
        
        # If not cached, create optimized storage
        if self._cache['session_level_data'] is not None:
            optimized_storage = self.optimize_session_data_storage(self._cache['session_level_data'])
            self._cache['optimized_storage'] = optimized_storage
            return optimized_storage['subjects'].get(subject_id, {})
        
        # Fallback to processing data
        raw_data = self.get_session_data(use_cache=True)
        session_data = self.process_data_pipeline(raw_data, use_cache=True)
        optimized_storage = self.optimize_session_data_storage(session_data)
        self._cache['optimized_storage'] = optimized_storage
        
        return optimized_storage['subjects'].get(subject_id, {})
    
    def get_strata_reference_data(self, strata: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get reference distribution data for a specific strata
        
        Parameters:
            strata: str
                Strata to get reference data for
            use_cache: bool
                Whether to use cached data
                
        Returns:
            Dict[str, Any]: Reference distribution data
        """
        # Check for optimized storage
        if use_cache and 'optimized_storage' in self._cache and self._cache['optimized_storage'] is not None:
            optimized_storage = self._cache['optimized_storage']
            return optimized_storage['strata_reference'].get(strata, {})
        
        # Create optimized storage if needed
        if self._cache['session_level_data'] is not None:
            optimized_storage = self.optimize_session_data_storage(self._cache['session_level_data'])
            self._cache['optimized_storage'] = optimized_storage
            return optimized_storage['strata_reference'].get(strata, {})
        
        return {}

    def get_storage_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current storage optimization
        
        Returns:
            Dict[str, Any]: Storage summary statistics
        """
        summary = {
            'raw_data_cached': self._cache['raw_data'] is not None,
            'session_level_cached': self._cache['session_level_data'] is not None,
            'optimized_storage_cached': 'optimized_storage' in self._cache and self._cache['optimized_storage'] is not None,
            'formatted_data_cached': self._cache['formatted_data'] is not None
        }
        
        # Add size information if available
        if summary['optimized_storage_cached']:
            storage = self._cache['optimized_storage']
            summary.update({
                'subjects_count': len(storage['subjects']),
                'strata_count': len(storage['strata_reference']),
                'total_sessions': storage['metadata']['total_sessions'],
                'storage_timestamp': storage['metadata']['storage_timestamp']
            })
        
        if summary['session_level_cached']:
            summary['session_level_rows'] = len(self._cache['session_level_data'])
        
        if summary['formatted_data_cached']:
            summary['formatted_data_rows'] = len(self._cache['formatted_data'])
        
        return summary

    def create_ui_optimized_structures(self, session_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Create UI-optimized data structures for fast component rendering
        
        Specialized structures for:
        1. Feature rank plot data
        2. Subject detail views
        3. Table display optimization
        4. Time series visualization
        
        Parameters:
            session_data: pd.DataFrame
                Complete session-level data from unified pipeline
                
        Returns:
            Dict[str, Any]: UI-optimized data structures
        """
        print("Creating UI-optimized data structures...")
        
        # DEBUG: Check what columns are available in session_data
        print(f"📊 Session data columns ({len(session_data.columns)} total):")
        session_percentile_cols = [col for col in session_data.columns if col.endswith('_session_percentile')]
        rolling_avg_cols = [col for col in session_data.columns if col.endswith('_rolling_avg')]
        category_cols = [col for col in session_data.columns if col.endswith('_category')]
        overall_cols = [col for col in session_data.columns if 'overall_percentile' in col]
        
        print(f"  Session percentile columns ({len(session_percentile_cols)}): {session_percentile_cols}")
        print(f"  Rolling average columns ({len(rolling_avg_cols)}): {rolling_avg_cols}")
        print(f"  Category columns ({len(category_cols)}): {category_cols}")
        print(f"  Overall percentile columns ({len(overall_cols)}): {overall_cols}")
        
        # Check if we have sample data with percentiles
        if len(session_data) > 0:
            print(f"  Sample data for first row:")
            sample_cols = session_percentile_cols[:2] + ['session_overall_percentile']
            for col in sample_cols:
                if col in session_data.columns:
                    value = session_data.iloc[0][col]
                    print(f"    {col}: {value}")
        print("")  # Empty line for readability
        
        ui_structures = {
            'feature_rank_data': {},
            'subject_lookup': {},
            'strata_lookup': {},
            'time_series_data': {},
            'table_display_cache': {}
        }
        
        # 1. Feature Rank Plot Optimization
        # Pre-compute feature ranking data for each subject
        features = ['finished_trials', 'ignore_rate', 'total_trials', 'foraging_performance', 'abs(bias_naive)']
        
        for subject_id, subject_sessions in session_data.groupby('subject_id'):
            # Get most recent session data
            latest_session = subject_sessions.sort_values('session_date').iloc[-1]
            
            feature_ranks = {}
            for feature in features:
                session_percentile_col = f"{feature}_session_percentile"
                category_col = f"{feature}_category"
                
                if session_percentile_col in latest_session:
                    feature_ranks[feature] = {
                        'percentile': latest_session[session_percentile_col],
                        'category': latest_session.get(category_col, 'NS'),
                        'value': latest_session.get(feature, None)
                    }
            
            ui_structures['feature_rank_data'][subject_id] = {
                'features': feature_ranks,
                'overall_percentile': latest_session.get('session_overall_percentile'),
                'overall_category': latest_session.get('overall_percentile_category', 'NS'),
                'strata': latest_session.get('strata', 'Unknown'),
                'session_date': latest_session.get('session_date'),
                'session_count': latest_session.get('session', 0)
            }
        
        # 2. Subject Detail Lookup Optimization
        # Create fast subject lookup with essential display data
        for subject_id, subject_sessions in session_data.groupby('subject_id'):
            subject_sessions = subject_sessions.sort_values('session_date')
            latest_session = subject_sessions.iloc[-1]
            
            ui_structures['subject_lookup'][subject_id] = {
                'latest': {
                    'session_date': latest_session['session_date'],
                    'session': latest_session['session'],
                    'strata': latest_session['strata'],
                    'overall_percentile': latest_session.get('session_overall_percentile'),
                    'overall_category': latest_session.get('overall_percentile_category', 'NS'),
                    'PI': latest_session.get('PI', 'N/A'),
                    'trainer': latest_session.get('trainer', 'N/A'),
                    'rig': latest_session.get('rig', 'N/A')
                },
                'summary': {
                    'total_sessions': len(subject_sessions),
                    'first_session_date': subject_sessions['session_date'].min(),
                    'last_session_date': subject_sessions['session_date'].max(),
                    'unique_strata': subject_sessions['strata'].nunique(),
                    'current_strata': latest_session['strata']
                }
            }
        
        # 3. Strata Lookup Optimization
        # Pre-compute strata summaries for filtering
        for strata, strata_sessions in session_data.groupby('strata'):
            unique_subjects = strata_sessions['subject_id'].nunique()
            total_sessions = len(strata_sessions)
            
            # Calculate strata performance metrics
            overall_percentiles = strata_sessions['session_overall_percentile'].dropna()
            
            ui_structures['strata_lookup'][strata] = {
                'subject_count': unique_subjects,
                'session_count': total_sessions,
                'avg_performance': overall_percentiles.mean() if len(overall_percentiles) > 0 else None,
                'performance_std': overall_percentiles.std() if len(overall_percentiles) > 0 else None,
                'subjects': strata_sessions['subject_id'].unique().tolist()
            }
        
        # 4. Time Series Data Optimization
        # Pre-compute time series data for subjects with compressed format
        for subject_id, subject_sessions in session_data.groupby('subject_id'):
            subject_sessions = subject_sessions.sort_values('session_date')
            
            # Extract time series data in compressed format
            time_series = {
                'sessions': subject_sessions['session'].tolist(),
                'dates': subject_sessions['session_date'].dt.strftime('%Y-%m-%d').tolist(),
                'overall_percentiles': subject_sessions['session_overall_percentile'].fillna(-1).tolist(),
                'overall_rolling_avg': subject_sessions['session_overall_rolling_avg'].fillna(-1).tolist(),
                'strata': subject_sessions['strata'].tolist()
            }
            
            # Add RAW feature values for timeseries plotting (not the processed rolling averages)
            for feature in features:
                # Store raw feature values for timeseries component to apply its own rolling average
                if feature in subject_sessions.columns:
                    time_series[f"{feature}_raw"] = subject_sessions[feature].fillna(-1).tolist()
                    print(f"Added raw data for {feature}: {len(subject_sessions[feature].dropna())} valid values")
                
                # Keep percentiles for fallback compatibility
                percentile_col = f"{feature}_session_percentile"
                if percentile_col in subject_sessions.columns:
                    time_series[f"{feature}_percentiles"] = subject_sessions[percentile_col].fillna(-1).tolist()
            
            ui_structures['time_series_data'][subject_id] = time_series
        
        # 5. Table Display Cache
        # Pre-compute table display data for fast rendering
        most_recent = session_data.sort_values('session_date').groupby('subject_id').last().reset_index()
        
        table_data = []
        for _, row in most_recent.iterrows():
            display_row = {
                'subject_id': row['subject_id'],
                'session_date': row['session_date'],
                'session': row['session'],
                'strata': row['strata'],
                'strata_abbr': self._get_strata_abbreviation(row['strata']),
                'overall_percentile': row.get('session_overall_percentile'),
                'overall_category': row.get('overall_percentile_category', 'NS'),
                'percentile_category': row.get('overall_percentile_category', 'NS'),  # Alias for compatibility
                'combined_alert': row.get('overall_percentile_category', 'NS'),  # Will be updated with alerts
                'session_overall_rolling_avg': row.get('session_overall_rolling_avg'),  # For percentile plot hover
                'PI': row.get('PI', 'N/A'),
                'trainer': row.get('trainer', 'N/A'),
                'rig': row.get('rig', 'N/A'),
                'current_stage_actual': row.get('current_stage_actual', 'N/A'),
                'curriculum_name': row.get('curriculum_name', 'N/A'),
                # Add essential metadata columns for filtering
                'water_day_total': row.get('water_day_total'),
                'base_weight': row.get('base_weight'),
                'target_weight': row.get('target_weight'),
                'weight_after': row.get('weight_after'),
                'total_trials': row.get('total_trials'),
                'finished_trials': row.get('finished_trials'),
                'ignore_rate': row.get('ignore_rate'),
                'foraging_performance': row.get('foraging_performance'),
                'abs(bias_naive)': row.get('abs(bias_naive)'),
                'finished_rate': row.get('finished_rate'),
                # Initialize alert columns with default values
                'threshold_alert': 'N',
                'total_sessions_alert': 'N',
                'stage_sessions_alert': 'N',
                'water_day_total_alert': 'N',
                'ns_reason': ''
            }
            
            # Add feature-specific data (both percentiles and rolling averages)
            for feature in features:
                percentile_col = f"{feature}_session_percentile"
                category_col = f"{feature}_category"
                rolling_avg_col = f"{feature}_processed_rolling_avg"
                
                display_row[f"{feature}_session_percentile"] = row.get(percentile_col)
                display_row[f"{feature}_category"] = row.get(category_col, 'NS')
                
                # CRITICAL FIX: Add rolling average columns to table display cache
                display_row[f"{feature}_processed_rolling_avg"] = row.get(rolling_avg_col)
            
            table_data.append(display_row)
        
        ui_structures['table_display_cache'] = table_data
        
        print(f"UI structures created:")
        print(f"  - Feature rank data: {len(ui_structures['feature_rank_data'])} subjects")
        print(f"  - Subject lookups: {len(ui_structures['subject_lookup'])} subjects")
        print(f"  - Strata lookups: {len(ui_structures['strata_lookup'])} strata")
        print(f"  - Time series data: {len(ui_structures['time_series_data'])} subjects")
        print(f"  - Table display cache: {len(ui_structures['table_display_cache'])} rows")
        
        return ui_structures
    
    def _get_strata_abbreviation(self, strata: str) -> str:
        """Get abbreviated strata name for UI display"""
        if not strata:
            return ''
        
        # Hard coded mappings for common terms
        strata_mappings = {
            'Uncoupled Baiting': 'UB',
            'Coupled Baiting': 'CB', 
            'Uncoupled Without Baiting': 'UWB',
            'Coupled Without Baiting': 'CWB',
            'BEGINNER': 'B',
            'INTERMEDIATE': 'I',
            'ADVANCED': 'A',
            'v1': '1',
            'v2': '2',
            'v3': '3'
        }
        
        # Split the strata name
        parts = strata.split('_')
        
        # Handle different strata formats
        if len(parts) >= 3:
            # Format: curriculum_Stage_Version
            curriculum = '_'.join(parts[:-2])
            stage = parts[-2]
            version = parts[-1]
            
            # Get abbreviations
            curriculum_abbr = strata_mappings.get(curriculum, curriculum[:2].upper())
            stage_abbr = strata_mappings.get(stage, stage[0])
            version_abbr = strata_mappings.get(version, version[-1])
            
            return f"{curriculum_abbr}{stage_abbr}{version_abbr}"
        
        return strata.replace(" ", "")

    def get_feature_rank_data(self, subject_id: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get optimized feature rank data for the feature rank plot
        
        Parameters:
            subject_id: str
                Subject ID to get feature rank data for
            use_cache: bool
                Whether to use cached UI structures
                
        Returns:
            Dict[str, Any]: Feature rank data optimized for UI rendering
        """
        # Check for UI-optimized cache
        if use_cache and 'ui_structures' in self._cache and self._cache['ui_structures'] is not None:
            ui_structures = self._cache['ui_structures']
            return ui_structures['feature_rank_data'].get(subject_id, {})
        
        # Fallback to creating UI structures
        if self._cache['session_level_data'] is not None:
            ui_structures = self.create_ui_optimized_structures(self._cache['session_level_data'])
            self._cache['ui_structures'] = ui_structures
            return ui_structures['feature_rank_data'].get(subject_id, {})
        
        return {}
    
    def get_subject_display_data(self, subject_id: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get optimized subject data for UI display components
        
        Parameters:
            subject_id: str
                Subject ID to get display data for
            use_cache: bool
                Whether to use cached UI structures
                
        Returns:
            Dict[str, Any]: Subject display data optimized for UI rendering
        """
        # Check for UI-optimized cache
        if use_cache and 'ui_structures' in self._cache and self._cache['ui_structures'] is not None:
            ui_structures = self._cache['ui_structures']
            return ui_structures['subject_lookup'].get(subject_id, {})
        
        # Fallback to creating UI structures
        if self._cache['session_level_data'] is not None:
            ui_structures = self.create_ui_optimized_structures(self._cache['session_level_data'])
            self._cache['ui_structures'] = ui_structures
            return ui_structures['subject_lookup'].get(subject_id, {})
        
        return {}
    
    def get_table_display_data(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Get optimized table display data for fast rendering
        
        Parameters:
            use_cache: bool
                Whether to use cached UI structures
                
        Returns:
            List[Dict[str, Any]]: Table data optimized for UI rendering
        """
        # Check for UI-optimized cache
        if use_cache and 'ui_structures' in self._cache and self._cache['ui_structures'] is not None:
            ui_structures = self._cache['ui_structures']
            return ui_structures['table_display_cache']
        
        # Fallback to creating UI structures
        if self._cache['session_level_data'] is not None:
            ui_structures = self.create_ui_optimized_structures(self._cache['session_level_data'])
            self._cache['ui_structures'] = ui_structures
            return ui_structures['table_display_cache']
        
        return []
    
    def get_time_series_data(self, subject_id: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get optimized time series data for visualization components
        
        Parameters:
            subject_id: str
                Subject ID to get time series data for
            use_cache: bool
                Whether to use cached UI structures
                
        Returns:
            Dict[str, Any]: Time series data optimized for UI rendering
        """
        # Check for UI-optimized cache
        if use_cache and 'ui_structures' in self._cache and self._cache['ui_structures'] is not None:
            ui_structures = self._cache['ui_structures']
            return ui_structures['time_series_data'].get(subject_id, {})
        
        # Fallback to creating UI structures
        if self._cache['session_level_data'] is not None:
            ui_structures = self.create_ui_optimized_structures(self._cache['session_level_data'])
            self._cache['ui_structures'] = ui_structures
            return ui_structures['time_series_data'].get(subject_id, {})
        
        return {}

    def get_memory_usage_summary(self) -> Dict[str, Any]:
        """
        Get detailed memory usage summary for optimization monitoring
        
        Returns:
            Dict[str, Any]: Memory usage statistics
        """
        try:
            import psutil
            import sys
            
            # Get process memory info
            process = psutil.Process()
            memory_info = process.memory_info()
            
            summary = {
                'process_memory_mb': memory_info.rss / 1024 / 1024,
                'cache_sizes': {},
                'optimization_status': {},
                'total_data_objects': 0
            }
            
            # Calculate cache sizes
            for cache_key, cache_data in self._cache.items():
                if cache_data is not None:
                    try:
                        cache_size = sys.getsizeof(cache_data)
                        summary['cache_sizes'][cache_key] = {
                            'size_mb': cache_size / 1024 / 1024,
                            'type': type(cache_data).__name__
                        }
                        
                        # Add detailed size for specific caches
                        if cache_key == 'session_level_data' and hasattr(cache_data, '__len__'):
                            summary['cache_sizes'][cache_key]['rows'] = len(cache_data)
                        elif cache_key == 'ui_structures' and isinstance(cache_data, dict):
                            for struct_key, struct_data in cache_data.items():
                                if isinstance(struct_data, dict):
                                    summary['cache_sizes'][f'{cache_key}.{struct_key}'] = {
                                        'size_mb': sys.getsizeof(struct_data) / 1024 / 1024,
                                        'count': len(struct_data)
                                    }
                        elif cache_key == 'optimized_storage' and isinstance(cache_data, dict):
                            metadata = cache_data.get('metadata', {})
                            summary['cache_sizes'][cache_key].update({
                                'subjects': metadata.get('total_subjects', 0),
                                'sessions': metadata.get('total_sessions', 0),
                                'strata': metadata.get('total_strata', 0)
                            })
                            
                    except Exception as e:
                        summary['cache_sizes'][cache_key] = {'error': str(e)}
            
            # Optimization status
            summary['optimization_status'] = {
                'unified_pipeline_active': self._cache.get('session_level_data') is not None,
                'optimized_storage_active': self._cache.get('optimized_storage') is not None,
                'ui_structures_active': self._cache.get('ui_structures') is not None,
                'memory_efficient_caching': True  # We always use efficient caching now
            }
            
            # Calculate total cached objects
            for cache_data in self._cache.values():
                if cache_data is not None:
                    summary['total_data_objects'] += 1
            
            return summary
            
        except Exception as e:
            return {'error': f"Memory monitoring failed: {str(e)}"}
    
    def compress_cache_data(self, force: bool = False) -> Dict[str, Any]:
        """
        Compress cached data to reduce memory usage
        
        Parameters:
            force: bool
                Whether to force compression even if memory usage is acceptable
                
        Returns:
            Dict[str, Any]: Compression results and memory savings
        """
        import pickle
        import gzip
        import sys
        
        results = {
            'compressed_caches': [],
            'memory_saved_mb': 0,
            'compression_ratios': {}
        }
        
        # Check if compression is needed
        memory_summary = self.get_memory_usage_summary()
        process_memory = memory_summary.get('process_memory_mb', 0)
        
        # Compress if memory usage is high or force is requested
        if process_memory > 500 or force:  # 500MB threshold
            print(f"Compressing cache data (current memory: {process_memory:.1f}MB)...")
            
            # Compress large cache objects
            compressible_caches = ['session_level_data', 'optimized_storage', 'ui_structures']
            
            for cache_key in compressible_caches:
                if cache_key in self._cache and self._cache[cache_key] is not None:
                    try:
                        # Get original size
                        original_data = self._cache[cache_key]
                        original_size = sys.getsizeof(original_data)
                        
                        # Compress using pickle + gzip
                        compressed_data = gzip.compress(pickle.dumps(original_data))
                        compressed_size = sys.getsizeof(compressed_data)
                        
                        # Calculate compression ratio
                        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1
                        
                        # Only compress if we get significant savings (>30% reduction)
                        if compression_ratio > 1.3:
                            # Store compressed data with metadata
                            self._cache[f'{cache_key}_compressed'] = {
                                'data': compressed_data,
                                'original_size': original_size,
                                'compressed_size': compressed_size,
                                'compression_ratio': compression_ratio
                            }
                            
                            # Remove original data to save memory
                            self._cache[cache_key] = None
                            
                            results['compressed_caches'].append(cache_key)
                            results['memory_saved_mb'] += (original_size - compressed_size) / 1024 / 1024
                            results['compression_ratios'][cache_key] = compression_ratio
                            
                            print(f"  Compressed {cache_key}: {compression_ratio:.1f}x reduction")
                        
                    except Exception as e:
                        print(f"  Failed to compress {cache_key}: {str(e)}")
        
        return results
    
    def decompress_cache_data(self, cache_key: str) -> Any:
        """
        Decompress cached data when needed
        
        Parameters:
            cache_key: str
                Cache key to decompress
                
        Returns:
            Any: Decompressed data
        """
        compressed_key = f'{cache_key}_compressed'
        
        if compressed_key in self._cache and self._cache[compressed_key] is not None:
            try:
                import pickle
                import gzip
                
                compressed_info = self._cache[compressed_key]
                compressed_data = compressed_info['data']
                
                # Decompress
                decompressed_data = pickle.loads(gzip.decompress(compressed_data))
                
                # Cache the decompressed data temporarily
                self._cache[cache_key] = decompressed_data
                
                print(f"Decompressed {cache_key} ({compressed_info['compression_ratio']:.1f}x)")
                
                return decompressed_data
                
            except Exception as e:
                print(f"Failed to decompress {cache_key}: {str(e)}")
                return None
        
        # Return regular cached data if not compressed
        return self._cache.get(cache_key)
    
    def optimize_memory_usage(self, aggressive: bool = False) -> Dict[str, Any]:
        """
        Optimize memory usage by compressing caches and cleaning up unused data
        
        Parameters:
            aggressive: bool
                Whether to use aggressive optimization (may impact performance)
                
        Returns:
            Dict[str, Any]: Optimization results
        """
        print("Optimizing memory usage...")
        
        # Get initial memory state
        initial_memory = self.get_memory_usage_summary()
        initial_memory_mb = initial_memory.get('process_memory_mb', 0)
        
        optimization_results = {
            'initial_memory_mb': initial_memory_mb,
            'final_memory_mb': 0,
            'memory_saved_mb': 0,
            'optimizations_applied': []
        }
        
        # 1. Compress large cache objects
        compression_results = self.compress_cache_data(force=aggressive)
        if compression_results['compressed_caches']:
            optimization_results['optimizations_applied'].append('cache_compression')
            optimization_results['compression_details'] = compression_results
        
        # 2. Clean up redundant caches (aggressive mode)
        if aggressive:
            # Remove raw data cache if processed data exists
            if (self._cache.get('session_level_data') is not None and 
                self._cache.get('raw_data') is not None):
                self._cache['raw_data'] = None
                optimization_results['optimizations_applied'].append('raw_data_cleanup')
            
            # Remove stratified data if UI structures exist
            if (self._cache.get('ui_structures') is not None and 
                self._cache.get('stratified_data') is not None):
                self._cache['stratified_data'] = None
                optimization_results['optimizations_applied'].append('stratified_data_cleanup')
        
        # 3. Force garbage collection
        import gc
        gc.collect()
        optimization_results['optimizations_applied'].append('garbage_collection')
        
        # Get final memory state
        final_memory = self.get_memory_usage_summary()
        final_memory_mb = final_memory.get('process_memory_mb', 0)
        
        optimization_results['final_memory_mb'] = final_memory_mb
        optimization_results['memory_saved_mb'] = initial_memory_mb - final_memory_mb
        
        print(f"Memory optimization complete:")
        print(f"  Initial memory: {initial_memory_mb:.1f}MB")
        print(f"  Final memory: {final_memory_mb:.1f}MB")
        print(f"  Memory saved: {optimization_results['memory_saved_mb']:.1f}MB")
        print(f"  Optimizations: {', '.join(optimization_results['optimizations_applied'])}")
        
        return optimization_results