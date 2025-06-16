from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app_utils.simple_logger import get_logger

logger = get_logger('pipeline_manager')

from .reference_processor import ReferenceProcessor
from .quantile_analyzer import QuantileAnalyzer
from .overall_percentile_calculator import OverallPercentileCalculator


class DataPipelineManager:
    """
    Core data processing pipeline coordination for the AIND Dashboard
    
    This class manages the unified data processing pipeline that transforms raw session data
    into processed data with percentiles, rolling averages, and metadata for UI consumption.
    
    Key Responsibilities:
    - Initialize and coordinate analysis modules (reference processor, quantile analyzer, etc.)
    - Execute the main unified processing pipeline
    - Add session metadata for UI consumption
    - Provide robust statistical analysis with Wilson confidence intervals
    """
    
    def __init__(self, cache_manager=None, ui_data_manager=None):
        """
        Initialize the pipeline manager
        
        Parameters:
            cache_manager: CacheManager instance for result caching
            ui_data_manager: UIDataManager instance for UI optimization
        """
        # Core analysis components
        self.reference_processor = None
        self.quantile_analyzer = None
        
        # Support utilities
        self.percentile_calculator = OverallPercentileCalculator()
        self.cache_manager = cache_manager
        self.ui_data_manager = ui_data_manager
        
        # Default features configuration for pipeline initialization
        self.default_features_config = {
            'finished_trials': False,    # Higher is better
            'ignore_rate': True,         # Lower is better
            'total_trials': False,       # Higher is better
            'foraging_performance': False,   # Higher is better
            'abs(bias_naive)': True      # Lower is better 
        }
        
        # Default outlier configuration for enhanced robustness
        self.default_outlier_config = {
            'method': 'iqr',
            'factor': 1.5,
            'handling': 'weighted',
            'outlier_weight': 0.5,
            'min_data_points': 4
        }
    
    def initialize_reference_processor(self, 
                                     features_config: Optional[Dict[str, bool]] = None, 
                                     min_sessions: int = 5, 
                                     min_days: int = 7, 
                                     outlier_config: Optional[Dict[str, Any]] = None) -> ReferenceProcessor:
        """
        Initialize reference processor with enhanced outlier detection support

        Parameters:
            features_config: Dict[str, bool]
                Configuration of features (feature_name: higher or lower better)
                Uses default if None
            min_sessions: int
                Minimum number of sessions required for eligibility
            min_days: int
                Minimum number of days required for eligibility
            outlier_config: Dict[str, Any], optional
                Outlier detection configuration

        Returns: 
            ReferenceProcessor: Initialized reference processor
        """
        # Use default features config if not provided
        if features_config is None:
            features_config = self.default_features_config
        
        # PHASE 2: Default outlier configuration for enhanced robustness
        if outlier_config is None:
            outlier_config = self.default_outlier_config
        
        self.reference_processor = ReferenceProcessor(
            features_config=features_config,
            min_sessions=min_sessions,
            min_days=min_days,
            outlier_config=outlier_config
        )
        
        logger.info(f"Reference processor initialized with {outlier_config['method']} outlier detection")
        
        return self.reference_processor
    
    def initialize_quantile_analyzer(self, stratified_data: Dict[str, pd.DataFrame]) -> QuantileAnalyzer:
        """
        Initialize quantile analyzer with Wilson confidence interval support

        Parameters:
            stratified_data: Dict[str, pd.DataFrame]
                Dictionary of stratified data

        Returns:
            QuantileAnalyzer: Initialized quantile analyzer with Wilson CI support
        """
        self.quantile_analyzer = QuantileAnalyzer(
            stratified_data=stratified_data,
            historical_data=None,  # Can be set later if needed
        )
        
        logger.info("QuantileAnalyzer initialized with Wilson confidence intervals")
        
        return self.quantile_analyzer
    
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
        logger.info(f"Starting unified data processing pipeline with {len(df)} sessions")
        
        # Check cache first
        if use_cache and self.cache_manager and self.cache_manager.has('session_level_data'):
            logger.info("Using cached session-level data")
            return self.cache_manager.get('session_level_data')
        
        # Step 1: Initialize processors if needed
        if self.reference_processor is None:
            logger.info("Initializing reference processor with default configuration...")
            self.initialize_reference_processor(
                features_config=self.default_features_config,
                min_sessions=1, 
                min_days=1, 
                outlier_config=self.default_outlier_config
            )
        
        # Step 2: Get eligible subjects and preprocess data with enhanced outlier detection
        eligible_subjects = self.reference_processor.get_eligible_subjects(df)
        eligible_df = df[df['subject_id'].isin(eligible_subjects)]
        logger.info(f"Got {len(eligible_subjects)} eligible subjects")
        
        # PHASE 2: Enhanced preprocessing with outlier detection
        processed_df = self.reference_processor.preprocess_data(eligible_df, remove_outliers=True)
        logger.info(f"Preprocessed data: {len(processed_df)} sessions")
        
        # Report outlier weights if present
        if 'outlier_weight' in processed_df.columns:
            outlier_sessions = (processed_df['outlier_weight'] < 1.0).sum()
            total_sessions = len(processed_df)
            outlier_rate = (outlier_sessions / total_sessions) * 100
            logger.info(f"{outlier_sessions}/{total_sessions} sessions ({outlier_rate:.1f}%) have outlier weights")
        
        # Step 3: Prepare session-level data with rolling averages and strata assignments
        session_level_data = self.reference_processor.prepare_session_level_data(processed_df)
        logger.info(f"Prepared session-level data: {len(session_level_data)} sessions")
        
        # Step 4: Calculate reference distributions for percentile calculation
        if self.quantile_analyzer is None:
            # Create reference distributions for session percentiles
            stratified_data = self.reference_processor.prepare_for_quantile_analysis(
                processed_df, include_history=True
            )
            # Create reference distributions for session percentiles
            self.initialize_quantile_analyzer(stratified_data)
            logger.info("QuantileAnalyzer initialized with Wilson confidence intervals")
        
        # Step 5: Calculate session-level percentiles using reference distributions
        session_with_percentiles = self.quantile_analyzer.calculate_session_level_percentiles(session_level_data)
        logger.info("Calculated session-level percentiles")
        
        # Step 6: Calculate overall percentiles for each session
        comprehensive_data = self.percentile_calculator.calculate_session_overall_percentile(
            session_with_percentiles
        )
        logger.info("Calculated overall session percentiles")
        
        # Step 6.5: Calculate overall rolling averages for hover information
        comprehensive_data = self.percentile_calculator.calculate_session_overall_rolling_average(
            comprehensive_data
        )
        logger.info("Calculated overall session rolling averages")
        
        # Step 7: Add alerts and metadata
        comprehensive_data = self._add_session_metadata(comprehensive_data)
        
        # Cache the results if cache manager available
        if self.cache_manager:
            self.cache_manager.set('session_level_data', comprehensive_data)
            self.cache_manager.set_timestamp('last_process_time')
            
            # Create optimized storage structure for fast lookups
            if self.ui_data_manager:
                optimized_storage = self.ui_data_manager.optimize_session_data_storage(
                    comprehensive_data, 
                    cache_manager=self.cache_manager
                )
                self.cache_manager.set('optimized_storage', optimized_storage)
                
                # Create UI-optimized structures for fast component rendering
                ui_structures = self.ui_data_manager.create_ui_optimized_structures(
                    comprehensive_data
                )
                self.cache_manager.set('ui_structures', ui_structures)
                
                logger.info(f"Created optimized storage and UI structures for {len(optimized_storage['subjects'])} subjects")
        
        logger.info(f"Unified pipeline complete: {len(comprehensive_data)} sessions processed")
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
        
        # Get feature list from reference processor
        if self.reference_processor is None:
            logger.warning("Reference processor not available for metadata generation")
            return result_df
        
        feature_list = list(self.reference_processor.features_config.keys())
        
        # Add percentile categories for each feature
        for feature in feature_list:
            percentile_col = f"{feature}_session_percentile"
            category_col = f"{feature}_category"
            
            if percentile_col in result_df.columns:
                # Map percentiles to categories using UI data manager if available
                if self.ui_data_manager:
                    result_df[category_col] = result_df[percentile_col].apply(
                        lambda x: self.ui_data_manager.map_percentile_to_category(x) if not pd.isna(x) else 'NS'
                    )
                else:
                    # Fallback to simple categorization
                    result_df[category_col] = result_df[percentile_col].apply(
                        lambda x: self._simple_percentile_to_category(x) if not pd.isna(x) else 'NS'
                    )
        
        # Add overall percentile category
        if 'session_overall_percentile' in result_df.columns:
            if self.ui_data_manager:
                result_df['overall_percentile_category'] = result_df['session_overall_percentile'].apply(
                    lambda x: self.ui_data_manager.map_percentile_to_category(x) if not pd.isna(x) else 'NS'
                )
            else:
                result_df['overall_percentile_category'] = result_df['session_overall_percentile'].apply(
                    lambda x: self._simple_percentile_to_category(x) if not pd.isna(x) else 'NS'
                )
        
        # PHASE 2: Add simple boolean outlier flag based on outlier_weight
        if 'outlier_weight' in result_df.columns:
            result_df['is_outlier'] = result_df['outlier_weight'] < 1.0
        else:
            result_df['is_outlier'] = False  # Default if no outlier detection applied
        
        return result_df
    
    def _simple_percentile_to_category(self, percentile: float) -> str:
        """
        Simple fallback percentile to category mapping when UI data manager unavailable
        
        Parameters:
            percentile: float
                Percentile value (0-100)
                
        Returns:
            str: Alert category (SB, B, N, G, SG)
        """
        if percentile <= 5:
            return 'SB'  # Significantly Below
        elif percentile <= 25:
            return 'B'   # Below
        elif percentile <= 75:
            return 'N'   # Normal
        elif percentile <= 95:
            return 'G'   # Good
        else:
            return 'SG'  # Significantly Good 