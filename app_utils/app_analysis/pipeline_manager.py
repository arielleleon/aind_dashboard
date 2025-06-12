from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app_utils.simple_logger import get_logger

logger = get_logger('pipeline_manager')

from .reference_processor import ReferenceProcessor
from .quantile_analyzer import QuantileAnalyzer
from .bootstrap_manager import BootstrapManager
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
    - Coordinate bootstrap enhancement and statistical robustness features
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
        self.bootstrap_manager = None
        
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
        Initialize quantile analyzer with Phase 3 bootstrap manager integration

        Parameters:
            stratified_data: Dict[str, pd.DataFrame]
                Dictionary of stratified data

        Returns:
            QuantileAnalyzer: Initialized quantile analyzer with bootstrap support
        """
        self.quantile_analyzer = QuantileAnalyzer(
            stratified_data=stratified_data,
            historical_data=None,  # Can be set later if needed
            # PHASE 3: Pass bootstrap manager for enhanced confidence intervals
            bootstrap_manager=self.bootstrap_manager
        )
        
        # PHASE 3: Report bootstrap integration status
        if self.bootstrap_manager is not None:
            logger.info("QuantileAnalyzer initialized with bootstrap enhancement")
        else:
            logger.info("QuantileAnalyzer initialized with standard confidence intervals")
        
        logger.info(f"QuantileAnalyzer initialized with {'bootstrap' if self.bootstrap_manager else 'standard'} confidence intervals")
        
        return self.quantile_analyzer
    
    def initialize_bootstrap_manager(self, bootstrap_config: Optional[Dict[str, Any]] = None) -> BootstrapManager:
        """
        Initialize bootstrap manager with configuration
        
        Parameters:
            bootstrap_config: Optional[Dict[str, Any]]
                Bootstrap configuration (uses defaults if None)
                
        Returns:
            BootstrapManager: Initialized bootstrap manager
        """
        self.bootstrap_manager = BootstrapManager(bootstrap_config)
        
        logger.info("Bootstrap Manager initialized for statistical robustness")
        
        return self.bootstrap_manager
    
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
            # PHASE 3: Initialize bootstrap manager before quantile analyzer if not already done
            if self.bootstrap_manager is None:
                logger.info("Initializing Bootstrap Manager")
                self.initialize_bootstrap_manager()
            
            # Create reference distributions for session percentiles
            stratified_data = self.reference_processor.prepare_for_quantile_analysis(
                processed_df, include_history=True
            )
            # PHASE 3: Pass bootstrap manager to quantile analyzer
            self.initialize_quantile_analyzer(stratified_data)
            logger.info("Bootstrap Manager initialized for statistical robustness")
            
            # PHASE 3: Generate bootstrap distributions for enhanced confidence intervals
            logger.info("Generating bootstrap distributions")
            bootstrap_result = self._generate_bootstrap_distributions(force_regenerate=False)
            
            # Report bootstrap generation results
            if bootstrap_result.get('bootstrap_enabled_count', 0) > 0:
                enabled_count = bootstrap_result['bootstrap_enabled_count']
                total_strata = bootstrap_result['total_strata']
                logger.info(f"Bootstrap enhancement enabled for {enabled_count}/{total_strata} strata")
            else:
                logger.info("No bootstrap distributions generated - using standard confidence intervals")
                if bootstrap_result.get('warnings'):
                    logger.warning(f"Bootstrap warnings: {len(bootstrap_result['warnings'])} issues detected")
        
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
        
        # Step 6.75: PHASE 1 OPTIMIZATION - Pre-compute bootstrap CIs during pipeline
        comprehensive_data = self._calculate_session_bootstrap_cis(comprehensive_data)
        
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
                    bootstrap_manager=self.bootstrap_manager,
                    cache_manager=self.cache_manager
                )
                self.cache_manager.set('optimized_storage', optimized_storage)
                
                # Create UI-optimized structures for fast component rendering
                ui_structures = self.ui_data_manager.create_ui_optimized_structures(
                    comprehensive_data, 
                    bootstrap_manager=self.bootstrap_manager
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
        
        # PHASE 3: Add bootstrap enhancement indicators
        if self.bootstrap_manager is not None:
            logger.info("Adding bootstrap enhancement indicators to session metadata...")
            
            # Process each session to determine which percentiles used bootstrap
            for idx, row in result_df.iterrows():
                strata = row.get('strata', '')
                
                # Check bootstrap availability for each feature
                for feature in feature_list:
                    percentile_col = f"{feature}_session_percentile"
                    ci_lower_col = f"{feature}_session_percentile_ci_lower"
                    ci_upper_col = f"{feature}_session_percentile_ci_upper"
                    bootstrap_indicator_col = f"{feature}_bootstrap_enhanced"
                    
                    # Check if this feature has bootstrap enhancement for this strata
                    if (percentile_col in result_df.columns and 
                        ci_lower_col in result_df.columns and 
                        ci_upper_col in result_df.columns):
                        
                        # Check if bootstrap is available and CI values are not NaN
                        bootstrap_available = self.bootstrap_manager.is_bootstrap_available(strata, feature)
                        has_valid_ci = (not pd.isna(row[ci_lower_col]) and not pd.isna(row[ci_upper_col]))
                        
                        # Bootstrap enhanced if bootstrap is available AND we have valid CIs
                        bootstrap_enhanced = bootstrap_available and has_valid_ci
                        
                        result_df.loc[idx, bootstrap_indicator_col] = bootstrap_enhanced
                    else:
                        result_df.loc[idx, bootstrap_indicator_col] = False

                # Check bootstrap enhancement for overall percentile
                overall_ci_lower_col = "session_overall_percentile_ci_lower"
                overall_ci_upper_col = "session_overall_percentile_ci_upper"
                overall_bootstrap_indicator_col = "session_overall_bootstrap_enhanced"
                
                if (overall_ci_lower_col in result_df.columns and 
                    overall_ci_upper_col in result_df.columns):
                    
                    # Overall percentile is bootstrap enhanced if ANY feature percentile is bootstrap enhanced
                    feature_bootstrap_indicators = [f"{feature}_bootstrap_enhanced" for feature in feature_list]
                    any_feature_bootstrap = any(
                        result_df.loc[idx, col] for col in feature_bootstrap_indicators 
                        if col in result_df.columns
                    )
                    
                    has_overall_ci = (not pd.isna(row[overall_ci_lower_col]) and not pd.isna(row[overall_ci_upper_col]))
                    
                    result_df.loc[idx, overall_bootstrap_indicator_col] = any_feature_bootstrap and has_overall_ci
                else:
                    result_df.loc[idx, overall_bootstrap_indicator_col] = False
            
            # Count and report bootstrap enhancement summary
            bootstrap_enhanced_sessions = 0
            for feature in feature_list:
                bootstrap_indicator_col = f"{feature}_bootstrap_enhanced"
                if bootstrap_indicator_col in result_df.columns:
                    enhanced_count = result_df[bootstrap_indicator_col].sum()
                    if enhanced_count > 0:
                        logger.info(f"  {feature}: {enhanced_count} sessions with bootstrap-enhanced CIs")
                        bootstrap_enhanced_sessions += enhanced_count
            
            # Report overall bootstrap enhancement
            overall_bootstrap_col = "session_overall_bootstrap_enhanced"
            if overall_bootstrap_col in result_df.columns:
                overall_enhanced_count = result_df[overall_bootstrap_col].sum()
                logger.info(f"  Overall percentile: {overall_enhanced_count} sessions with bootstrap-enhanced CIs")
            
            unique_bootstrap_sessions = len(result_df[result_df[[f"{feature}_bootstrap_enhanced" for feature in feature_list if f"{feature}_bootstrap_enhanced" in result_df.columns]].any(axis=1)])
            logger.info(f"Total sessions with any bootstrap enhancement: {unique_bootstrap_sessions}")

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
    
    def _generate_bootstrap_distributions(self, 
                                        force_regenerate: bool = False,
                                        strata_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate bootstrap distributions for eligible strata using the current data
        
        Parameters:
            force_regenerate: bool
                Force regeneration of all bootstrap distributions
            strata_filter: Optional[List[str]]
                List of specific strata to process (None = all strata)
                
        Returns:
            Dict[str, Any]: Bootstrap generation results
        """
        # REFACTORING: Delegate to enhanced StatisticalUtils
        from .statistical_utils import StatisticalUtils
        return StatisticalUtils.generate_bootstrap_distributions(
            bootstrap_manager=self.bootstrap_manager,
            quantile_analyzer=self.quantile_analyzer,
            reference_processor=self.reference_processor,
            cache_manager=self.cache_manager,
            force_regenerate=force_regenerate,
            strata_filter=strata_filter
        )
    
    def _calculate_session_bootstrap_cis(self, session_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate bootstrap CIs for all session rolling averages during pipeline processing
        
        This pre-computes bootstrap CIs once and stores them in session data to avoid
        expensive real-time calculations during UI creation.
        
        Parameters:
            session_data: pd.DataFrame
                Session-level data with rolling averages and percentiles
                
        Returns:
            pd.DataFrame
                Session data with bootstrap CI columns added
        """
        # REFACTORING: Delegate to enhanced StatisticalUtils
        from .statistical_utils import StatisticalUtils
        return StatisticalUtils.calculate_session_bootstrap_cis(
            session_data=session_data,
            bootstrap_manager=self.bootstrap_manager,
            reference_processor=self.reference_processor,
            quantile_analyzer=self.quantile_analyzer
        )

    def get_bootstrap_enhancement_summary(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of bootstrap enhancement coverage
        
        Parameters:
            use_cache: bool
                Whether to use cached data if available
                
        Returns:
            Dict[str, Any]
                Bootstrap enhancement summary with statistics and subject details
        """
        # REFACTORING: Delegate to enhanced StatisticalUtils
        from .statistical_utils import StatisticalUtils
        return StatisticalUtils.get_bootstrap_enhancement_summary(
            cache_manager=self.cache_manager,
            bootstrap_manager=self.bootstrap_manager,
            reference_processor=self.reference_processor,
            use_cache=use_cache
        ) 