from .app_data_load import EnhancedDataLoader
from .app_analysis import ReferenceProcessor, QuantileAnalyzer, ThresholdAnalyzer, BootstrapManager, DataPipelineManager
from .app_analysis.overall_percentile_calculator import OverallPercentileCalculator
from .app_alerts import AlertService, AlertCoordinator
from .cache_utils import CacheManager
from .ui_utils import UIDataManager
from .percentile_utils import PercentileCoordinator
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from datetime import datetime, timedelta
import numpy as np


class AppUtils:
    """
    Slim coordinator class for AIND Dashboard app utilities
    
    This class serves as the main entry point for all app functionality,
    delegating to specialized modules while maintaining backward compatibility.
    All heavy lifting is done by specialized utility classes.
    """

    def __init__(self):
        """Initialize app utils with specialized modules"""
        # Core modules
        self.data_loader = EnhancedDataLoader()
        self.cache_manager = CacheManager()
        self.ui_data_manager = UIDataManager()
        
        # Specialized managers 
        self.pipeline_manager = DataPipelineManager(
            cache_manager=self.cache_manager,
            ui_data_manager=self.ui_data_manager
        )
        
        self.percentile_coordinator = PercentileCoordinator(
            cache_manager=self.cache_manager,
            pipeline_manager=self.pipeline_manager
        )
        
        self.alert_coordinator = AlertCoordinator(
            cache_manager=self.cache_manager,
            pipeline_manager=self.pipeline_manager
        )
        
        # Backward compatibility references
        self.reference_processor = None
        self.quantile_analyzer = None
        self.bootstrap_manager = None
        self.alert_service = None
        self.threshold_analyzer = None
        
        # Legacy compatibility
        self.percentile_calculator = self.percentile_coordinator.percentile_calculator
        self._cache = self.cache_manager._cache

    def get_session_data(self, load_bpod: bool = False, use_cache: bool = True) -> pd.DataFrame:
        """Get session data with caching support"""
        if use_cache and self.cache_manager.has('raw_data'):
            print("Using cached raw session data")
            return self.cache_manager.get('raw_data')
        
        data = self.data_loader.load(load_bpod=load_bpod) if load_bpod else self.data_loader.get_data()
        self.cache_manager.set('raw_data', data)
        self._invalidate_derived_caches()
        return data
    
    def reload_data(self, load_bpod: bool = False) -> pd.DataFrame:
        """Force reload session data"""
        data = self.data_loader.reload_data(load_bpod=load_bpod)
        self.cache_manager.set('raw_data', data)
        self._invalidate_derived_caches()
        return data
    
    def get_subject_sessions(self, subject_id: str) -> Optional[pd.DataFrame]:
        """Get all sessions for a specific subject"""
        return self.data_loader.get_subject_sessions(subject_id)

    def get_most_recent_subject_sessions(self, use_cache: bool = True) -> pd.DataFrame:
        """Get most recent session for each subject with processed metrics"""
        if use_cache and self.cache_manager.has('session_level_data'):
            session_data = self.cache_manager.get('session_level_data')
        else:
            raw_data = self.get_session_data(use_cache=True)
            session_data = self.process_data_pipeline(raw_data, use_cache=True)
        
        session_data = session_data.sort_values(['subject_id', 'session_date'], ascending=[True, False])
        return session_data.groupby('subject_id').first().reset_index()

    def process_data_pipeline(self, df: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
        """Process raw data through the unified session-level pipeline"""
        result = self.pipeline_manager.process_data_pipeline(df, use_cache=use_cache)
        
        # Sync backward compatibility references
        self.reference_processor = self.pipeline_manager.reference_processor
        self.quantile_analyzer = self.pipeline_manager.quantile_analyzer
        self.bootstrap_manager = self.pipeline_manager.bootstrap_manager
        
        return result
    
    def initialize_reference_processor(self, features_config, min_sessions=5, min_days=7, outlier_config=None):
        """Initialize reference processor"""
        processor = self.pipeline_manager.initialize_reference_processor(
            features_config=features_config,
            min_sessions=min_sessions,
            min_days=min_days,
            outlier_config=outlier_config
        )
        self.reference_processor = processor
        return processor
    
    def initialize_quantile_analyzer(self, stratified_data):
        """Initialize quantile analyzer"""
        analyzer = self.pipeline_manager.initialize_quantile_analyzer(stratified_data)
        self.quantile_analyzer = analyzer
        return analyzer
    
    def initialize_bootstrap_manager(self, bootstrap_config: Optional[Dict[str, Any]] = None) -> BootstrapManager:
        """Initialize bootstrap manager"""
        manager = self.pipeline_manager.initialize_bootstrap_manager(bootstrap_config)
        self.bootstrap_manager = manager
        return manager

    def initialize_alert_service(self, config: Optional[Dict[str, Any]] = None) -> AlertService:
        """Initialize alert service"""
        self.alert_service = self.alert_coordinator.initialize_alert_service(
            app_utils=self, 
            config=config
        )
        return self.alert_service

    def get_session_overall_percentiles(self, subject_ids=None, use_cache=True, feature_weights=None):
        """Get session-level overall percentile scores"""
        return self.percentile_coordinator.get_session_overall_percentiles(
            subject_ids=subject_ids,
            use_cache=use_cache,
            feature_weights=feature_weights
        )
    
    def get_quantile_alerts(self, subject_ids=None):
        """Get quantile alerts for subjects"""
        if self.alert_coordinator.alert_service is None:
            self.initialize_alert_service()
        return self.alert_coordinator.get_quantile_alerts(subject_ids)

    def get_unified_alerts(self, subject_ids=None, use_cache=True):
        """Get unified alerts for subjects"""
        if self.alert_coordinator.alert_service is None:
            self.initialize_alert_service()
        return self.alert_coordinator.get_unified_alerts(
            subject_ids=subject_ids, 
            use_cache=use_cache
        )
    
    def optimize_session_data_storage(self, session_data: pd.DataFrame) -> Dict[str, Any]:
        """Create optimized data structures for UI components"""
        return self.ui_data_manager.optimize_session_data_storage(
            session_data, 
            bootstrap_manager=self.bootstrap_manager,
            cache_manager=self.cache_manager
        )
    
    def create_ui_optimized_structures(self, session_data: pd.DataFrame) -> Dict[str, Any]:
        """Create UI-optimized data structures for fast rendering"""
        return self.ui_data_manager.create_ui_optimized_structures(
            session_data, 
            bootstrap_manager=self.bootstrap_manager
        )
    
    def get_subject_display_data(self, subject_id: str, use_cache: bool = True) -> Dict[str, Any]:
        """Get optimized subject data for UI display"""
        return self._get_ui_data_with_fallback(
            lambda ui_structures: self.ui_data_manager.get_subject_display_data(subject_id, ui_structures),
            use_cache
        )
    
    def get_table_display_data(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """Get optimized table display data"""
        return self._get_ui_data_with_fallback(
            lambda ui_structures: self.ui_data_manager.get_table_display_data(ui_structures),
            use_cache
        )
    
    def get_time_series_data(self, subject_id: str, use_cache: bool = True) -> Dict[str, Any]:
        """Get optimized time series data for visualization"""
        return self._get_ui_data_with_fallback(
            lambda ui_structures: self.ui_data_manager.get_time_series_data(subject_id, ui_structures),
            use_cache
        )

    def get_bootstrap_coverage_stats(self, use_cache: bool = True) -> Dict[str, Any]:
        """Get bootstrap coverage statistics"""
        from .app_analysis.statistical_utils import StatisticalUtils
        return StatisticalUtils.get_bootstrap_coverage_stats(
            cache_manager=self.cache_manager,
            use_cache=use_cache
        )
    
    def get_bootstrap_enabled_strata(self, use_cache: bool = True) -> set:
        """Get strata with bootstrap enhancement enabled"""
        from .app_analysis.statistical_utils import StatisticalUtils
        return StatisticalUtils.get_bootstrap_enabled_strata(
            cache_manager=self.cache_manager,
            use_cache=use_cache
        )
    
    def generate_bootstrap_distributions(self, force_regenerate: bool = False, strata_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate bootstrap distributions for eligible strata"""
        from .app_analysis.statistical_utils import StatisticalUtils
        result = StatisticalUtils.generate_bootstrap_distributions(
            bootstrap_manager=self.bootstrap_manager,
            quantile_analyzer=self.quantile_analyzer,
            reference_processor=self.reference_processor,
            cache_manager=self.cache_manager,
            force_regenerate=force_regenerate,
            strata_filter=strata_filter
        )
        
        if result.get('bootstrap_enabled_count', 0) > 0:
            self._update_optimized_cache_with_bootstrap()
        
        return result

    def get_bootstrap_enhancement_summary(self, use_cache: bool = True) -> Dict[str, Any]:
        """Get comprehensive bootstrap enhancement summary"""
        from .app_analysis.statistical_utils import StatisticalUtils
        return StatisticalUtils.get_bootstrap_enhancement_summary(
            cache_manager=self.cache_manager,
            bootstrap_manager=self.bootstrap_manager,
            reference_processor=self.reference_processor,
            use_cache=use_cache
        )

    def calculate_session_bootstrap_cis(self, session_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate bootstrap confidence intervals for session data"""
        from .app_analysis.statistical_utils import StatisticalUtils
        return StatisticalUtils.calculate_session_bootstrap_cis(
            session_data=session_data,
            bootstrap_manager=self.bootstrap_manager,
            reference_processor=self.reference_processor,
            quantile_analyzer=self.quantile_analyzer
        )
    
    def get_memory_usage_summary(self) -> Dict[str, Any]:
        """Get memory usage summary for monitoring"""
        return self.cache_manager.get_memory_usage_summary()
    
    def compress_cache_data(self, force: bool = False) -> Dict[str, Any]:
        """Compress cached data to reduce memory usage"""
        return self.cache_manager.compress_cache_data(force)

    def _invalidate_derived_caches(self):
        """Reset all derived data caches when raw data changes"""
        self.cache_manager.invalidate_derived_caches()
        
        # Clear component caches
        if hasattr(self, 'percentile_calculator'):
            self.percentile_calculator.clear_cache()
        if hasattr(self, 'percentile_coordinator'):
            self.percentile_coordinator.clear_percentile_cache()
        if hasattr(self, 'alert_coordinator'):
            self.alert_coordinator.clear_alert_cache()
        if hasattr(self, 'bootstrap_manager') and self.bootstrap_manager is not None:
            self.bootstrap_manager.clear_cache()
        if hasattr(self, 'pipeline_manager') and self.pipeline_manager.bootstrap_manager is not None:
            self.pipeline_manager.bootstrap_manager.clear_cache()
    
    def _get_ui_data_with_fallback(self, data_getter, use_cache: bool):
        """Helper method for UI data retrieval with cache fallback"""
        if use_cache and self.cache_manager.has('ui_structures'):
            ui_structures = self.cache_manager.get('ui_structures')
            return data_getter(ui_structures)
        
        if self.cache_manager.has('session_level_data'):
            ui_structures = self.create_ui_optimized_structures(self.cache_manager.get('session_level_data'))
            self.cache_manager.set('ui_structures', ui_structures)
            return data_getter(ui_structures)
        
        return {} if 'subject' in str(data_getter) else []
    
    def _update_optimized_cache_with_bootstrap(self):
        """Update optimized cache with new bootstrap information"""
        print("Updating optimized cache with new bootstrap information...")
        if self.cache_manager.has('session_level_data'):
            optimized_storage = self.optimize_session_data_storage(self.cache_manager.get('session_level_data'))
            self.cache_manager.set('optimized_storage', optimized_storage)

    # Legacy compatibility methods
    def _add_session_metadata(self, session_data: pd.DataFrame) -> pd.DataFrame:
        """Add metadata to session data (delegates to pipeline manager)"""
        return self.pipeline_manager._add_session_metadata(session_data)
    
    def _map_percentile_to_category(self, percentile: float) -> str:
        """Map percentile to category (delegates to UI manager)"""
        return self.ui_data_manager.map_percentile_to_category(percentile)
    
    def _get_strata_abbreviation(self, strata: str) -> str:
        """Get strata abbreviation (delegates to UI manager)"""
        return self.ui_data_manager.get_strata_abbreviation(strata)
    
    def _calculate_data_hash(self, df: pd.DataFrame) -> str:
        """Calculate data hash for validation"""
        import hashlib
        data_str = f"{len(df)}_{df['subject_id'].nunique()}_{df['session_date'].max()}"
        return hashlib.md5(data_str.encode()).hexdigest()[:8]

