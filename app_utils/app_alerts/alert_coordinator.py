from typing import Dict, List, Optional, Any
from .alert_service import AlertService


class AlertCoordinator:
    """
    Alert service coordination and initialization class
    
    This class handles alert service initialization, coordination,
    and caching for the AIND Dashboard application.
    """
    
    def __init__(self, cache_manager=None, pipeline_manager=None):
        """
        Initialize the AlertCoordinator
        
        Parameters:
            cache_manager: CacheManager instance for alert caching
            pipeline_manager: DataPipelineManager instance for data access
        """
        self.cache_manager = cache_manager
        self.pipeline_manager = pipeline_manager
        self.alert_service = None
    
    def initialize_alert_service(self, app_utils, config: Optional[Dict[str, Any]] = None) -> AlertService:
        """
        Initialize alert service for monitoring and reporting issues
        
        This method creates an AlertService instance with the necessary
        configuration and dependencies, providing clean initialization
        and cache management.
        
        Parameters:
            app_utils: AppUtils instance that provides data access
            config (Optional[Dict[str, Any]]): Configuration for alert service
        
        Returns:
            AlertService: Initialized alert service
        """
        # Create alert service with access to the AppUtils instance
        self.alert_service = AlertService(app_utils=app_utils, config=config)
        
        # Force reset caches for a clean start
        if hasattr(self.alert_service, 'force_reset'):
            self.alert_service.force_reset()
        
        return self.alert_service
    
    def get_quantile_alerts(self, subject_ids=None):
        """
        Get quantile alerts for given subjects with proper delegation
        
        This method provides a clean interface for accessing quantile alerts
        while ensuring the alert service is properly initialized.
        
        Parameters:
            subject_ids (List[str], optional): List of subject IDs to get alerts for
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping subject IDs to their quantile alerts
        """
        if self.alert_service is None:
            raise ValueError("Alert service not initialized. Call initialize_alert_service() first.")
        
        return self.alert_service.get_quantile_alerts(subject_ids)
    
    def get_unified_alerts(self, subject_ids=None, use_cache=True):
        """
        Get unified alerts with caching support and proper coordination
        
        This method provides unified alert access with intelligent caching
        to optimize performance while maintaining data freshness.
        
        Parameters:
            subject_ids (List[str], optional): List of subject IDs to get alerts for
            use_cache (bool): Whether to use cached alerts if available
            
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping subject IDs to their unified alerts
        """
        # Check if alert service is initialized first
        if self.alert_service is None:
            raise ValueError("Alert service not initialized. Call initialize_alert_service() first.")
        
        # Return cached alerts if available and not requesting specific subjects
        if (use_cache and subject_ids is None and 
            self.cache_manager and self.cache_manager.has('unified_alerts')):
            print("Using cached unified alerts")
            return self.cache_manager.get('unified_alerts')

        # Get alerts from alert service
        alerts = self.alert_service.get_unified_alerts(subject_ids)
        
        # Cache results if getting alerts for all subjects
        if subject_ids is None and self.cache_manager:
            self.cache_manager.set('unified_alerts', alerts)
            
        return alerts
    
    def clear_alert_cache(self):
        """
        Clear cached alert data to force refresh
        
        This method provides a clean way to invalidate alert caches
        when the underlying data has changed.
        """
        if self.cache_manager:
            # Clear unified alerts cache
            if self.cache_manager.has('unified_alerts'):
                print("Clearing unified alerts cache")
                # Remove from cache (implementation depends on cache manager)
                # For now, we'll assume the cache manager handles this
                pass
        
        # Clear alert service internal caches if available
        if self.alert_service and hasattr(self.alert_service, '_quantile_alerts'):
            self.alert_service._quantile_alerts = {}
    
    def get_alert_summary_stats(self, subject_ids=None) -> Dict[str, Any]:
        """
        Get summary statistics for alerts across all subjects
        
        This method provides useful metrics for monitoring the
        overall alert distribution and system health.
        
        Parameters:
            subject_ids (List[str], optional): List of subject IDs to analyze
            
        Returns:
            Dict[str, Any]: Alert summary statistics
        """
        if self.alert_service is None:
            return {
                'error': 'Alert service not initialized',
                'total_subjects': 0,
                'category_counts': {}
            }
        
        try:
            # Get unified alerts for analysis
            alerts = self.get_unified_alerts(subject_ids=subject_ids, use_cache=True)
            
            # Calculate summary statistics
            total_subjects = len(alerts)
            category_counts = {}
            
            for subject_id, alert_data in alerts.items():
                category = alert_data.get('alert_category', 'Unknown')
                category_counts[category] = category_counts.get(category, 0) + 1
            
            # Calculate percentages
            category_percentages = {}
            if total_subjects > 0:
                for category, count in category_counts.items():
                    category_percentages[category] = (count / total_subjects) * 100
            
            return {
                'total_subjects': total_subjects,
                'category_counts': category_counts,
                'category_percentages': category_percentages,
                'categories_found': list(category_counts.keys())
            }
            
        except Exception as e:
            return {
                'error': f'Error calculating alert summary: {str(e)}',
                'total_subjects': 0,
                'category_counts': {}
            }
    
    def validate_alert_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate alert service configuration
        
        This method ensures that alert configuration is valid before
        initializing the alert service.
        
        Parameters:
            config (Dict[str, Any]): Alert configuration to validate
            
        Returns:
            Dict[str, Any]: Validation results with any issues found
        """
        validation_result = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check percentile categories if provided
        if 'percentile_categories' in config:
            categories = config['percentile_categories']
            
            # Expected category keys
            expected_keys = ['SB', 'B', 'N', 'G', 'SG']
            
            for key in expected_keys:
                if key not in categories:
                    validation_result['errors'].append(f"Missing category threshold: {key}")
                    validation_result['valid'] = False
                elif not isinstance(categories[key], (int, float)):
                    validation_result['errors'].append(f"Invalid threshold type for {key}: must be numeric")
                    validation_result['valid'] = False
            
            # Check threshold ordering
            if validation_result['valid']:
                thresholds = [categories[key] for key in expected_keys]
                if thresholds != sorted(thresholds):
                    validation_result['warnings'].append("Category thresholds may not be in expected order")
        
        # Check feature configuration
        if 'feature_config' in config:
            feature_config = config['feature_config']
            if not isinstance(feature_config, dict):
                validation_result['errors'].append("feature_config must be a dictionary")
                validation_result['valid'] = False
        
        return validation_result 