import pandas as pd
from typing import Dict, Any, Optional, List

class SubjectTooltipService:
    """ Service for retrieving and preparing data for subject tooltips """

    def __init__(self, dataframe=None):
        """ Initialize the service with a dataframe """
        self.dataframe = dataframe

    def update_dataframe(self, dataframe):
        """ Update the dataframe when table data changes"""
        self.dataframe = dataframe

    def get_subject_alert_data(self, subject_id: str) -> Dict[str, Any]:
        """ Get alert information for a subject
        
        Parameters:
            subject_id (str): The ID of the subject to get alert data for

        Returns:
            Dict[str, Any]: A dictionary containing alert information
        """
        if self.dataframe is None or subject_id not in self.dataframe['subject_id'].values:
            return {
                'percentile_category': 'NS',
                'threshold_alert': 'N',
                'combined_alert': 'NS'
            }

        # Filter dataframe for the specific subject
        subject_row = self.dataframe[self.dataframe['subject_id'] == subject_id].iloc[0]

        # Extract alert data
        alert_data = {
            'percentile_category': subject_row.get('percentile_category', 'NS'),
            'threshold_alert': subject_row.get('threshold_alert', 'N'),
            'combined_alert': subject_row.get('combined_alert', 'NS')
        }

        return alert_data
    
    
        
            