"""
Subject image loader for AIND Dashboard

This module handles image URL generation and loading for subject session data.
"""

import pandas as pd
import numpy as np
import sys
import traceback
from app_utils.simple_logger import get_logger

logger = get_logger('subject_image_loader')

class AppSubjectImageLoader:
    def __init__(self):
        """Initialize image loader"""
        logger.info("Initializing AppSubjectImageLoader")
        
    def get_s3_public_url(self, subject_id, session_date, nwb_suffix, figure_suffix="choice_history.png",
                         result_path="foraging_nwb_bonsai_processed", bucket_name="aind-behavior-data"):
        """
        Generate a public S3 URL for accessing a specific figure
        """
        logger.info(f"Building S3 URL: subject={subject_id}, date={session_date}, nwb_suffix={nwb_suffix}, figure={figure_suffix}")
        
        try:
            # Handle NWB suffix properly - this was the key fix
            if pd.isna(nwb_suffix) or nwb_suffix == 0:
                nwb_suffix_str = ""
                logger.info("Using empty nwb_suffix")
            else:
                nwb_suffix_str = f"_{int(nwb_suffix)}"
                logger.info(f"Using nwb_suffix: {nwb_suffix_str}")
            
            # Build the URL with the correct format
            url = (
                f"https://{bucket_name}.s3.us-west-2.amazonaws.com/{result_path}/"
                f"{subject_id}_{session_date}{nwb_suffix_str}/{subject_id}_{session_date}{nwb_suffix_str}_{figure_suffix}"
            )
            
            logger.info(f"Generated URL: {url}")
            return url
            
        except Exception as e:
            logger.error(f"ERROR generating URL: {str(e)}")
            print(traceback.format_exc(), file=sys.stderr)
            raise