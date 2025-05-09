import requests
import base64
import pandas as pd
import numpy as np

class AppSubjectImageLoader:
    def __init__(self):
        """Initialize image loader"""
        pass
        
    def get_s3_public_url(self, subject_id, session_date, nwb_suffix, figure_suffix="choice_history.png",
                         result_path="foraging_nwb_bonsai_processed", bucket_name="aind-behavior-data"):
        """
        Generate a public S3 URL for accessing a specific figure
        """
        # Handle NWB suffix properly - this was the key fix
        nwb_suffix_str = "" if (pd.isna(nwb_suffix) or nwb_suffix == 0) else f"_{int(nwb_suffix)}"
        
        # Build the URL with the correct format
        url = (
            f"https://{bucket_name}.s3.us-west-2.amazonaws.com/{result_path}/"
            f"{subject_id}_{session_date}{nwb_suffix_str}/{subject_id}_{session_date}{nwb_suffix_str}_{figure_suffix}"
        )
        
        return url

    def get_session_image(self, subject_id, session_date, nwb_suffix, figure_suffix="choice_history.png"):
        """
        Fetches an image from S3 based on subject and session information
        """
        try:
            # Generate the S3 URL
            url = self.get_s3_public_url(
                subject_id=subject_id,
                session_date=session_date,
                nwb_suffix=nwb_suffix,
                figure_suffix=figure_suffix
            )
            
            # Return the direct URL for the image
            return {
                'success': True,
                'image_src': url,
                'url': url,
                'error': None
            }
                
        except Exception as e:
            return {
                'success': False,
                'image_src': None,
                'url': None,
                'error': f"Error generating image URL: {str(e)}"
            }