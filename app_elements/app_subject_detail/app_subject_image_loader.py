import requests
import base64
import pandas as pd
import numpy as np
import sys

class AppSubjectImageLoader:
    def __init__(self):
        """Initialize image loader"""
        print("Initializing AppSubjectImageLoader")
        
    def get_s3_public_url(self, subject_id, session_date, nwb_suffix, figure_suffix="choice_history.png",
                         result_path="foraging_nwb_bonsai_processed", bucket_name="aind-behavior-data"):
        """
        Generate a public S3 URL for accessing a specific figure
        """
        print(f"Building S3 URL: subject={subject_id}, date={session_date}, nwb_suffix={nwb_suffix}, figure={figure_suffix}")
        
        try:
            # Handle NWB suffix properly - this was the key fix
            if pd.isna(nwb_suffix) or nwb_suffix == 0:
                nwb_suffix_str = ""
                print("Using empty nwb_suffix")
            else:
                nwb_suffix_str = f"_{int(nwb_suffix)}"
                print(f"Using nwb_suffix: {nwb_suffix_str}")
            
            # Build the URL with the correct format
            url = (
                f"https://{bucket_name}.s3.us-west-2.amazonaws.com/{result_path}/"
                f"{subject_id}_{session_date}{nwb_suffix_str}/{subject_id}_{session_date}{nwb_suffix_str}_{figure_suffix}"
            )
            
            print(f"Generated URL: {url}")
            return url
            
        except Exception as e:
            print(f"ERROR generating URL: {str(e)}", file=sys.stderr)
            import traceback
            print(traceback.format_exc(), file=sys.stderr)
            raise

    def get_session_image(self, subject_id, session_date, nwb_suffix, figure_suffix="choice_history.png"):
        """
        Fetches an image from S3 based on subject and session information
        """
        print(f"\n=== IMAGE REQUEST DEBUG ===")
        print(f"Requesting image for subject={subject_id}, date={session_date}, suffix={figure_suffix}")
        
        try:
            # Generate the S3 URL
            url = self.get_s3_public_url(
                subject_id=subject_id,
                session_date=session_date,
                nwb_suffix=nwb_suffix,
                figure_suffix=figure_suffix
            )
            
            # Verify URL is valid for debugging
            print(f"Testing URL accessibility: {url}")
            try:
                # Just do a HEAD request to check if the URL is valid without downloading the image
                head_response = requests.head(url, timeout=1)
                if head_response.status_code == 200:
                    print(f"URL is accessible (status code 200)")
                else:
                    print(f"URL may not be accessible - status code: {head_response.status_code}")
            except Exception as req_err:
                print(f"Warning: Could not verify URL accessibility: {str(req_err)}")
            
            # Return the direct URL for the image
            return {
                'success': True,
                'image_src': url,
                'url': url,
                'error': None
            }
                
        except Exception as e:
            error_msg = f"Error generating image URL: {str(e)}"
            print(f"ERROR: {error_msg}", file=sys.stderr)
            import traceback
            print(traceback.format_exc(), file=sys.stderr)
            return {
                'success': False,
                'image_src': None,
                'url': None,
                'error': error_msg
            }