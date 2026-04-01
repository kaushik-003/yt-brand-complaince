'''
Connector between Python service code and Azure Video Indexer APIs.
'''

import logging
import os
import time

import requests
import yt_dlp
from azure.identity import DefaultAzureCredential

logger = logging.getLogger("video-indexer")

class VideoIndexerService:
    def __init__(self):
        self.subscription_id = os.getenv("AZURE_VIDEO_INDEXER_SUBSCRIPTION_ID")
        self.resource_group = os.getenv("AZURE_VIDEO_INDEXER_RESOURCE_GROUP")
        self.account_id = os.getenv("AZURE_VIDEO_INDEXER_ACCOUNT_ID")
        self.location = os.getenv("AZURE_VIDEO_INDEXER_LOCATION", "trial")
        self.vi_name = os.getenv("AZURE_VI_NAME", "brandgaurd-video")
        self.credential = DefaultAzureCredential()

    def get_access_token(self):
        '''
        Generates an ARM Access token for authenticating with Azure Video Indexer API
        '''
        try:
            token_obj = self.credential.get_token("https://management.azure.com/.default")
            return token_obj.token
        except Exception as e:
            logger.error(f"Error obtaining access token: {e}")
            raise

    def get_account_token(self, arm_access_token):
        '''
        Gets the account access token required for subsequent API calls to Azure Video Indexer
        '''
        url = (
            f"https://management.azure.com/subscriptions/{self.subscription_id}"
            f"/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.VideoIndexer/accounts/{self.vi_name}"
            f"/generateAccountAccessToken?api-version=2024-12-01-preview"
        )
        headers = {"Authorization": f"Bearer {arm_access_token}"}
        payload = {"permissionType": "Contributor", "scope" : "Account"}
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"Failed to get account access token: {response.text}")
        return response.json().get("accessToken")
    
    def download_youtube_video(self, url, output_path="temp_video.mp4"):
        '''
        Downloads a YouTube video using yt_dlp and saves it to the specified output path.
        '''
        logger.info(f"Downloading video from YouTube URL: {url}")

        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': output_path,
            'quiet': True,
            'overwrite': True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            logger.info(f"Video downloaded successfully and saved to: {output_path}")
            return output_path
        except Exception as e:
            raise Exception(f"Error downloading video: {e}")
        
    def upload_video_to_azure(self, video_path, video_name):
        '''
        Uploads the video to Azure Video Indexer and returns the video ID.
        '''
        arm_token = self.get_access_token()
        account_token = self.get_account_token(arm_token)

        api_url = f"https://api.videoindexer.ai/{self.location}/Accounts/{self.account_id}/Videos"

        params = {
            "name": video_name,
            "privacy": "Private",
            "accessToken": account_token,
            "indexingPreset" : "Default",
        }

        logger.info(f"Uploading file {video_path} to Azure Video Indexer with name {video_name}")

        # open the video file in binary mode 
        with open(video_path, "rb") as video_file:
            files = {"file": video_file}
            response = requests.post(api_url, params=params, files=files)

        if response.status_code != 200:
            raise Exception(f"Failed to upload video: {response.text}")

        video_id = response.json().get("id")
        logger.info(f"Video uploaded successfully with ID: {video_id}")
        return video_id
    
    def wait_for_video_processing(self, video_id):
        logger.info(f"Waiting for video processing to complete for video ID: {video_id}")
        while True:
            arm_token = self.get_access_token()
            account_token = self.get_account_token(arm_token)

            url = f"https://api.videoindexer.ai/{self.location}/Accounts/{self.account_id}/Videos/{video_id}/Index"
            params = {
                "accessToken": account_token,
            }
            response = requests.get(url, params=params)
            data = response.json()

            state = data.get("state")
            if state == "Processed":
                logger.info("Video processing completed.")
                return data
            elif state == "Failed":
                raise Exception("Video processing failed.")
            elif state == "Quarantined":
                raise Exception("Video is quarantined due to policy violation.")
            logger.info(f"Status: {state}. Waiting for 30 seconds before checking again...")
            time.sleep(30)

    def extract_data(self, vi_json):
        'parses the JSON into our state format'
        transcript_lines = []
        for v in vi_json.get("videos", []):
            for insight in v.get("insights", {}).get("transcript", []):
                text = insight.get("text")
                if text:
                    transcript_lines.append(text)
        ocr_texts = []
        for v in vi_json.get("videos", []):
            for insight in v.get("insights", {}).get("ocr", []):
                text = insight.get("text")
                if text:
                    ocr_texts.append(text)
        return {
            "transcript": "\n".join(transcript_lines),
            "ocr_text": ocr_texts,
            "video_metadata": {
                "duration": vi_json.get("summarizedInsights", {}).get("duration"),
                "platform": "youtube",
            }
        }

        