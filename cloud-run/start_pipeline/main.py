import os
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Cloud Run endpoint for the transcode service
TRANSCODE_URL = os.getenv("TRANSCODE_URL", "https://transcode-service-xyz.a.run.app/transcode")

def start_pipeline(event, context):
    """
    Triggered by a file upload to the getty-ingest-media bucket.
    Sends the file name and bucket to the transcode service to begin the pipeline.
    """

    file_name = event.get("name")
    bucket = event.get("bucket")

    if not file_name or not bucket:
        logging.error("Missing file name or bucket in event payload.")
        return

    payload = {
        "file_name": file_name,
        "bucket": bucket
    }

    try:
        response = requests.post(TRANSCODE_URL, json=payload, timeout=10)
        response.raise_for_status()
        logging.info(f"Pipeline triggered for {file_name}. Status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to trigger transcode service: {e}")
