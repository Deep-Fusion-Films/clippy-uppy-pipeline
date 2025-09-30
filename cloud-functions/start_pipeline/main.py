import requests
import json

def start_pipeline(event, context):
"""
This function is triggered when a new video file is uploaded to Cloud Storage.
It sends the file information to the first Cloud Run service to start the pipeline.
"""

# Get the file name and bucket name from the event
  file_name = event['name']
  bucket_name = event['bucket']

# Prepare the data to send to the Cloud Run service
  payload = {
      "file_name": file_name,
      "bucket": bucket_name
    }

# URL of your Cloud Run service (replace with your actual endpoint)
  cloud_run_url = "https://transcode-service-abc123-uk.a.run.app/start"

  try:
# Send the data to Cloud Run using an HTTP POST request
    response = requests.post(cloud_run_url, json=payload)

# Check if the request was successful
  if response.status_code == 200:
        print(f"✅ Pipeline started for {file_name}")
  else:
        print(f"⚠️ Failed to start pipeline: {response.status_code} - {response.text}")

  except Exception as e:
      print(f"❌ Error sending request to Cloud Run: {str(e)}")
