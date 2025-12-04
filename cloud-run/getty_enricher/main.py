import os
import requests
from fastapi import FastAPI, Request
from google.cloud import storage
from datetime import datetime

app = FastAPI()

# Environment variables
ASSETS_BUCKET = os.getenv("ASSETS_BUCKET")  # e.g. gs://df-films-assets-euw1
GETTY_CLIENT_ID = os.getenv("GETTY_CLIENT_ID")
GETTY_CLIENT_SECRET = os.getenv("GETTY_CLIENT_SECRET")

# GCS client
storage_client = storage.Client()

def get_access_token():
    """Authenticate with Getty API and return access token."""
    resp = requests.post(
        "https://api.gettyimages.com/oauth2/token",
        data={
            "client_id": GETTY_CLIENT_ID,
            "client_secret": GETTY_CLIENT_SECRET,
            "grant_type": "client_credentials"
        }
    )
    resp.raise_for_status()
    return resp.json()["access_token"]

def download_to_gcs(url: str, asset_id: str):
    """Download Getty asset and store in GCS raw/ folder."""
    bucket_name = ASSETS_BUCKET.replace("gs://", "")
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"raw/{asset_id}.mp4")

    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    blob.upload_from_string(resp.content)

    return f"gs://{bucket_name}/raw/{asset_id}.mp4"

@app.get("/health")
def health():
    return {
        "status": "ok",
        "ASSETS_BUCKET": ASSETS_BUCKET,
        "GETTY_CLIENT_ID": bool(GETTY_CLIENT_ID),
        "GETTY_CLIENT_SECRET": bool(GETTY_CLIENT_SECRET)
    }

@app.post("/validate")
async def validate(req: Request):
    """Fetch Getty metadata, download asset, return JSON block."""
    data = await req.json()
    asset_id = data.get("asset_id")
    if not asset_id:
        return {"error": "asset_id required"}

    token = get_access_token()
    headers = {"Authorization": f"Bearer {token}"}

    # Fetch metadata from Getty
    resp = requests.get(
        f"https://api.gettyimages.com/v3/assets/{asset_id}",
        headers=headers
    )
    resp.raise_for_status()
    asset = resp.json()["asset"]

    # Download file to GCS
    download_url = asset.get("file_download_url")
    gcs_path = download_to_gcs(download_url, asset_id)

    # Build JSON block
    result = {
        "asset_id": asset_id,
        "paths": {
            "raw": gcs_path
        },
        "getty": {
            "title": asset.get("title"),
            "caption": asset.get("caption"),
            "keywords": asset.get("keywords", []),
            "credit_line": asset.get("artist"),
            "download_url": download_url
        },
        "status": "fetched",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    return result
