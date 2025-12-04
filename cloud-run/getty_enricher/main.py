import os
import requests
from fastapi import FastAPI, Request, HTTPException
from google.cloud import storage
from datetime import datetime

app = FastAPI()

# Environment variables (mounted from Secret Manager in Cloud Run)
ASSETS_BUCKET = os.getenv("ASSETS_BUCKET")  # e.g. gs://df-films-assets-euw1
GETTY_CLIENT_ID = os.getenv("GETTY_CLIENT_ID")
GETTY_CLIENT_SECRET = os.getenv("GETTY_CLIENT_SECRET")

# GCS client
storage_client = storage.Client()

def get_access_token() -> str:
    """
    Always fetch a fresh OAuth2 token from Getty.
    Tokens expire after ~3600s, so this guarantees validity.
    """
    resp = requests.post(
        "https://api.gettyimages.com/oauth2/token",
        data={
            "client_id": GETTY_CLIENT_ID,
            "client_secret": GETTY_CLIENT_SECRET,
            "grant_type": "client_credentials"
        }
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Getty token error: {resp.text}")
    return resp.json()["access_token"]

def download_to_gcs(url: str, asset_id: str) -> str:
    """
    Download Getty asset and store in GCS raw/ folder.
    """
    bucket_name = ASSETS_BUCKET.replace("gs://", "")
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"raw/{asset_id}.mp4")

    resp = requests.get(url, stream=True)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Download error: {resp.text}")

    blob.upload_from_string(resp.content)
    return f"gs://{bucket_name}/raw/{asset_id}.mp4"

@app.get("/health")
def health():
    """
    Health check endpoint to confirm env vars and bucket wiring.
    """
    return {
        "status": "ok",
        "ASSETS_BUCKET": ASSETS_BUCKET,
        "GETTY_CLIENT_ID": bool(GETTY_CLIENT_ID),
        "GETTY_CLIENT_SECRET": bool(GETTY_CLIENT_SECRET)
    }

@app.post("/validate")
async def validate(req: Request):
    """
    Validate a Getty asset by ID:
      1. Fetch metadata from Getty API
      2. Download the file into GCS
      3. Return structured JSON block
    """
    data = await req.json()
    asset_id = data.get("asset_id")
    if not asset_id:
        raise HTTPException(status_code=400, detail="asset_id required")

    # Fetch fresh token
    token = get_access_token()
    headers = {
        "Api-Key": GETTY_CLIENT_ID,
        "Authorization": f"Bearer {token}"
    }

    # Fetch metadata from Getty
    resp = requests.get(f"https://api.gettyimages.com/v3/assets/{asset_id}", headers=headers)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Getty asset error: {resp.text}")

    asset = resp.json().get("asset")
    if not asset:
        raise HTTPException(status_code=500, detail="No asset data returned from Getty")

    # Download file to GCS
    download_url = asset.get("file_download_url")
    if not download_url:
        raise HTTPException(status_code=500, detail="No download URL in Getty metadata")

    gcs_path = download_to_gcs(download_url, asset_id)

    # Build JSON block
    return {
        "asset_id": asset_id,
        "paths": {"raw": gcs_path},
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
