import os
import random
import requests
from fastapi import FastAPI, Request, HTTPException
from google.cloud import storage, firestore
from datetime import datetime

app = FastAPI()

# Environment variables (mounted from Secret Manager in Cloud Run)
ASSETS_BUCKET = os.getenv("ASSETS_BUCKET")  # e.g. gs://df-films-assets-euw1
GETTY_CLIENT_ID = os.getenv("GETTY_CLIENT_ID")
GETTY_CLIENT_SECRET = os.getenv("GETTY_CLIENT_SECRET")

def get_storage_client():
    """Create a new GCS client when needed."""
    return storage.Client()

def get_firestore_client():
    """Create a new Firestore client when needed."""
    return firestore.Client()

def get_access_token() -> str:
    """Always fetch a fresh OAuth2 token from Getty."""
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

def already_processed(asset_id: str) -> bool:
    """Check Firestore to see if asset_id has already been processed."""
    db = get_firestore_client()
    return db.collection("assets").document(asset_id).get().exists

def download_to_gcs(url: str, asset_id: str) -> str:
    """Download Getty asset and store in GCS raw/ folder."""
    storage_client = get_storage_client()
    bucket_name = ASSETS_BUCKET.replace("gs://", "")
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"raw/{asset_id}.mp4")

    resp = requests.get(url, stream=True)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Download error: {resp.text}")

    blob.upload_from_string(resp.content)
    return f"gs://{bucket_name}/raw/{asset_id}.mp4"

def search_assets(count: int = 10) -> list:
    """Search Getty API for a random batch of assets (no keyword)."""
    token = get_access_token()
    headers = {
        "Api-Key": GETTY_CLIENT_ID,
        "Authorization": f"Bearer {token}"
    }
    page = random.randint(1, 100)  # randomize to avoid same results
    resp = requests.get(
        f"https://api.gettyimages.com/v3/search/videos?page_size={count}&page={page}",
        headers=headers
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Getty search error: {resp.text}")
    return [video["id"] for video in resp.json().get("videos", [])]

@app.get("/health")
def health():
    """Health check endpoint to confirm env vars and bucket wiring."""
    return {
        "status": "ok",
        "ASSETS_BUCKET": ASSETS_BUCKET,
        "GETTY_CLIENT_ID": bool(GETTY_CLIENT_ID),
        "GETTY_CLIENT_SECRET": bool(GETTY_CLIENT_SECRET)
    }

@app.post("/validate")
async def validate(req: Request):
    """
    Auto-fetch Getty assets:
      1. Search Getty for N random assets (default 10)
      2. Skip duplicates already in Firestore
      3. Fetch metadata + download into GCS
      4. Return structured JSON blocks
    """
    data = await req.json()
    count = int(data.get("count", 10))

    asset_ids = search_assets(count=count)
    results = []

    db = get_firestore_client()

    for asset_id in asset_ids:
        if already_processed(asset_id):
            continue  # skip duplicates

        token = get_access_token()
        headers = {
            "Api-Key": GETTY_CLIENT_ID,
            "Authorization": f"Bearer {token}"
        }
        resp = requests.get(f"https://api.gettyimages.com/v3/assets/{asset_id}", headers=headers)
        if resp.status_code != 200:
            continue
        asset = resp.json().get("asset")
        if not asset:
            continue

        download_url = asset.get("file_download_url")
        if not download_url:
            continue

        gcs_path = download_to_gcs(download_url, asset_id)

        # Write stub record into Firestore immediately
        db.collection("assets").document(asset_id).set({
            "status": "processed",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })

        results.append({
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
        })

    return {"assets": results}
