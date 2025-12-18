import os
import json
import logging
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
import requests

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

# Environment variables
GETTY_CLIENT_ID = os.getenv("GETTY_CLIENT_ID")
GETTY_CLIENT_SECRET = os.getenv("GETTY_CLIENT_SECRET")
ASSETS_BUCKET = os.getenv("ASSETS_BUCKET")  # e.g. gs://df-films-assets-euw1

REQUEST_TIMEOUT = 30

def require_env(var_name: str) -> str:
    val = os.getenv(var_name)
    if not val:
        raise HTTPException(status_code=500, detail=f"Missing required environment variable: {var_name}")
    return val

def get_access_token() -> str:
    """Fetch OAuth2 token from Getty API."""
    client_id = require_env("GETTY_CLIENT_ID")
    client_secret = require_env("GETTY_CLIENT_SECRET")
    try:
        resp = requests.post(
            "https://api.gettyimages.com/oauth2/token",
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "client_credentials",
            },
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        logger.error(f"Getty token request failed: {e}")
        raise HTTPException(status_code=502, detail=f"Getty token request failed: {e}")

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Getty token error: {resp.text}")

    token = resp.json().get("access_token")
    if not token:
        raise HTTPException(status_code=502, detail="Getty token response missing access_token")
    return token

def fetch_getty_metadata(asset_id: str) -> dict:
    """Fetch metadata for a Getty asset."""
    token = get_access_token()
    client_id = require_env("GETTY_CLIENT_ID")
    headers = {"Api-Key": client_id, "Authorization": f"Bearer {token}"}
    resp = requests.get(
        f"https://api.gettyimages.com/v3/assets/{asset_id}",
        headers=headers,
        params={"fields": "id,title,caption,keywords,artist,collection_name,date_created,aspect_ratio,clip_length"},
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail=f"Asset {asset_id} not found")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Getty metadata error: {resp.text}")

    asset = resp.json().get("asset")
    if not asset:
        raise HTTPException(status_code=502, detail=f"No asset data returned for id {asset_id}")
    return asset

def build_local_metadata(file_name: str, bucket: str) -> dict:
    """Build minimal metadata for local files."""
    return {
        "title": os.path.basename(file_name),
        "caption": f"Local file {file_name} stored in bucket {bucket}.",
        "keywords": ["local", "video", "sample"]
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "ASSETS_BUCKET": ASSETS_BUCKET,
        "has_getty_client_id": bool(GETTY_CLIENT_ID),
        "has_getty_client_secret": bool(GETTY_CLIENT_SECRET),
    }

@app.post("/validate")
async def validate(req: Request):
    """
    Input modes:
      Getty: {"asset_id": "getty-123456"}
      Local: {"file_name": "raw/CE_025_0.mp4", "bucket": "df-films-assets-euw1"}
    Output: JSON block with asset_id, metadata, and raw path.
    """
    data = await req.json()

    # Mode 1: Getty asset_id
    if "asset_id" in data:
        asset_id = data["asset_id"]
        getty_meta = fetch_getty_metadata(asset_id)
        result = {
            "asset_id": asset_id,
            "getty": getty_meta,
            "paths": {
                "raw": f"{ASSETS_BUCKET}/raw/{asset_id}.mp4" if ASSETS_BUCKET else f"raw/{asset_id}.mp4"
            },
            "status": "validated",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    # Mode 2: Local file_name + bucket
    elif "file_name" in data and "bucket" in data:
        file_name = data["file_name"]
        bucket = data["bucket"]
        asset_id = os.path.splitext(os.path.basename(file_name))[0]
        local_meta = build_local_metadata(file_name, bucket)
        result = {
            "asset_id": asset_id,
            "getty": local_meta,  # reuse same key for consistency
            "paths": {
                "raw": f"gs://{bucket}/{file_name}"
            },
            "status": "validated",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either asset_id (Getty) OR file_name+bucket (local)"
        )

    return result
