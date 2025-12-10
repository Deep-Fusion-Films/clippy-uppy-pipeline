import os
import requests
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any
from fastapi import FastAPI, Request, HTTPException

app = FastAPI()

REQUEST_TIMEOUT = 30  # seconds
logger = logging.getLogger("uvicorn.error")

_seen_ids: set[str] = set()  # track IDs we've already used


def require_env(var_name: str) -> str:
    val = os.getenv(var_name)
    if not val:
        raise HTTPException(status_code=500, detail=f"Missing required environment variable: {var_name}")
    return val


def get_storage_client():
    from google.cloud import storage
    return storage.Client()


def get_firestore_client():
    from google.cloud import firestore
    return firestore.Client()


def get_access_token() -> str:
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


def search_assets(count: int = 2) -> List[str]:
    """Search Getty videos and return only fresh, accessible IDs with download URLs."""
    token = get_access_token()
    client_id = require_env("GETTY_CLIENT_ID")
    headers = {"Api-Key": client_id, "Authorization": f"Bearer {token}"}

    ids: List[str] = []
    page = 1

    while len(ids) < count:
        resp = requests.get(
            "https://api.gettyimages.com/v3/search/videos",
            headers=headers,
            params={
                "page_size": count * 2,
                "page": page,
                "fields": "id,title,caption,file_download_url"
            },
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"Getty search error: {resp.text}")

        videos = resp.json().get("videos", [])
        for v in videos:
            vid = v.get("id")
            if vid and v.get("file_download_url") and vid not in _seen_ids:
                ids.append(vid)
                _seen_ids.add(vid)
                if len(ids) >= count:
                    break
        page += 1

    return ids


def already_processed(asset_id: str) -> bool:
    db = get_firestore_client()
    doc = db.collection("assets").document(asset_id).get()
    return doc.exists


def fetch_asset_metadata(asset_id: str) -> Dict[str, Any]:
    token = get_access_token()
    client_id = require_env("GETTY_CLIENT_ID")
    headers = {"Api-Key": client_id, "Authorization": f"Bearer {token}"}

    resp = requests.get(
        f"https://api.gettyimages.com/v3/assets/{asset_id}",
        headers=headers,
        params={
            "fields": "id,title,caption,keywords,artist,asset_family,allowed_use,usage_restrictions,"
                      "collection_name,date_created,aspect_ratio,clip_length,editorial_segments,"
                      "referral_destinations,preview,thumb,file_download_url"
        },
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail=f"Asset {asset_id} not found or inaccessible")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Getty asset error: {resp.text}")

    asset = resp.json().get("asset")
    if not asset:
        raise HTTPException(status_code=404, detail=f"No asset data returned for id {asset_id}")
    return asset


def download_to_gcs(url: str, asset_id: str) -> str:
    """Stream download directly into GCS to avoid buffering large files in memory."""
    bucket_uri = require_env("ASSETS_BUCKET")
    bucket_name = bucket_uri.replace("gs://", "")
    storage_client = get_storage_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"raw/{asset_id}.mp4")

    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as resp:
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"Download error: {resp.text}")
        blob.upload_from_file(resp.raw, rewind=True)

    return f"gs://{bucket_name}/raw/{asset_id}.mp4"


def write_stub_record(asset_id: str, getty: Dict[str, Any], gcs_path: str) -> None:
    db = get_firestore_client()
    db.collection("assets").document(asset_id).set(
        {
            "status": "processed",
            "paths": {"raw": gcs_path},
            "getty": getty,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "has_assets_bucket": bool(os.getenv("ASSETS_BUCKET")),
        "has_getty_client_id": bool(os.getenv("GETTY_CLIENT_ID")),
        "has_getty_client_secret": bool(os.getenv("GETTY_CLIENT_SECRET")),
    }


@app.post("/validate")
async def validate(req: Request):
    body = await req.json()
    try:
        count = int(body.get("count", 2))  # keep small to avoid timeout
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid 'count' value. Must be an integer.")

    asset_ids = search_assets(count=count)

    async def process_asset(asset_id: str):
        if already_processed(asset_id):
            return {
                "asset_id": asset_id,
                "status": "skipped_duplicate",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        try:
            asset = fetch_asset_metadata(asset_id)
            download_url = asset.get("file_download_url")
            if not download_url:
                return {
                    "asset_id": asset_id,
                    "status": "no_download_url",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
            gcs_path = download_to_gcs(download_url, asset_id)
            write_stub_record(asset_id, asset, gcs_path)
            return {
                "asset_id": asset_id,
                "paths": {"raw": gcs_path},
                "getty": asset,
                "status": "fetched",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        except Exception as e:
            logger.error(f"Error processing {asset_id}: {e}")
            return {
                "asset_id": asset_id,
                "status": "error",
                "detail": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

    # Run asset processing concurrently
    results = await asyncio.gather(*(process_asset(aid) for aid in asset_ids))
    return {"assets": results}
