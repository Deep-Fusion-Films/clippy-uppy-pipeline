import os
import random
import requests
import logging
from datetime import datetime
from typing import List, Dict, Any

from fastapi import FastAPI, Request, HTTPException

app = FastAPI()

# Environment variables (mounted via Cloud Run -> Variables & Secrets)
ASSETS_BUCKET = os.getenv("ASSETS_BUCKET")  # e.g. "gs://df-films-assets-euw1"
GETTY_CLIENT_ID = os.getenv("GETTY_CLIENT_ID")
GETTY_CLIENT_SECRET = os.getenv("GETTY_CLIENT_SECRET")

REQUEST_TIMEOUT = 30  # seconds

# Use uvicorn's logger so messages appear in Cloud Run logs
logger = logging.getLogger("uvicorn.error")


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
        logger.error(f"Getty token error {resp.status_code}: {resp.text}")
        raise HTTPException(status_code=resp.status_code, detail=f"Getty token error: {resp.text}")

    data = resp.json()
    token = data.get("access_token")
    if not token:
        logger.error("Getty token response missing access_token")
        raise HTTPException(status_code=502, detail="Getty token response missing access_token")
    return token


def search_assets(count: int = 10) -> List[str]:
    token = get_access_token()
    client_id = require_env("GETTY_CLIENT_ID")

    headers = {"Api-Key": client_id, "Authorization": f"Bearer {token}"}
    page = random.randint(1, 100)

    try:
        resp = requests.get(
            "https://api.gettyimages.com/v3/search/videos",
            headers=headers,
            params={
                "page_size": count,
                "page": page,
                "fields": "id,title,caption,file_download_url"
            },
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        logger.error(f"Getty search request failed: {e}")
        raise HTTPException(status_code=502, detail=f"Getty search request failed: {e}")

    if resp.status_code != 200:
        logger.error(f"Getty search error {resp.status_code}: {resp.text}")
        raise HTTPException(status_code=resp.status_code, detail=f"Getty search error: {resp.text}")

    videos = resp.json().get("videos", [])
    return [v.get("id") for v in videos if v.get("id")]


def already_processed(asset_id: str) -> bool:
    db = get_firestore_client()
    doc = db.collection("assets").document(asset_id).get()
    return doc.exists


def fetch_asset_metadata(asset_id: str) -> Dict[str, Any]:
    token = get_access_token()
    client_id = require_env("GETTY_CLIENT_ID")
    headers = {"Api-Key": client_id, "Authorization": f"Bearer {token}"}

    try:
        resp = requests.get(
            f"https://api.gettyimages.com/v3/assets/{asset_id}",
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        logger.error(f"Getty asset request failed for {asset_id}: {e}")
        raise HTTPException(status_code=502, detail=f"Getty asset request failed: {e}")

    if resp.status_code == 404:
        logger.warning(f"Getty asset {asset_id} not found or inaccessible")
        raise HTTPException(status_code=404, detail=f"Asset {asset_id} not found or inaccessible")

    if resp.status_code != 200:
        logger.error(f"Getty asset fetch failed for {asset_id}: {resp.status_code} {resp.text}")
        raise HTTPException(status_code=resp.status_code, detail=f"Getty asset error: {resp.text}")

    asset = resp.json().get("asset")
    if not asset:
        logger.error(f"No asset data returned for id {asset_id}")
        raise HTTPException(status_code=404, detail=f"No asset data returned for id {asset_id}")
    return asset


def download_to_gcs(url: str, asset_id: str) -> str:
    bucket_uri = require_env("ASSETS_BUCKET")
    bucket_name = bucket_uri.replace("gs://", "")
    storage_client = get_storage_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"raw/{asset_id}.mp4")

    try:
        resp = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        logger.error(f"Download request failed for {asset_id}: {e}")
        raise HTTPException(status_code=502, detail=f"Download request failed: {e}")

    if resp.status_code != 200:
        logger.error(f"Download error {resp.status_code} for {asset_id}: {resp.text}")
        raise HTTPException(status_code=resp.status_code, detail=f"Download error: {resp.text}")

    blob.upload_from_string(resp.content)
    return f"gs://{bucket_name}/raw/{asset_id}.mp4"


def write_stub_record(asset_id: str, getty: Dict[str, Any], gcs_path: str) -> None:
    db = get_firestore_client()
    db.collection("assets").document(asset_id).set(
        {
            "status": "processed",
            "paths": {"raw": gcs_path},
            "getty": {
                "title": getty.get("title"),
                "caption": getty.get("caption"),
                "keywords": getty.get("keywords", []),
                "credit_line": getty.get("artist"),
            },
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "has_assets_bucket": bool(ASSETS_BUCKET),
        "has_getty_client_id": bool(GETTY_CLIENT_ID),
        "has_getty_client_secret": bool(GETTY_CLIENT_SECRET),
    }


@app.post("/validate")
async def validate(req: Request):
    body = await req.json()
    try:
        count = int(body.get("count", 10))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid 'count' value. Must be an integer.")

    asset_ids = search_assets(count=count)
    results: List[Dict[str, Any]] = []

    for asset_id in asset_ids:
        if already_processed(asset_id):
            results.append({
                "asset_id": asset_id,
                "status": "skipped_duplicate",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            })
            continue

        try:
            asset = fetch_asset_metadata(asset_id)
        except HTTPException as e:
            results.append({
                "asset_id": asset_id,
                "status": "metadata_error",
                "detail": e.detail,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            })
            continue

        download_url = asset.get("file_download_url")
        if not download_url:
            results.append({
                "asset_id": asset_id,
                "status": "no_download_url",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            })
            continue

        try:
            gcs_path = download_to_gcs(download_url, asset_id)
        except HTTPException as e:
            results.append({
                "asset_id": asset_id,
                "status": "download_error",
                "detail": e.detail,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            })
            continue

        try:
            write_stub_record(asset_id, asset, gcs_path)
        except Exception as e:
            logger.error(f"Firestore write failed for {asset_id}: {e}")
            results.append({
                "asset_id": asset_id,
                "paths": {"raw": gcs_path},
                "status": "firestore_error",
                "detail": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            })
            continue

        results.append({
            "asset_id": asset_id,
            "paths": {"raw": gcs_path},
            "getty": {
                "title": asset.get("title"),
                "caption": asset.get("caption"),
                "keywords": asset.get("keywords", []),
                "credit_line": asset.get("artist"),
                "download_url": download_url,
