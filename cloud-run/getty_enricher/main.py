import os
import requests
import logging
import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request, HTTPException

app = FastAPI()

REQUEST_TIMEOUT = 30  # seconds for external calls
MAX_DOWNLOAD_RETRIES = 5
logger = logging.getLogger("uvicorn.error")

# Track IDs we've already processed during container lifetime to avoid repeats
_seen_ids: set[str] = set()


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

    token = resp.json().get("access_token")
    if not token:
        logger.error("Getty token response missing access_token")
        raise HTTPException(status_code=502, detail="Getty token response missing access_token")
    return token


def already_in_firestore(asset_id: str) -> bool:
    db = get_firestore_client()
    doc = db.collection("assets").document(asset_id).get()
    return doc.exists


def search_downloadable_asset_ids(count: int) -> List[str]:
    """
    Return exactly 'count' unique asset IDs that have a file_download_url and are not already processed.
    Keeps paging until it accumulates the requested number.
    """
    token = get_access_token()
    client_id = require_env("GETTY_CLIENT_ID")
    headers = {"Api-Key": client_id, "Authorization": f"Bearer {token}"}

    ids: List[str] = []
    page = 1

    while len(ids) < count:
        try:
            resp = requests.get(
                "https://api.gettyimages.com/v3/search/videos",
                headers=headers,
                params={
                    "page_size": max(50, count * 3),  # bias toward larger pages to find enough valid assets quickly
                    "page": page,
                    "fields": "id,file_download_url",
                },
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as e:
            logger.warning(f"Getty search failed on page {page}: {e}")
            # brief backoff then continue paging
            time.sleep(1)
            continue

        if resp.status_code != 200:
            logger.warning(f"Getty search error {resp.status_code}: {resp.text}")
            time.sleep(1)
            page += 1
            continue

        videos = resp.json().get("videos", [])
        for v in videos:
            vid = v.get("id")
            url = v.get("file_download_url")
            if not vid or not url:
                continue
            if vid in _seen_ids:
                continue
            if already_in_firestore(vid):
                # skip duplicates that have been processed earlier
                _seen_ids.add(vid)
                continue

            ids.append(vid)
            _seen_ids.add(vid)
            if len(ids) >= count:
                break

        page += 1

    return ids


def fetch_asset_metadata(asset_id: str) -> Dict[str, Any]:
    token = get_access_token()
    client_id = require_env("GETTY_CLIENT_ID")
    headers = {"Api-Key": client_id, "Authorization": f"Bearer {token}"}

    # Simple retry for metadata requests
    for attempt in range(3):
        try:
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
        except requests.RequestException as e:
            logger.warning(f"Metadata request error for {asset_id} (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
            continue

        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Asset {asset_id} not found or inaccessible")
        if resp.status_code != 200:
            logger.warning(f"Getty asset error {resp.status_code} for {asset_id}: {resp.text}")
            time.sleep(2 ** attempt)
            continue

        asset = resp.json().get("asset")
        if not asset:
            logger.warning(f"No asset data returned for id {asset_id} (attempt {attempt+1})")
            time.sleep(2 ** attempt)
            continue

        return asset

    raise HTTPException(status_code=502, detail=f"Failed to fetch metadata for {asset_id} after retries")


def download_to_gcs_with_retries(url: str, asset_id: str, attempts: int = MAX_DOWNLOAD_RETRIES) -> str:
    """
    Stream download directly into GCS with retries to guarantee success.
    """
    bucket_uri = require_env("ASSETS_BUCKET")
    bucket_name = bucket_uri.replace("gs://", "")
    storage_client = get_storage_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"raw/{asset_id}.mp4")

    last_err: Optional[str] = None
    for attempt in range(attempts):
        try:
            with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as resp:
                if resp.status_code != 200:
                    last_err = f"Download HTTP {resp.status_code}: {resp.text}"
                    logger.warning(f"Download failed for {asset_id} (attempt {attempt+1}): {last_err}")
                    time.sleep(2 ** attempt)
                    continue
                # Stream to GCS without buffering entire file in memory
                blob.upload_from_file(resp.raw, rewind=True)
                return f"gs://{bucket_name}/raw/{asset_id}.mp4"
        except requests.RequestException as e:
            last_err = str(e)
            logger.warning(f"Network error during download for {asset_id} (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)

    raise HTTPException(status_code=502, detail=f"Download failed for {asset_id} after retries: {last_err}")


def write_record(asset_id: str, getty: Dict[str, Any], gcs_path: str) -> None:
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
    """
    Strict guarantee:
    - Only select assets that have a download URL and are not already processed.
    - Fetch full metadata + download to GCS (with retries).
    - Write Firestore record ONLY when both steps succeed.
    - Return exactly 'count' successful assets.
    """
    body = await req.json()
    try:
        count = int(body.get("count", 2))  # keep small to avoid timeouts; caller can raise if needed
        if count <= 0:
            raise ValueError()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid 'count' value. Must be a positive integer.")

    successes: List[Dict[str, Any]] = []

    while len(successes) < count:
        needed = count - len(successes)
        # Fetch enough candidate IDs that are guaranteed downloadable at source
        candidate_ids = search_downloadable_asset_ids(needed)

        async def process_one(asset_id: str) -> Dict[str, Any]:
            # If a race condition caused prior processing, skip and let the outer loop refill
            if already_in_firestore(asset_id):
                return {"status": "skipped_duplicate"}

            # Step 1: metadata
            asset = fetch_asset_metadata(asset_id)
            download_url = asset.get("file_download_url")
            if not download_url:
                # Shouldn't happen because we filter at search, but enforce anyway
                raise HTTPException(status_code=502, detail=f"No download URL for {asset_id}")

            # Step 2: download with retries
            gcs_path = download_to_gcs_with_retries(download_url, asset_id)

            # Step 3: write Firestore only after both metadata + download succeed
            write_record(asset_id, asset, gcs_path)

            return {
                "asset_id": asset_id,
                "paths": {"raw": gcs_path},
                "getty": asset,
                "status": "fetched",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

        # Process the batch concurrently
        batch_results = await asyncio.gather(
            *(process_one(aid) for aid in candidate_ids),
            return_exceptions=True,
        )

        # Collect successes; on errors, just continue the loop to fetch more IDs
        for res in batch_results:
            if isinstance(res, dict) and res.get("status") == "fetched":
                successes.append(res)

        # Defensive: prevent runaway loops if upstream is starved
        if len(candidate_ids) == 0:
            raise HTTPException(status_code=502, detail="No downloadable assets available from Getty at this time.")

    return {"assets": successes[:count]}

