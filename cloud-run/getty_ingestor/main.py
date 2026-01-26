import os
import json
import random
from datetime import datetime

import requests
from fastapi import FastAPI, HTTPException
from google.cloud import storage

import google.auth.transport.requests
import google.oauth2.id_token

app = FastAPI()

# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------
PIPELINE_URL = os.getenv("PIPELINE_URL")
GETTY_API_KEY = os.getenv("GETTY_API_KEY")
GETTY_API_SECRET = os.getenv("GETTY_API_SECRET")
GCS_BUCKET = os.getenv("GCS_BUCKET")

storage_client = None
bucket = None
if GCS_BUCKET:
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET)


# ---------------------------------------------------------
# Authenticated Cloud Run call
# ---------------------------------------------------------
def call_cloud_run(url: str, endpoint: str, json_payload: dict):
    if not url:
        raise HTTPException(500, "PIPELINE_URL is not configured")

    full_url = f"{url.rstrip('/')}/{endpoint.lstrip('/')}"
    auth_req = google.auth.transport.requests.Request()
    token = google.oauth2.id_token.fetch_id_token(auth_req, url)

    return requests.post(
        full_url,
        json=json_payload,
        headers={"Authorization": f"Bearer {token}"},
        timeout=300,
    )


# ---------------------------------------------------------
# Getty Search
# ---------------------------------------------------------
@app.get("/search")
def search(q: str):
    if not GETTY_API_KEY:
        raise HTTPException(500, "GETTY_API_KEY is not configured")

    url = "https://api.gettyimages.com/v3/search/videos"
    headers = {"Api-Key": GETTY_API_KEY}

    resp = requests.get(url, params={"phrase": q}, headers=headers, timeout=60)
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, resp.text)

    data = resp.json()
    videos = data.get("videos", [])

    return {
        "query": q,
        "count": len(videos),
        "assets": videos,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


# ---------------------------------------------------------
# Licensed Download + Preview Fallback → GCS Upload
# ---------------------------------------------------------
@app.post("/download")
def download(payload: dict):
    if not GETTY_API_KEY:
        raise HTTPException(500, "GETTY_API_KEY is not configured")
    if not GETTY_API_SECRET:
        raise HTTPException(500, "GETTY_API_SECRET is not configured")
    if not GCS_BUCKET or bucket is None:
        raise HTTPException(500, "GCS_BUCKET is not configured")

    asset_id = payload.get("asset_id")
    if not asset_id:
        raise HTTPException(400, "asset_id is required")

    # -----------------------------------------------------
    # 1. Try licensed download endpoint first
    # -----------------------------------------------------
    licensed_url = None
    licensed_status = None
    licensed_body = None

    download_url = f"https://api.gettyimages.com/v3/downloads/videos/{asset_id}"
    headers = {
        "Api-Key": GETTY_API_KEY,
        "Api-Secret": GETTY_API_SECRET,
        "Accept": "application/json"
    }

    try:
        resp = requests.post(download_url, headers=headers, timeout=60)
        licensed_status = resp.status_code
        licensed_body = resp.text

        if resp.status_code == 200:
            delivery_json = resp.json()
            licensed_url = delivery_json.get("uri")
    except Exception:
        pass

    # -----------------------------------------------------
    # 2. If licensed URL unavailable → fallback to preview
    # -----------------------------------------------------
    if not licensed_url:
        meta_url = f"https://api.gettyimages.com/v3/videos/{asset_id}"
        meta_headers = {"Api-Key": GETTY_API_KEY}

        meta_resp = requests.get(meta_url, headers=meta_headers, timeout=60)
        if meta_resp.status_code != 200:
            raise HTTPException(meta_resp.status_code, "Failed to fetch preview metadata")

        try:
            meta_json = meta_resp.json()
            preview_url = meta_json["display_sizes"][0]["uri"]
        except Exception:
            raise HTTPException(500, "No preview URL available for this asset")

        final_url = preview_url
    else:
        final_url = licensed_url

    # -----------------------------------------------------
    # 3. Download the media file
    # -----------------------------------------------------
    media_resp = requests.get(final_url, timeout=300)
    if media_resp.status_code != 200:
        raise HTTPException(media_resp.status_code, "Failed to download media")

    media_bytes = media_resp.content

    # -----------------------------------------------------
    # 4. Upload to GCS
    # -----------------------------------------------------
    object_name = f"getty/{asset_id}.mp4"
    blob = bucket.blob(object_name)
    blob.upload_from_string(media_bytes, content_type="video/mp4")

    media_url = f"gs://{GCS_BUCKET}/{object_name}"

    return {
        "asset_id": asset_id,
        "media_url": media_url,
        "licensed_download_status": licensed_status,
        "licensed_download_body": licensed_body,
        "used_preview_fallback": licensed_url is None,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


# ---------------------------------------------------------
# Search + Random Asset + Download + Pipeline
# ---------------------------------------------------------
@app.get("/search-and-run")
def search_and_run(q: str):
    # 1. Search Getty
    try:
        search_json = search(q)
    except Exception as e:
        return {
            "stage": "search_failed",
            "query": q,
            "asset_id": None,
            "pipeline_status": None,
            "pipeline_preview": None,
            "error": str(e),
        }

    assets = search_json.get("assets", [])
    if not assets:
        return {
            "stage": "no_results",
            "query": q,
            "asset_id": None,
            "pipeline_status": None,
            "pipeline_preview": None,
            "error": "No Getty assets found",
        }

    # -----------------------------------------------------
    # RANDOMISE ASSET SELECTION
    # -----------------------------------------------------
    selected = random.choice(assets)
    asset_id = str(selected.get("id"))

    # 2. Download Getty asset
    try:
        download_json = download({"asset_id": asset_id})
        media_url = download_json["media_url"]
    except HTTPException as e:
        return {
            "stage": "download_failed",
            "query": q,
            "asset_id": asset_id,
            "pipeline_status": None,
            "pipeline_preview": None,
            "error": str(e.detail),
        }

    # -----------------------------------------------------
    # 3. Trigger pipeline WITH GETTY METADATA INCLUDED
    # -----------------------------------------------------
    pipeline_payload = {
        "media_url": media_url,
        "asset_id": asset_id,
        "source": "getty",
        "media_type": "video",
        "getty_metadata": selected,   # <-- FULL METADATA INCLUDED
    }

    pipeline_resp = call_cloud_run(PIPELINE_URL, "run_all", pipeline_payload)
    pipeline_status = pipeline_resp.status_code

    try:
        pipeline_json = pipeline_resp.json()
        pipeline_preview = json.dumps(pipeline_json)[:2000]
    except Exception:
        pipeline_preview = pipeline_resp.text[:2000]

    return {
        "stage": "complete" if pipeline_status == 200 else "pipeline_error",
        "query": q,
        "asset_id": asset_id,
        "pipeline_status": pipeline_status,
        "pipeline_preview": pipeline_preview,
        "error": None if pipeline_status == 200 else "Pipeline returned non-200",
    }


# ---------------------------------------------------------
# Health
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "pipeline_url": PIPELINE_URL,
        "bucket": GCS_BUCKET,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
