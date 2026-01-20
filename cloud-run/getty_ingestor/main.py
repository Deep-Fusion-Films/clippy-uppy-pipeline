import os
import json
from datetime import datetime

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from google.cloud import storage

import google.auth.transport.requests
import google.oauth2.id_token

app = FastAPI()

# -------------------------------------------------------------------
# Environment
# -------------------------------------------------------------------
PIPELINE_URL = os.getenv("PIPELINE_URL")
GETTY_API_KEY = os.getenv("GETTY_API_KEY")
GCS_BUCKET = os.getenv("GCS_BUCKET")  # e.g. df-films-assets-euw1

if not GETTY_API_KEY:
    raise RuntimeError("GETTY_API_KEY must be set")
if not GCS_BUCKET:
    raise RuntimeError("GCS_BUCKET must be set")

storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET)


# -------------------------------------------------------------------
# Authenticated Cloud Run call
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# Getty Search
# -------------------------------------------------------------------
@app.get("/search")
def search(q: str):
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


# -------------------------------------------------------------------
# Getty Download → GCS Upload
# -------------------------------------------------------------------
@app.post("/download")
def download(payload: dict):
    asset_id = payload.get("asset_id")
    if not asset_id:
        raise HTTPException(400, "asset_id is required")

    # 1. Get Getty delivery URL
    url = f"https://api.gettyimages.com/v3/downloads/{asset_id}"
    headers = {"Api-Key": GETTY_API_KEY}

    resp = requests.get(url, headers=headers, timeout=60)
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, resp.text)

    try:
        delivery_json = resp.json()
        delivery_url = delivery_json["uri"]
    except Exception:
        raise HTTPException(500, "Invalid Getty download response")

    # 2. Download the media file
    media_resp = requests.get(delivery_url, timeout=300)
    if media_resp.status_code != 200:
        raise HTTPException(media_resp.status_code, "Failed to download media")

    media_bytes = media_resp.content

    # 3. Upload to GCS
    object_name = f"getty/{asset_id}.mp4"
    blob = bucket.blob(object_name)
    blob.upload_from_string(media_bytes, content_type="video/mp4")

    media_url = f"gs://{GCS_BUCKET}/{object_name}"

    return {
        "asset_id": asset_id,
        "media_url": media_url,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


# -------------------------------------------------------------------
# Search + Download + Pipeline
# -------------------------------------------------------------------
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
            "download_attempt_status": None,
            "download_attempt_body": None,
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
            "download_attempt_status": None,
            "download_attempt_body": None,
        }

    first = assets[0]
    asset_id = str(first.get("id"))

    # 2. Download Getty asset
    try:
        download_json = download({"asset_id": asset_id})
        media_url = download_json["media_url"]
        download_status = 200
        download_body = json.dumps(download_json)
    except HTTPException as e:
        return {
            "stage": "download_failed",
            "query": q,
            "asset_id": asset_id,
            "pipeline_status": None,
            "pipeline_preview": None,
            "error": str(e.detail),
            "download_attempt_status": e.status_code,
            "download_attempt_body": None,
        }

    # 3. Trigger pipeline
    pipeline_payload = {
        "media_url": media_url,
        "asset_id": asset_id,
        "source": "getty",
        "media_type": "video",
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
        "download_attempt_status": download_status,
        "download_attempt_body": download_body,
    }


# -------------------------------------------------------------------
# Health
# -------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "pipeline_url": PIPELINE_URL,
        "bucket": GCS_BUCKET,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
