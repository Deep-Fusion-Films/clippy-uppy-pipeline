import os
import json
from datetime import datetime
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

import google.auth.transport.requests
import google.oauth2.id_token

app = FastAPI()

# -------------------------------------------------------------------
# Environment
# -------------------------------------------------------------------
PIPELINE_URL = os.getenv("PIPELINE_URL")  # only required env var


# -------------------------------------------------------------------
# Authenticated Cloud Run call
# -------------------------------------------------------------------
def call_cloud_run(url: str, endpoint: str, json_payload: dict) -> requests.Response:
    if not url:
        raise HTTPException(status_code=500, detail="PIPELINE_URL is not configured")

    full_url = f"{url.rstrip('/')}/{endpoint.lstrip('/')}"
    auth_req = google.auth.transport.requests.Request()
    token = google.oauth2.id_token.fetch_id_token(auth_req, url)

    resp = requests.post(
        full_url,
        json=json_payload,
        headers={"Authorization": f"Bearer {token}"},
        timeout=300,
    )
    return resp


# -------------------------------------------------------------------
# Getty search (direct API call)
# -------------------------------------------------------------------
def getty_search(query: str) -> dict:
    """
    Calls Getty Images API directly.
    Replace YOUR_API_KEY with your actual key.
    """
    api_key = os.getenv("GETTY_API_KEY")
    if not api_key:
        raise HTTPException(500, "GETTY_API_KEY is not configured")

    url = "https://api.gettyimages.com/v3/search/videos"
    headers = {"Api-Key": api_key}

    resp = requests.get(url, params={"phrase": query}, headers=headers, timeout=60)

    if resp.status_code != 200:
        raise HTTPException(resp.status_code, f"Getty search failed: {resp.text}")

    return resp.json()


# -------------------------------------------------------------------
# Getty download (direct API call)
# -------------------------------------------------------------------
def getty_download(asset_id: str) -> tuple[int, str]:
    """
    Calls Getty delivery API and returns the download URL.
    """
    api_key = os.getenv("GETTY_API_KEY")
    if not api_key:
        return 500, "GETTY_API_KEY is not configured"

    url = f"https://api.gettyimages.com/v3/downloads/{asset_id}"
    headers = {"Api-Key": api_key}

    resp = requests.get(url, headers=headers, timeout=60)
    return resp.status_code, resp.text


# -------------------------------------------------------------------
# Health
# -------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "pipeline_url": PIPELINE_URL,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# -------------------------------------------------------------------
# Search + run pipeline
# -------------------------------------------------------------------
@app.get("/search-and-run")
def search_and_run(q: str):
    """
    1. Search Getty
    2. Download asset
    3. Trigger pipeline
    4. Return summary
    """

    # ---------------------------------------------------------------
    # 1. Search Getty
    # ---------------------------------------------------------------
    try:
        search_json = getty_search(q)
    except HTTPException as e:
        return {
            "stage": "search_failed",
            "query": q,
            "asset_id": None,
            "pipeline_status": None,
            "pipeline_preview": None,
            "error": str(e.detail),
            "download_attempt_status": None,
            "download_attempt_body": None,
        }

    assets = search_json.get("videos", [])
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

    # ---------------------------------------------------------------
    # 2. Download Getty asset
    # ---------------------------------------------------------------
    download_status, download_body = getty_download(asset_id)

    if download_status != 200:
        return {
            "stage": "download_failed",
            "query": q,
            "asset_id": asset_id,
            "pipeline_status": None,
            "pipeline_preview": None,
            "error": "Getty download failed",
            "download_attempt_status": download_status,
            "download_attempt_body": download_body,
        }

    try:
        download_json = json.loads(download_body)
        media_url = download_json["uri"]
    except Exception:
        return {
            "stage": "download_failed",
            "query": q,
            "asset_id": asset_id,
            "pipeline_status": None,
            "pipeline_preview": None,
            "error": "Invalid Getty download response",
            "download_attempt_status": download_status,
            "download_attempt_body": download_body,
        }

    # ---------------------------------------------------------------
    # 3. Trigger pipeline
    # ---------------------------------------------------------------
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
