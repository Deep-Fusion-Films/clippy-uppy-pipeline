import os
import time
import random
from typing import Optional, Dict, Any, List

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# -------------------------------------------------------------------
# Environment Variables
# -------------------------------------------------------------------
GETTY_API_KEY = os.environ["GETTY_API_KEY"]
GETTY_API_SECRET = os.environ["GETTY_API_SECRET"]
START_PIPELINE_URL = os.environ["START_PIPELINE_URL"]

# OAuth token cache
_token_cache: Optional[Dict[str, Any]] = None


# -------------------------------------------------------------------
# Response Model
# -------------------------------------------------------------------
class SearchAndRunResponse(BaseModel):
    stage: str
    query: str
    asset_id: Optional[str] = None
    pipeline_status: Optional[int] = None
    pipeline_preview: Optional[str] = None
    error: Optional[str] = None
    download_attempt_status: Optional[int] = None
    download_attempt_body: Optional[str] = None


# -------------------------------------------------------------------
# OAuth Token
# -------------------------------------------------------------------
def get_getty_access_token() -> str:
    global _token_cache
    now = time.time()

    if _token_cache and now < _token_cache["expires_at"]:
        return _token_cache["access_token"]

    url = "https://authentication.gettyimages.com/oauth2/token"
    data = {
        "client_id": GETTY_API_KEY,
        "client_secret": GETTY_API_SECRET,
        "grant_type": "client_credentials",
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    resp = requests.post(url, data=data, headers=headers)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Getty OAuth failed: {resp.text[:500]}",
        )

    body = resp.json()
    access_token = body["access_token"]
    expires_in = int(body.get("expires_in", 3600))

    _token_cache = {
        "access_token": access_token,
        "expires_at": now + expires_in - 60,
    }

    return access_token


# -------------------------------------------------------------------
# Attempt Licensed Download (Strict Mode)
# -------------------------------------------------------------------
def attempt_licensed_download(asset_id: str) -> Dict[str, Any]:
    url = f"https://api.gettyimages.com/v3/downloads/videos/{asset_id}"

    headers = {
        "Api-Key": GETTY_API_KEY,
        "Authorization": f"Bearer {get_getty_access_token()}",
        "Accept": "application/json",
    }

    resp = requests.post(url, headers=headers)

    return {
        "status": resp.status_code,
        "body": resp.text[:500],
    }


# -------------------------------------------------------------------
# Extract Preview MP4 (P2: first MP4 returned)
# -------------------------------------------------------------------
def extract_preview_mp4(video: Dict[str, Any]) -> Optional[str]:
    for d in video.get("display_sizes", []):
        uri = d.get("uri")
        if isinstance(uri, str) and uri.endswith(".mp4"):
            return uri
    return None


# -------------------------------------------------------------------
# Search Core Video API in Blocks of 25, Across 5 Pages
# -------------------------------------------------------------------
def find_first_usable_asset(query: str) -> Dict[str, Any]:
    base_url = "https://api.gettyimages.com/v3/search/videos"

    headers = {
        "Api-Key": GETTY_API_KEY,
        "Authorization": f"Bearer {get_getty_access_token()}",
        "Accept": "application/json",
    }

    for page in range(1, 6):  # Pages 1–5
        params = {
            "phrase": query,
            "fields": "id,title,caption,display_sizes",
            "page_size": 25,
            "page": page,
            "sort_order": "best_match",
        }

        resp = requests.get(base_url, headers=headers, params=params)
        if resp.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Getty search failed on page {page}: {resp.text[:500]}",
            )

        videos = resp.json().get("videos", [])
        if not videos:
            continue

        # Evaluate each asset
        for video in videos:
            asset_id = video.get("id")
            if not asset_id:
                continue

            # 1. Strict Mode: Attempt licensed download
            download_attempt = attempt_licensed_download(asset_id)

            if download_attempt["status"] == 200:
                # Licensed download succeeded
                return {
                    "asset": video,
                    "asset_id": asset_id,
                    "download_attempt": download_attempt,
                    "download_url": download_attempt["body"],
                    "preview_url": None,
                }

            # 2. Fallback: Preview MP4
            preview_url = extract_preview_mp4(video)
            if preview_url:
                return {
                    "asset": video,
                    "asset_id": asset_id,
                    "download_attempt": download_attempt,
                    "download_url": None,
                    "preview_url": preview_url,
                }

    raise HTTPException(
        status_code=404,
        detail=f"No usable assets (download or preview) found for '{query}' across 5 pages.",
    )


# -------------------------------------------------------------------
# Trigger Pipeline
# -------------------------------------------------------------------
def trigger_pipeline(asset_id: str, metadata: Dict[str, Any], url: str):
    payload = {
        "asset_id": asset_id,
        "getty_metadata": metadata,
        "download_url": url,
    }
    return requests.post(START_PIPELINE_URL, json=payload)


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/search-and-run", response_model=SearchAndRunResponse)
def search_and_run(q: str):
    try:
        result = find_first_usable_asset(q)
    except HTTPException as http_err:
        return SearchAndRunResponse(
            stage="getty_search",
            query=q,
            error=str(http_err.detail),
        )

    asset = result["asset"]
    asset_id = result["asset_id"]
    download_attempt = result["download_attempt"]
    download_url = result["download_url"]
    preview_url = result["preview_url"]

    usable_url = download_url or preview_url

    try:
        pipeline_resp = trigger_pipeline(asset_id, asset, usable_url)
    except Exception as e:
        return SearchAndRunResponse(
            stage="pipeline_trigger",
            query=q,
            asset_id=asset_id,
            download_attempt_status=download_attempt["status"],
            download_attempt_body=download_attempt["body"],
            error=str(e),
        )

    return SearchAndRunResponse(
        stage="complete",
        query=q,
        asset_id=asset_id,
        pipeline_status=pipeline_resp.status_code,
        pipeline_preview=pipeline_resp.text[:500],
        download_attempt_status=download_attempt["status"],
        download_attempt_body=download_attempt["body"],
    )
    )
