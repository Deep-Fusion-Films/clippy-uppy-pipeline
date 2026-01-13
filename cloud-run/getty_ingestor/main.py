import os
import time
import random
from typing import Optional, Dict, Any, List

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
GETTY_API_KEY = os.environ["GETTY_API_KEY"]
GETTY_API_SECRET = os.environ["GETTY_API_SECRET"]
START_PIPELINE_URL = os.environ["START_PIPELINE_URL"]

# OAuth token cache
_token_cache: Optional[Dict[str, Any]] = None


# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------
class SearchAndRunResponse(BaseModel):
    stage: str
    query: str
    asset_id: Optional[str] = None
    pipeline_status: Optional[int] = None
    pipeline_preview: Optional[str] = None
    error: Optional[str] = None


# -------------------------------------------------------------------
# Getty OAuth (still required for your account)
# -------------------------------------------------------------------
def get_getty_access_token() -> str:
    global _token_cache
    now = time.time()

    if _token_cache and now < _token_cache["expires_at"]:
        return _token_cache["access_token"]

    url = "https://api.gettyimages.com/oauth2/token"
    data = {
        "client_id": GETTY_API_KEY,
        "client_secret": GETTY_API_SECRET,
        "grant_type": "client_credentials",
    }
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }

    resp = requests.post(url, data=data, headers=headers)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Getty auth failed: {resp.text[:500]}",
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
# Creative Video Search (OAuth + Api-Key)
# -------------------------------------------------------------------
def search_getty_videos_random(
    query: str,
    pages: int = 5,
    page_size: int = 30,
) -> Dict[str, Any]:

    url = "https://api.gettyimages.com/v3/search/videos/creative"

    headers = {
        "Api-Key": GETTY_API_KEY,
        "Authorization": f"Bearer {get_getty_access_token()}",
        "Accept": "application/json",
    }

    base_params = {
        "phrase": query,
        "fields": (
            "id,title,caption,display_sizes,collection_code,collection_name,"
            "license_model,asset_family,product_types"
        ),
        "sort_order": "best_match",
        "exclude_nudity": "true",
    }

    valid_assets: List[Dict[str, Any]] = []

    for page in range(1, pages + 1):
        params = {**base_params, "page": page, "page_size": page_size}
        resp = requests.get(url, headers=headers, params=params)

        if resp.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Getty search failed on page {page}: {resp.text[:500]}",
            )

        data = resp.json()
        videos = data.get("videos", [])
        if not videos:
            continue

        # Only accept assets with preview MP4s
        for v in videos:
            if any(
                isinstance(d.get("uri"), str) and d["uri"].endswith(".mp4")
                for d in v.get("display_sizes", [])
            ):
                valid_assets.append(v)

        if valid_assets:
            break

    if not valid_assets:
        raise HTTPException(
            status_code=404,
            detail=f"No preview-enabled Creative assets found for '{query}'",
        )

    return random.choice(valid_assets)


# -------------------------------------------------------------------
# Extract Preview MP4 (your only download path)
# -------------------------------------------------------------------
def extract_preview_mp4(video: Dict[str, Any]) -> str:
    mp4s = [
        d["uri"]
        for d in video.get("display_sizes", [])
        if isinstance(d.get("uri"), str) and d["uri"].endswith(".mp4")
    ]

    if not mp4s:
        raise HTTPException(
            status_code=502,
            detail="No preview MP4 found in display_sizes",
        )

    return mp4s[0]


# -------------------------------------------------------------------
# Pipeline Trigger
# -------------------------------------------------------------------
def trigger_pipeline(asset_id: str, metadata: Dict[str, Any], download_url: str):
    payload = {
        "asset_id": asset_id,
        "getty_metadata": metadata,
        "download_url": download_url,
    }
    return requests.post(START_PIPELINE_URL, json=payload)


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/debug-getty")
def debug_getty():
    try:
        token = get_getty_access_token()
        return {"token_preview": token[:20] + "..."}
    except Exception as e:
        return {"error": str(e)}


@app.get("/debug-raw")
def debug_raw(q: str):
    url = "https://api.gettyimages.com/v3/search/videos/creative"
    headers = {
        "Api-Key": GETTY_API_KEY,
        "Authorization": f"Bearer {get_getty_access_token()}",
        "Accept": "application/json",
    }
    params = {"phrase": q, "page_size": 1}
    resp = requests.get(url, headers=headers, params=params)
    try:
        return resp.json()
    except:
        return {"status": resp.status_code, "body": resp.text[:500]}


@app.get("/search-and-run", response_model=SearchAndRunResponse)
def search_and_run(q: str):

    # 1. Search Creative
    try:
        video = search_getty_videos_random(q, page_size=10)
    except HTTPException as http_err:
        return SearchAndRunResponse(
            stage="getty_search",
            query=q,
            error=str(http_err.detail),
        )

    asset_id = video.get("id")
    if not asset_id:
        return SearchAndRunResponse(
            stage="getty_search",
            query=q,
            error="Missing asset ID",
        )

    # 2. Extract preview MP4
    try:
        download_url = extract_preview_mp4(video)
    except HTTPException as http_err:
        return SearchAndRunResponse(
            stage="preview_extract",
            query=q,
            asset_id=asset_id,
            error=str(http_err.detail),
        )

    # 3. Trigger pipeline
    try:
        pipeline_resp = trigger_pipeline(asset_id, video, download_url)
    except Exception as e:
        return SearchAndRunResponse(
            stage="pipeline_trigger",
            query=q,
            asset_id=asset_id,
            error=str(e),
        )

    return SearchAndRunResponse(
        stage="complete",
        query=q,
        asset_id=asset_id,
        pipeline_status=pipeline_resp.status_code,
        pipeline_preview=pipeline_resp.text[:500],
    )

