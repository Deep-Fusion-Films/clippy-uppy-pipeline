import os
import time
import random
import requests
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
GETTY_API_KEY = os.environ["GETTY_API_KEY"]
GETTY_API_SECRET = os.environ["GETTY_API_SECRET"]
START_PIPELINE_URL = os.environ["START_PIPELINE_URL"]

# Simple in-memory token cache
_getty_token_cache: Optional[Dict[str, Any]] = None


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
# Getty helpers
# -------------------------------------------------------------------
def get_getty_access_token() -> str:
    """
    Get and cache a Getty OAuth token (client_credentials).
    Token is cached until expiry to avoid repeated calls.
    """
    global _getty_token_cache

    now = time.time()

    # Use cached token if still valid
    if _getty_token_cache:
        if now < _getty_token_cache["expires_at"]:
            return _getty_token_cache["access_token"]

    # Request new token
    url = "https://api.gettyimages.com/oauth2/token"
    data = {
        "client_id": GETTY_API_KEY,
        "client_secret": GETTY_API_SECRET,
        "grant_type": "client_credentials",
    }
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "User-Agent": "python-requests",
    }

    resp = requests.post(url, data=data, headers=headers)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Getty auth failed: {resp.text[:500]}",
        )

    body = resp.json()
    access_token = body["access_token"]

    # Getty may return expires_in as string → convert safely
    expires_in_raw = body.get("expires_in", 3600)
    try:
        expires_in = int(expires_in_raw)
    except Exception:
        expires_in = 3600

    _getty_token_cache = {
        "access_token": access_token,
        "expires_at": now + expires_in - 60,  # renew a bit early
    }

    return access_token


def search_getty_videos_random(query: str, page_size: int = 10) -> Dict[str, Any]:
    """
    Search Getty Creative Videos and return ONE RANDOM video object
    from up to `page_size` results for the given query.
    """
    token = get_getty_access_token()
    url = "https://api.gettyimages.com/v3/search/videos/creative"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    params = {
        "phrase": query,
        "page_size": page_size,
    }

    resp = requests.get(url, headers=headers, params=params)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Getty search failed: {resp.text[:500]}",
        )

    data = resp.json()
    videos = data.get("videos", [])
    if not videos:
        raise HTTPException(
            status_code=404,
            detail=f"No Getty video results found for query '{query}'",
        )

    # Pure random choice among the returned results
    return random.choice(videos)


def get_getty_download_url(asset_id: str) -> str:
    """
    Retrieve the download URL for a Getty video asset.
    """
    token = get_getty_access_token()
    url = f"https://api.gettyimages.com/v3/downloads/{asset_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }

    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Getty download lookup failed: {resp.text[:500]}",
        )

    data = resp.json()
    download_url = data.get("uri")
    if not download_url:
        raise HTTPException(
            status_code=502,
            detail=f"Getty did not return a download URL: {data}",
        )

    return download_url


def trigger_pipeline(asset_id: str, metadata: Dict[str, Any], download_url: str) -> requests.Response:
    """
    Trigger the downstream pipeline with asset_id, metadata, and download URL.
    (Option C payload)
    """
    payload = {
        "asset_id": asset_id,
        "getty_metadata": metadata,
        "download_url": download_url,
    }

    resp = requests.post(START_PIPELINE_URL, json=payload)
    return resp


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/debug-getty")
def debug_getty():
    """
    Simple endpoint to verify Getty auth works from this service.
    """
    try:
        token = get_getty_access_token()
        return {
            "stage": "getty_auth",
            "token_received": bool(token),
            "token_preview": token[:20] + "..." if token else None,
        }
    except Exception as e:
        return {
            "stage": "getty_auth",
            "error": str(e),
        }


@app.get("/debug-token")
def debug_token():
    """
    Lightweight 'is the service callable' debug endpoint.
    (Cloud Run identity token is checked by Cloud Run itself.)
    """
    return {"status": "ok", "message": "debug endpoint reachable"}


@app.get("/search-and-run", response_model=SearchAndRunResponse)
def search_and_run(q: str):
    """
    Auto-ingest endpoint (pure random selection, fast mode):
    1. Search Getty creative videos for query q (page_size=10)
    2. Randomly pick ONE video from the results
    3. Fetch the download URL
    4. Trigger the enrichment pipeline with asset_id, metadata, and download_url
    """

    # 1–2. Search Getty and randomly select one asset
    try:
        video = search_getty_videos_random(q, page_size=10)
    except HTTPException as http_err:
        # http_err.detail is whatever we set above
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
            error="Getty video result missing 'id'",
        )

    # 3. Fetch download URL
    try:
        download_url = get_getty_download_url(asset_id)
    except HTTPException as http_err:
        return SearchAndRunResponse(
            stage="getty_download",
            query=q,
            asset_id=asset_id,
            error=str(http_err.detail),
        )

    # 4. Trigger pipeline
    try:
        pipeline_resp = trigger_pipeline(asset_id, video, download_url)
    except Exception as e:
        return SearchAndRunResponse(
            stage="pipeline_trigger",
            query=q,
            asset_id=asset_id,
            error=f"Pipeline request failed: {str(e)}",
        )

    return SearchAndRunResponse(
        stage="complete",
        query=q,
        asset_id=asset_id,
        pipeline_status=pipeline_resp.status_code,
        pipeline_preview=pipeline_resp.text[:500],
    )
