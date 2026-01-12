import os
import time
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

# Simple in-memory token cache
_getty_token_cache: dict | None = None


# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------
class SearchAndRunResponse(BaseModel):
    stage: str
    query: str
    asset_id: str | None = None
    pipeline_status: int | None = None
    pipeline_preview: str | None = None
    error: str | None = None


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
    if _getty_token_cache:
        if now < _getty_token_cache["expires_at"]:
            return _getty_token_cache["access_token"]

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
    # Default expiry ~3600s; subtract a bit for safety
    expires_in = body.get("expires_in", 3600)
    _getty_token_cache = {
        "access_token": access_token,
        "expires_at": now + expires_in - 60,
    }
    return access_token


def search_getty_videos(query: str) -> dict:
    """
    Search Getty Creative Videos and return the first video object.
    Strong, focused video search for ingestion.
    """
    token = get_getty_access_token()
    url = "https://api.gettyimages.com/v3/search/videos/creative"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    params = {
        "phrase": query,
        "page_size": 1,
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

    return videos[0]


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


def trigger_pipeline(asset_id: str, metadata: dict, download_url: str) -> requests.Response:
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
    Auto-ingest endpoint:
    1. Search Getty creative videos for query q
    2. Pick the first result
    3. Fetch the download URL
    4. Trigger the enrichment pipeline with asset_id, metadata, and download_url
    """

    # 1–2. Search Getty and extract first asset
    try:
        video = search_getty_videos(q)
    except HTTPException as http_err:
        return SearchAndRunResponse(
            stage="getty_search",
            query=q,
            error=http_err.detail,  # type: ignore[arg-type]
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
            error=http_err.detail,  # type: ignore[arg-type]
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
