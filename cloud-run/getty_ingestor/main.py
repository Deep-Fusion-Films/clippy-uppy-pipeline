import os
import time
import random
from typing import Optional, Dict, Any, List, Tuple

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
# Getty OAuth
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
        "expires_at": now + expires_in - 60,  # refresh 1 min early
    }

    return access_token


# -------------------------------------------------------------------
# Getty Search (Creative Video)
# -------------------------------------------------------------------
def search_getty_videos_random(
    query: str,
    pages: int = 5,
    page_size: int = 30,
) -> Dict[str, Any]:
    """
    Getty Creative Video search:
    - Uses fields to force Getty to return display_sizes (including MP4s)
    - Uses product_types and sort_order to stabilize results
    - Searches multiple pages
    - Filters for MP4-enabled assets only
    - Returns one random valid asset with full metadata
    """

    url = "https://api.gettyimages.com/v3/search/videos/creative"

    headers = {
        "Api-Key": GETTY_API_KEY,
        "Accept": "application/json",
    }

    # NOTE: minimum_size is NOT valid for video → removed
    base_params = {
        "phrase": query,
        "fields": (
            "id,title,caption,display_sizes,collection_code,collection_name,"
            "license_model,asset_family,product_types"
        ),
        "product_types": "easyaccess,royaltyfree",
        "sort_order": "best_match",
        "exclude_nudity": "true",
    }

    valid_assets: List[Dict[str, Any]] = []

    for page in range(1, pages + 1):
        params = {
            **base_params,
            "page": page,
            "page_size": page_size,
        }

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

        # Filter for assets that contain at least one MP4 preview
        for v in videos:
            display_sizes = v.get("display_sizes", [])
            if any(
                isinstance(d.get("uri"), str) and d["uri"].endswith(".mp4")
                for d in display_sizes
            ):
                valid_assets.append(v)

        # Early exit if we found valid assets
        if valid_assets:
            break

    if not valid_assets:
        raise HTTPException(
            status_code=404,
            detail=f"No MP4-enabled Getty assets found for query '{query}' across {pages} pages",
        )

    return random.choice(valid_assets)


# -------------------------------------------------------------------
# Preview MP4 extraction (fallback path)
# -------------------------------------------------------------------
def extract_public_mp4(video: Dict[str, Any]) -> str:
    """
    Extract a public MP4 URL from display_sizes.
    This uses preview MP4s and does NOT require OAuth/download entitlement.
    """
    mp4_candidates = [
        d.get("uri")
        for d in video.get("display_sizes", [])
        if isinstance(d.get("uri"), str) and d["uri"].endswith(".mp4")
    ]

    if not mp4_candidates:
        raise HTTPException(
            status_code=502,
            detail="No MP4 URL found in display_sizes",
        )

    return mp4_candidates[0]


# -------------------------------------------------------------------
# Licensed download (primary path)
# -------------------------------------------------------------------
def get_licensed_download_url(asset_id: str, token: str) -> str:
    """
    Request a licensed download URL for a video asset via Getty downloads API.
    Requires a valid OAuth token and appropriate license agreement.
    """
    url = f"https://api.gettyimages.com/v3/downloads/videos/{asset_id}"
    headers = {
        "Api-Key": GETTY_API_KEY,
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    params = {
        "auto_download": "false",
        "product_type": "easyaccess",
    }

    resp = requests.post(url, headers=headers, params=params)

    if resp.status_code != 200:
        # Try to parse structured error to detect NoAgreement
        try:
            body = resp.json()
        except Exception:
            body = None

        if body and body.get("ErrorCode") == "NoAgreement":
            # Signal to caller that this is a licensing issue, not a technical failure
            raise HTTPException(
                status_code=409,
                detail=f"Getty download NoAgreement for asset {asset_id}: {resp.text[:500]}",
            )

        # Generic failure
        raise HTTPException(
            status_code=502,
            detail=f"Getty download failed for asset {asset_id}: {resp.text[:500]}",
        )

    body = resp.json()
    download = body.get("download")
    if not isinstance(download, dict) or "uri" not in download:
        raise HTTPException(
            status_code=502,
            detail=f"Getty download response missing 'download.uri' for asset {asset_id}: {resp.text[:500]}",
        )

    return download["uri"]


# -------------------------------------------------------------------
# Hybrid download selection
# -------------------------------------------------------------------
def resolve_download_url_hybrid(video: Dict[str, Any]) -> Tuple[str, str]:
    """
    Hybrid strategy:
    1. Try to get licensed download URL via OAuth/downloads endpoint.
    2. If Getty returns NoAgreement, fall back to preview MP4 in display_sizes.
    Returns (download_url, source) where source is 'licensed' or 'preview'.
    """
    asset_id = video.get("id")
    if not asset_id:
        raise HTTPException(
            status_code=502,
            detail="Getty video result missing 'id' during download resolution",
        )

    # First, attempt licensed download
    try:
        token = get_getty_access_token()
        licensed_url = get_licensed_download_url(asset_id, token)
        return licensed_url, "licensed"
    except HTTPException as e:
        detail = str(e.detail)
        # NoAgreement → fall back to preview MP4
        if "NoAgreement" in detail or "ErrorCode\":\"NoAgreement\"" in detail:
            # Fall back to preview MP4
            preview_url = extract_public_mp4(video)
            return preview_url, "preview"

        # Other errors are treated as hard failures
        raise


# -------------------------------------------------------------------
# Pipeline trigger
# -------------------------------------------------------------------
def trigger_pipeline(asset_id: str, metadata: Dict[str, Any], download_url: str) -> requests.Response:
    """
    Trigger the downstream pipeline with asset_id, metadata, and download URL.
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
    try:
        token = get_getty_access_token()
        return {
            "stage": "getty_auth",
            "token_received": bool(token),
            "token_preview": token[:20] + "..." if token else None,
        }
    except Exception as e:
        return {"stage": "getty_auth", "error": str(e)}


@app.get("/debug-token")
def debug_token():
    return {"status": "ok", "message": "debug endpoint reachable"}


@app.get("/debug-raw")
def debug_raw(q: str):
    """
    Return the raw Getty JSON response exactly as Cloud Run sees it.
    This allows us to compare Cloud Run's response with manual curl tests.
    """
    url = "https://api.gettyimages.com/v3/search/videos/creative"

    headers = {
        "Api-Key": GETTY_API_KEY,
        "Accept": "application/json",
    }

    params = {
        "phrase": q,
        "page_size": 1,
    }

    resp = requests.get(url, headers=headers, params=params)

    try:
        return resp.json()
    except Exception:
        return {
            "error": "Non-JSON response",
            "status": resp.status_code,
            "body": resp.text[:500],
        }


@app.get("/debug-search-run")
def debug_search_run(q: str):
    """
    Debug endpoint that runs the search and returns the raw chosen video object.
    """
    result = search_getty_videos_random(q, pages=5, page_size=30)
    return result


@app.get("/search-and-run", response_model=SearchAndRunResponse)
def search_and_run(q: str):
    """
    Auto-ingest endpoint (hybrid mode):
    1. Search Getty creative videos
    2. Randomly pick one MP4-capable asset
    3. Try licensed download URL via OAuth
    4. If NoAgreement, fall back to preview MP4
    5. Trigger pipeline with chosen URL
    """

    # 1–2. Search Getty and randomly select one asset
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
            error="Getty video result missing 'id'",
        )

    # 3–4. Resolve download URL using hybrid strategy
    try:
        download_url, source = resolve_download_url_hybrid(video)
    except HTTPException as http_err:
        return SearchAndRunResponse(
            stage="download_resolve",
            query=q,
            asset_id=asset_id,
            error=str(http_err.detail),
        )

    # 5. Trigger pipeline
    try:
        pipeline_resp = trigger_pipeline(asset_id, video, download_url)
    except Exception as e:
        return SearchAndRunResponse(
            stage="pipeline_trigger",
            query=q,
            asset_id=asset_id,
            error=f"Pipeline request failed: {str(e)}",
        )

    # Note: we don't expose source in the response model, but it’s available in logs if you add logging
    return SearchAndRunResponse(
        stage="complete",
        query=q,
        asset_id=asset_id,
        pipeline_status=pipeline_resp.status_code,
        pipeline_preview=pipeline_resp.text[:500],
    )
