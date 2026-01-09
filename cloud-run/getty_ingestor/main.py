import os
import base64
import time
from typing import Optional, Dict, Any, List

import requests
from fastapi import FastAPI, Request, HTTPException
from datetime import datetime
import google.auth.transport.requests
import google.oauth2.id_token

app = FastAPI()

# -------------------------------------------------------------------
# Environment / Config
# -------------------------------------------------------------------
GETTY_API_KEY = os.getenv("GETTY_API_KEY")
GETTY_API_SECRET = os.getenv("GETTY_API_SECRET")

if not GETTY_API_KEY or not GETTY_API_SECRET:
    raise RuntimeError("GETTY_API_KEY and GETTY_API_SECRET must be set")

GETTY_API_BASE = os.getenv("GETTY_API_BASE", "https://api.gettyimages.com")
START_PIPELINE_URL = os.getenv("START_PIPELINE_URL")

if not START_PIPELINE_URL:
    raise RuntimeError("START_PIPELINE_URL must be set")

# simple in-memory token cache
_getty_token: Optional[str] = None
_getty_token_expiry: float = 0.0


# -------------------------------------------------------------------
# Getty OAuth2 token handling
# -------------------------------------------------------------------
def get_getty_access_token() -> str:
    global _getty_token, _getty_token_expiry

    now = time.time()
    if _getty_token and now < _getty_token_expiry - 60:
        return _getty_token

    token_url = f"{GETTY_API_BASE}/oauth2/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": GETTY_API_KEY,
        "client_secret": GETTY_API_SECRET,
    }

    try:
        resp = requests.post(token_url, data=data, timeout=10)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error contacting Getty OAuth2 endpoint: {e}",
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Failed to get Getty access token: {resp.text}",
        )

    payload = resp.json()
    _getty_token = payload.get("access_token")
    expires_in = payload.get("expires_in", 1800)
    _getty_token_expiry = now + expires_in

    if not _getty_token:
        raise HTTPException(
            status_code=500,
            detail="Getty OAuth2 response missing access_token",
        )

    return _getty_token


def get_getty_headers() -> Dict[str, str]:
    token = get_getty_access_token()
    return {
        "Authorization": f"Bearer {token}",
        "Api-Key": GETTY_API_KEY,
    }


# -------------------------------------------------------------------
# Getty search helpers
# -------------------------------------------------------------------
def search_images(query: str, page: int = 1, page_size: int = 1) -> List[Dict[str, Any]]:
    url = f"{GETTY_API_BASE}/v3/search/images/creative"
    params = {
        "phrase": query,
        "page": page,
        "page_size": page_size,
    }

    try:
        resp = requests.get(url, headers=get_getty_headers(), params=params, timeout=10)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calling Getty images search: {e}",
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Getty images search failed: {resp.text}",
        )

    data = resp.json()
    return data.get("images", [])


def search_videos(query: str, page: int = 1, page_size: int = 1) -> List[Dict[str, Any]]:
    url = f"{GETTY_API_BASE}/v3/search/videos/creative"
    params = {
        "phrase": query,
        "page": page,
        "page_size": page_size,
    }

    try:
        resp = requests.get(url, headers=get_getty_headers(), params=params, timeout=10)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calling Getty videos search: {e}",
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Getty videos search failed: {resp.text}",
        )

    data = resp.json()
    return data.get("videos", [])


# -------------------------------------------------------------------
# Getty download helpers
# -------------------------------------------------------------------
def get_image_download_url(asset_id: str) -> str:
    url = f"{GETTY_API_BASE}/v3/downloads/images/{asset_id}"
    body = {
        "auto_download": False
    }

    try:
        resp = requests.post(url, headers=get_getty_headers(), json=body, timeout=10)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calling Getty image download endpoint: {e}",
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Getty image download failed: {resp.text}",
        )

    data = resp.json()
    download_url = data.get("uri")
    if not download_url:
        raise HTTPException(
            status_code=500,
            detail="Getty image download response missing 'uri'",
        )

    return download_url


def get_video_download_url(asset_id: str) -> str:
    url = f"{GETTY_API_BASE}/v3/downloads/videos/{asset_id}"
    body = {
        "auto_download": False
    }

    try:
        resp = requests.post(url, headers=get_getty_headers(), json=body, timeout=10)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calling Getty video download endpoint: {e}",
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Getty video download failed: {resp.text}",
        )

    data = resp.json()
    download_url = data.get("uri")
    if not download_url:
        raise HTTPException(
            status_code=500,
            detail="Getty video download response missing 'uri'",
        )

    return download_url


def fetch_bytes_from_url(url: str) -> bytes:
    try:
        resp = requests.get(url, timeout=30)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error downloading media from Getty: {e}",
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Failed to download media from Getty: {resp.text}",
        )

    content = resp.content
    if not content:
        raise HTTPException(
            status_code=500,
            detail="Downloaded media is empty",
        )

    return content


# -------------------------------------------------------------------
# Call start_pipeline (authenticated Cloud Run call)
# -------------------------------------------------------------------
def call_start_pipeline(payload: dict) -> dict:
    if not START_PIPELINE_URL:
        raise HTTPException(
            status_code=500,
            detail="START_PIPELINE_URL is not configured",
        )

    auth_req = google.auth.transport.requests.Request()
    token = google.oauth2.id_token.fetch_id_token(auth_req, START_PIPELINE_URL)

    try:
        resp = requests.post(
            f"{START_PIPELINE_URL}/run_all",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=60,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calling start_pipeline: {e}",
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"start_pipeline returned error: {resp.text}",
        )

    return resp.json()


# -------------------------------------------------------------------
# Utility: determine media type for an asset
# -------------------------------------------------------------------
def infer_media_type_from_asset(asset: Dict[str, Any], kind: str) -> str:
    """
    kind: 'image' or 'video' depending on which search produced it.
    """
    if kind == "video":
        return "video"
    return "image"


# -------------------------------------------------------------------
# Health check
# -------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": {
            "getty_base": GETTY_API_BASE,
            "start_pipeline_url": START_PIPELINE_URL,
        },
    }


# -------------------------------------------------------------------
# Search endpoint (manual testing / debugging)
# -------------------------------------------------------------------
@app.post("/search")
async def search(req: Request):
    body = await req.json()
    query = body.get("query")
    page = body.get("page", 1)
    page_size = body.get("page_size", 1)

    if not query:
        raise HTTPException(
            status_code=400,
            detail="query is required",
        )

    # First try images, then videos if no images found
    images = search_images(query, page=page, page_size=page_size)
    if images:
        return images

    videos = search_videos(query, page=page, page_size=page_size)
    return videos


# -------------------------------------------------------------------
# Main processing endpoint
# -------------------------------------------------------------------
@app.post("/process")
async def process(req: Request):
    """
    Accepts either:
    - { "asset_id": "2196875009", "media_type": "image" | "video" }
    - { "query": "woman hiking near mountains" }

    If query is provided, it will:
      - Search images first; if none found, search videos.
      - Take the first result.
    Then:
      - Call Getty downloads API to get a download URL.
      - Download the media bytes.
      - Base64-encode the bytes.
      - Forward to start_pipeline as a Getty asset.
    """
    body = await req.json()

    asset_id = body.get("asset_id")
    explicit_media_type = body.get("media_type")
    query = body.get("query")

    asset: Optional[Dict[str, Any]] = None
    media_type: Optional[str] = None
    search_kind: Optional[str] = None  # 'image' or 'video'

    # -----------------------------------------------------------------
    # Path 1: query-driven (auto-find asset)
    # -----------------------------------------------------------------
    if query and not asset_id:
        images = search_images(query, page=1, page_size=1)
        if images:
            asset = images[0]
            asset_id = asset.get("id")
            search_kind = "image"
        else:
            videos = search_videos(query, page=1, page_size=1)
            if not videos:
                raise HTTPException(
                    status_code=404,
                    detail=f"No Getty assets found for query: {query}",
                )
            asset = videos[0]
            asset_id = asset.get("id")
            search_kind = "video"

        if not asset_id:
            raise HTTPException(
                status_code=500,
                detail="Getty search result missing asset id",
            )

        media_type = infer_media_type_from_asset(asset, search_kind)

    # -----------------------------------------------------------------
    # Path 2: explicit asset_id (and optional explicit media_type)
    # -----------------------------------------------------------------
    elif asset_id:
        # If media_type is provided by caller, trust it; otherwise assume image.
        media_type = explicit_media_type or "image"
        asset = {"id": asset_id}  # minimal; we won't have full metadata here
        search_kind = media_type

    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either asset_id or query",
        )

    # -----------------------------------------------------------------
    # Decide which Getty download endpoint to use
    # -----------------------------------------------------------------
    if media_type == "video":
        download_url = get_video_download_url(asset_id)
    else:
        download_url = get_image_download_url(asset_id)

    media_bytes = fetch_bytes_from_url(download_url)
    media_b64 = base64.b64encode(media_bytes).decode("utf-8")

    # -----------------------------------------------------------------
    # Build Getty metadata payload
    # -----------------------------------------------------------------
    getty_metadata = asset or {}
    getty_metadata["search_kind"] = search_kind
    getty_metadata["resolved_media_type"] = media_type

    # -----------------------------------------------------------------
    # Build payload for start_pipeline
    # -----------------------------------------------------------------
    pipeline_payload = {
        "asset_id": asset_id,
        "source": "getty",
        "media_type": media_type,         # "image" or "video"
        "media_bytes": media_b64,
        "getty_metadata": getty_metadata,
    }

    pipeline_result = call_start_pipeline(pipeline_payload)

    return {
        "asset_id": asset_id,
        "query": query,
        "media_type": media_type,
        "status": "processed",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "pipeline_result": pipeline_result,
    }
