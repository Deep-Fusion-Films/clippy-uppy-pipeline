import time
import logging
from typing import Optional, Dict, Any

import requests
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from pydantic_settings import BaseSettings

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("getty_ingestor_service")


# -------------------------------------------------------------------
# Settings (Pydantic v2, Cloud Run compatible)
# -------------------------------------------------------------------
class Settings(BaseSettings):
    getty_api_key: str
    getty_api_secret: str
    start_pipeline_url: str

    getty_search_page_size: int = 25
    getty_search_max_pages: int = 5
    getty_timeout_seconds: int = 10
    getty_max_retries: int = 3
    getty_backoff_seconds: float = 0.5

    model_config = {
        "env_file": None,
        "env_prefix": "",
        "case_sensitive": True,
        "extra": "allow",
    }


def get_settings() -> Settings:
    """Load settings at request time (Cloud Run safe)."""
    return Settings()


# -------------------------------------------------------------------
# FastAPI App
# -------------------------------------------------------------------
app = FastAPI(title="Getty Ingestor Service", version="1.0.0")

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
    download_attempt_status: Optional[int] = None
    download_attempt_body: Optional[str] = None


# -------------------------------------------------------------------
# HTTP helper with retry
# -------------------------------------------------------------------
def http_request_with_retry(
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    timeout: int = 10,
    max_retries: int = 3,
    backoff_seconds: float = 0.5,
) -> requests.Response:

    attempt = 0
    while True:
        attempt += 1
        try:
            return requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                data=data,
                json=json_body,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            if attempt >= max_retries:
                raise
            logger.warning(
                "HTTP error on %s %s (attempt %d/%d): %s",
                method, url, attempt, max_retries, exc
            )
            time.sleep(backoff_seconds)


# -------------------------------------------------------------------
# Getty OAuth Token
# -------------------------------------------------------------------
def get_getty_access_token(settings: Settings) -> str:
    global _token_cache
    now = time.time()

    if _token_cache and now < _token_cache["expires_at"]:
        return _token_cache["access_token"]

    url = "https://authentication.gettyimages.com/oauth2/token"
    data = {
        "client_id": settings.getty_api_key,
        "client_secret": settings.getty_api_secret,
        "grant_type": "client_credentials",
    }

    resp = http_request_with_retry(
        method="POST",
        url=url,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data=data,
        timeout=settings.getty_timeout_seconds,
        max_retries=settings.getty_max_retries,
        backoff_seconds=settings.getty_backoff_seconds,
    )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Getty OAuth failed: {resp.text[:500]}",
        )

    body = resp.json()
    token = body["access_token"]
    expires_in = int(body.get("expires_in", 3600))

    _token_cache = {
        "access_token": token,
        "expires_at": now + expires_in - 60,
    }

    return token


# -------------------------------------------------------------------
# Licensed Download Attempt
# -------------------------------------------------------------------
def attempt_licensed_download(asset_id: str, settings: Settings) -> Dict[str, Any]:
    url = f"https://api.gettyimages.com/v3/downloads/videos/{asset_id}"

    resp = http_request_with_retry(
        method="POST",
        url=url,
        headers={
            "Api-Key": settings.getty_api_key,
            "Authorization": f"Bearer {get_getty_access_token(settings)}",
            "Accept": "application/json",
        },
        timeout=settings.getty_timeout_seconds,
        max_retries=settings.getty_max_retries,
        backoff_seconds=settings.getty_backoff_seconds,
    )

    return {
        "status": resp.status_code,
        "body": resp.text[:500],
    }


# -------------------------------------------------------------------
# Preview MP4 Extraction
# -------------------------------------------------------------------
def extract_preview_mp4(video: Dict[str, Any]) -> Optional[str]:
    for d in video.get("display_sizes", []) or []:
        uri = d.get("uri")
        if isinstance(uri, str) and uri.lower().endswith(".mp4"):
            return uri
    return None


# -------------------------------------------------------------------
# Getty Search
# -------------------------------------------------------------------
def find_first_usable_asset(query: str, settings: Settings) -> Dict[str, Any]:
    base_url = "https://api.gettyimages.com/v3/search/videos"

    headers = {
        "Api-Key": settings.getty_api_key,
        "Authorization": f"Bearer {get_getty_access_token(settings)}",
        "Accept": "application/json",
    }

    for page in range(1, settings.getty_search_max_pages + 1):
        resp = http_request_with_retry(
            method="GET",
            url=base_url,
            headers=headers,
            params={
                "phrase": query,
                "fields": "id,title,caption,display_sizes",
                "page_size": settings.getty_search_page_size,
                "page": page,
                "sort_order": "best_match",
            },
            timeout=settings.getty_timeout_seconds,
            max_retries=settings.getty_max_retries,
            backoff_seconds=settings.getty_backoff_seconds,
        )

        if resp.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Getty search failed on page {page}: {resp.text[:500]}",
            )

        videos = resp.json().get("videos", []) or []
        if not videos:
            continue

        for video in videos:
            asset_id = video.get("id")
            if not asset_id:
                continue

            download_attempt = attempt_licensed_download(asset_id, settings)

            if download_attempt["status"] == 200:
                return {
                    "asset": video,
                    "asset_id": asset_id,
                    "download_attempt": download_attempt,
                    "download_url": download_attempt["body"],
                    "preview_url": None,
                }

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
        detail=f"No usable assets found for '{query}' across {settings.getty_search_max_pages} pages.",
    )


# -------------------------------------------------------------------
# Trigger Pipeline
# -------------------------------------------------------------------
def trigger_pipeline(asset_id: str, metadata: Dict[str, Any], url: str, settings: Settings):
    resp = http_request_with_retry(
        method="POST",
        url=settings.start_pipeline_url,
        json_body={
            "asset_id": asset_id,
            "getty_metadata": metadata,
            "download_url": url,
        },
        timeout=settings.getty_timeout_seconds,
        max_retries=settings.getty_max_retries,
        backoff_seconds=settings.getty_backoff_seconds,
    )
    return resp


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/debug/env")
def debug_env(settings: Settings = Depends(get_settings)):
    return {
        "getty_api_key": settings.getty_api_key,
        "getty_api_secret": settings.getty_api_secret,
        "start_pipeline_url": settings.start_pipeline_url,
    }


@app.get("/debug/raw")
def debug_raw():
    import os
    return {"env": dict(os.environ)}


@app.get("/search-and-run", response_model=SearchAndRunResponse)
def search_and_run(q: str, settings: Settings = Depends(get_settings)):

    try:
        result = find_first_usable_asset(q, settings)
    except Exception as e:
        return SearchAndRunResponse(stage="getty_search", query=q, error=str(e))

    asset = result["asset"]
    asset_id = result["asset_id"]
    download_attempt = result["download_attempt"]
    usable_url = result["download_url"] or result["preview_url"]

    try:
        pipeline_resp = trigger_pipeline(asset_id, asset, usable_url, settings)
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
