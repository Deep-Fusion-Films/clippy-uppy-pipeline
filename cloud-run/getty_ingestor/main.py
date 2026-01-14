import time
import logging
from typing import Optional, Dict, Any, List

import requests
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# -------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("getty_ingestor_service")


# -------------------------------------------------------------------
# Settings / Environment (Pydantic v2-friendly)
# -------------------------------------------------------------------
class Settings(BaseSettings):
    getty_api_key: str = Field(..., env="GETTY_API_KEY")
    getty_api_secret: str = Field(..., env="GETTY_API_SECRET")
    start_pipeline_url: str = Field(..., env="START_PIPELINE_URL")

    # Tunables
    getty_search_page_size: int = 25
    getty_search_max_pages: int = 5
    getty_timeout_seconds: int = 10
    getty_max_retries: int = 3
    getty_backoff_seconds: float = 0.5

    model_config = {
        "case_sensitive": True
    }


def get_settings() -> Settings:
    try:
        s = Settings()
        logger.info(
            "Settings loaded: page_size=%d max_pages=%d timeout=%d",
            s.getty_search_page_size,
            s.getty_search_max_pages,
            s.getty_timeout_seconds,
        )
        return s
    except Exception as e:
        logger.error("Settings validation failed: %s", e)
        # Let this bubble up so Cloud Run logs show the full Pydantic error
        raise


settings = get_settings()

app = FastAPI(title="Getty Ingestor Service", version="1.0.0")

# OAuth token cache (in-memory, per-container)
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
    """
    Simple synchronous HTTP helper with basic retry behavior for transient errors.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                data=data,
                json=json_body,
                timeout=timeout,
            )
            return resp
        except requests.RequestException as exc:
            logger.warning(
                "HTTP request error on %s %s (attempt %d/%d): %s",
                method,
                url,
                attempt,
                max_retries,
                exc,
            )
            if attempt >= max_retries:
                raise
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
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    logger.info("Requesting new Getty OAuth token")
    resp = http_request_with_retry(
        method="POST",
        url=url,
        headers=headers,
        data=data,
        timeout=settings.getty_timeout_seconds,
        max_retries=settings.getty_max_retries,
        backoff_seconds=settings.getty_backoff_seconds,
    )

    if resp.status_code != 200:
        logger.error(
            "Getty OAuth failed: status=%d body=%s",
            resp.status_code,
            resp.text[:500],
        )
        raise HTTPException(
            status_code=502,
            detail=f"Getty OAuth failed: {resp.text[:500]}",
        )

    body = resp.json()
    access_token = body["access_token"]
    expires_in = int(body.get("expires_in", 3600))

    _token_cache = {
        "access_token": access_token,
        "expires_at": now + expires_in - 60,  # expire slightly early
    }

    logger.info("Getty OAuth token acquired, expires_in=%d", expires_in)
    return access_token


# -------------------------------------------------------------------
# Attempt Licensed Download (Strict Mode)
# -------------------------------------------------------------------
def attempt_licensed_download(asset_id: str, settings: Settings) -> Dict[str, Any]:
    url = f"https://api.gettyimages.com/v3/downloads/videos/{asset_id}"

    headers = {
        "Api-Key": settings.getty_api_key,
        "Authorization": f"Bearer {get_getty_access_token(settings)}",
        "Accept": "application/json",
    }

    logger.info("Attempting licensed download for asset_id=%s", asset_id)

    resp = http_request_with_retry(
        method="POST",
        url=url,
        headers=headers,
        timeout=settings.getty_timeout_seconds,
        max_retries=settings.getty_max_retries,
        backoff_seconds=settings.getty_backoff_seconds,
    )

    return {
        "status": resp.status_code,
        "body": resp.text[:500],
    }


# -------------------------------------------------------------------
# Extract Preview MP4 (P2: first MP4 returned)
# -------------------------------------------------------------------
def extract_preview_mp4(video: Dict[str, Any]) -> Optional[str]:
    display_sizes = video.get("display_sizes", []) or []
    for d in display_sizes:
        uri = d.get("uri")
        if isinstance(uri, str) and uri.lower().endswith(".mp4"):
            return uri
    return None


# -------------------------------------------------------------------
# Search Core Video API in Blocks of N, Across M Pages
# -------------------------------------------------------------------
def find_first_usable_asset(query: str, settings: Settings) -> Dict[str, Any]:
    base_url = "https://api.gettyimages.com/v3/search/videos"

    headers = {
        "Api-Key": settings.getty_api_key,
        "Authorization": f"Bearer {get_getty_access_token(settings)}",
        "Accept": "application/json",
    }

    logger.info(
        "Searching Getty Core Video: query=%s page_size=%d max_pages=%d",
        query,
        settings.getty_search_page_size,
        settings.getty_search_max_pages,
    )

    for page in range(1, settings.getty_search_max_pages + 1):
        params = {
            "phrase": query,
            "fields": "id,title,caption,display_sizes",
            "page_size": settings.getty_search_page_size,
            "page": page,
            "sort_order": "best_match",
        }

        logger.info("Getty search page=%d", page)
        resp = http_request_with_retry(
            method="GET",
            url=base_url,
            headers=headers,
            params=params,
            timeout=settings.getty_timeout_seconds,
            max_retries=settings.getty_max_retries,
            backoff_seconds=settings.getty_backoff_seconds,
        )

        if resp.status_code != 200:
            logger.error(
                "Getty search failed on page %d: status=%d body=%s",
                page,
                resp.status_code,
                resp.text[:500],
            )
            raise HTTPException(
                status_code=502,
                detail=f"Getty search failed on page {page}: {resp.text[:500]}",
            )

        payload = resp.json()
        videos: List[Dict[str, Any]] = payload.get("videos", []) or []
        logger.info("Getty page=%d returned %d videos", page, len(videos))

        if not videos:
            continue

        for video in videos:
            asset_id = video.get("id")
            if not asset_id:
                continue

            # 1. Strict Mode: Attempt licensed download
            download_attempt = attempt_licensed_download(asset_id, settings)

            if download_attempt["status"] == 200:
                logger.info(
                    "Licensed download succeeded for asset_id=%s", asset_id
                )
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
                logger.info(
                    "Using preview MP4 fallback for asset_id=%s", asset_id
                )
                return {
                    "asset": video,
                    "asset_id": asset_id,
                    "download_attempt": download_attempt,
                    "download_url": None,
                    "preview_url": preview_url,
                }

    logger.warning(
        "No usable assets found for query=%s across %d pages",
        query,
        settings.getty_search_max_pages,
    )
    raise HTTPException(
        status_code=404,
        detail=(
            f"No usable assets (download or preview) found for "
            f"'{query}' across {settings.getty_search_max_pages} pages."
        ),
    )


# -------------------------------------------------------------------
# Trigger Pipeline
# -------------------------------------------------------------------
def trigger_pipeline(
    asset_id: str,
    metadata: Dict[str, Any],
    url: str,
    settings: Settings,
) -> requests.Response:
    payload = {
        "asset_id": asset_id,
        "getty_metadata": metadata,
        "download_url": url,
    }

    logger.info(
        "Triggering pipeline: url=%s asset_id=%s",
        settings.start_pipeline_url,
        asset_id,
    )

    resp = http_request_with_retry(
        method="POST",
        url=settings.start_pipeline_url,
        json_body=payload,
        timeout=settings.getty_timeout_seconds,
        max_retries=settings.getty_max_retries,
        backoff_seconds=settings.getty_backoff_seconds,
    )

    logger.info(
        "Pipeline response: status=%d body_preview=%s",
        resp.status_code,
        resp.text[:300],
    )
    return resp


# -------------------------------------------------------------------
# Dependencies
# -------------------------------------------------------------------
def get_service_dependencies() -> Settings:
    return settings


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/search-and-run", response_model=SearchAndRunResponse)
def search_and_run(q: str, settings: Settings = Depends(get_service_dependencies)):
    logger.info("Received /search-and-run request: query=%s", q)

    try:
        result = find_first_usable_asset(q, settings)
    except HTTPException as http_err:
        logger.warning(
            "Getty search pipeline failed for query=%s: %s",
            q,
            http_err.detail,
        )
        return SearchAndRunResponse(
            stage="getty_search",
            query=q,
            error=str(http_err.detail),
        )
    except Exception as e:
        logger.exception(
            "Unexpected error during Getty search for query=%s", q
        )
        return SearchAndRunResponse(
            stage="getty_search",
            query=q,
            error=f"Unexpected error: {e}",
        )

    asset = result["asset"]
    asset_id = result["asset_id"]
    download_attempt = result["download_attempt"]
    download_url = result["download_url"]
    preview_url = result["preview_url"]

    usable_url = download_url or preview_url

    if not usable_url:
        logger.error(
            "No usable URL found for asset_id=%s after selection", asset_id
        )
        return SearchAndRunResponse(
            stage="selection",
            query=q,
            asset_id=asset_id,
            download_attempt_status=download_attempt["status"],
            download_attempt_body=download_attempt["body"],
            error="No usable URL found for selected asset.",
        )

    try:
        pipeline_resp = trigger_pipeline(asset_id, asset, usable_url, settings)
    except Exception as e:
        logger.exception(
            "Error triggering pipeline for asset_id=%s", asset_id
        )
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
