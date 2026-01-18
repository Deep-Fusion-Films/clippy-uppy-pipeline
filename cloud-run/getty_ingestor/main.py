import time
import logging
import json
from typing import Optional, Dict, Any

import requests
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from google.auth.transport.requests import Request
from google.oauth2 import id_token
import google.auth
from google.cloud import storage
import random

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

    # NEW: pipeline timeout (default 5 minutes)
    pipeline_timeout_seconds: int = 300

    model_config = {
        "env_file": None,
        "env_prefix": "",
        "case_sensitive": False,
        "extra": "allow",
    }


def get_settings() -> Settings:
    return Settings()


# -------------------------------------------------------------------
# Cloud Run → Cloud Run ID Token Auth
# -------------------------------------------------------------------
def get_id_token(audience: str) -> str:
    creds, _ = google.auth.default()
    auth_req = Request()
    return id_token.fetch_id_token(auth_req, audience)


# -------------------------------------------------------------------
# FastAPI App
# -------------------------------------------------------------------
app = FastAPI(title="Getty Ingestor Service", version="1.0.0")

_token_cache: Optional[Dict[str, Any]] = None


# -------------------------------------------------------------------
# Used IDs tracking (GCS-backed, for non-repeating assets)
# -------------------------------------------------------------------
USED_IDS_BUCKET = "df-films-assets-euw1"
USED_IDS_BLOB = "getty/used_ids.json"


def load_used_ids() -> set[str]:
    client = storage.Client()
    bucket = client.bucket(USED_IDS_BUCKET)
    blob = bucket.blob(USED_IDS_BLOB)

    if not blob.exists():
        return set()

    data = blob.download_as_text()
    try:
        return set(json.loads(data))
    except Exception:
        logger.warning("Failed to parse used_ids JSON, resetting.")
        return set()


def save_used_ids(used_ids: set[str]) -> None:
    client = storage.Client()
    bucket = client.bucket(USED_IDS_BUCKET)
    blob = bucket.blob(USED_IDS_BLOB)
    blob.upload_from_string(json.dumps(list(used_ids)))


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
# GCS Upload Helper (streaming)
# -------------------------------------------------------------------
def upload_to_gcs(bucket_name: str, blob_name: str, source_url: str) -> str:
    with requests.get(source_url, stream=True) as resp:
        resp.raise_for_status()

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        blob.upload_from_file(resp.raw, content_type="video/mp4")

    return f"gs://{bucket_name}/{blob_name}"


# -------------------------------------------------------------------
# Getty Search (random + non-repeating)
# -------------------------------------------------------------------
def find_first_usable_asset(query: str, settings: Settings) -> Dict[str, Any]:
    base_url = "https://api.gettyimages.com/v3/search/videos"

    headers = {
        "Api-Key": settings.getty_api_key,
        "Authorization": f"Bearer {get_getty_access_token(settings)}",
        "Accept": "application/json",
    }

    used_ids = load_used_ids()

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

        random.shuffle(videos)

        for video in videos:
            asset_id = video.get("id")
            if not asset_id:
                continue

            if asset_id in used_ids:
                continue

            download_attempt = attempt_licensed_download(asset_id, settings)

            if download_attempt["status"] == 200:
                used_ids.add(asset_id)
                save_used_ids(used_ids)
                return {
                    "asset": video,
                    "asset_id": asset_id,
                    "download_attempt": download_attempt,
                    "download_url": download_attempt["body"],
                    "preview_url": None,
                }

            preview_url = extract_preview_mp4(video)
            if preview_url:
                used_ids.add(asset_id)
                save_used_ids(used_ids)
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
# Trigger Pipeline (with extended timeout)
# -------------------------------------------------------------------
def trigger_pipeline(asset_id: str, metadata: Dict[str, Any], url: str, settings: Settings):

    if url.strip().startswith("{"):
        try:
            url = json.loads(url)["uri"]
        except Exception:
            raise HTTPException(status_code=500, detail="Failed to parse Getty download URL")

    gcs_blob = f"getty/{asset_id}.mp4"
    gcs_url = upload_to_gcs("df-films-assets-euw1", gcs_blob, url)

    payload = {
        "media_url": gcs_url,
        "media_type": "video",
        "getty_metadata": metadata,
        "source": "getty",
        "asset_id": asset_id,
    }

    id_tok = get_id_token(settings.start_pipeline_url)

    logger.info(f"Calling pipeline URL: {settings.start_pipeline_url}")

    resp = http_request_with_retry(
        method="POST",
        url=settings.start_pipeline_url,
        headers={"Authorization": f"Bearer {id_tok}"},
        json_body=payload,

        # NEW: use extended timeout
        timeout=settings.pipeline_timeout_seconds,

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
def debug_env(settings: Settings = Depends(get_settings, use_cache=False)):
    return {
        "getty_api_key": settings.getty_api_key,
        "getty_api_secret": settings.getty_api_secret,
        "start_pipeline_url": settings.start_pipeline_url,
        "pipeline_timeout_seconds": settings.pipeline_timeout_seconds,
    }


@app.get("/debug/raw")
def debug_raw():
    import os
    return {"env": dict(os.environ)}


@app.get("/search-and-run", response_model=SearchAndRunResponse)
def search_and_run(q: str, settings: Settings = Depends(get_settings, use_cache=False)):

    try:
        result = find_first_usable_asset(q, settings)
    except Exception as e:
        return SearchAndRunResponse(stage="getty_search", query=q, error=str(e))

    asset = result["asset"]
    asset_id = result["asset_id"]
    download_attempt = result["download_attempt"]

    raw_url = result["download_url"] or result["preview_url"]

    if raw_url and raw_url.strip().startswith("{"):
        try:
            usable_url = json.loads(raw_url)["uri"]
        except Exception:
            return SearchAndRunResponse(
                stage="pipeline_trigger",
                query=q,
                asset_id=asset_id,
                download_attempt_status=download_attempt["status"],
                download_attempt_body=download_attempt["body"],
                error="Failed to parse Getty download URL",
            )
    else:
        usable_url = raw_url

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
