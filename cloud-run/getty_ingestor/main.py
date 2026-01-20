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
# Existing envs you already use in this project
GETTY_URL = os.getenv("GETTY_URL")        # e.g. https://getty-helper-service-...run.app
PIPELINE_URL = os.getenv("PIPELINE_URL")  # e.g. https://start-pipeline-service-...run.app

# NOTE: we no longer raise at import time if these are missing.
# /health will expose them so you can see misconfig without killing the container.


# -------------------------------------------------------------------
# Authenticated Cloud Run call (for pipeline)
# -------------------------------------------------------------------
def call_cloud_run(url: str, endpoint: str, json_payload: dict) -> requests.Response:
    """
    Call a Cloud Run service with an identity token.
    Returns the raw Response so caller can inspect status/text/json.
    """
    if not url:
        raise HTTPException(status_code=500, detail=f"Missing base URL for {endpoint}")

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
# Getty helper calls (using your existing GETTY_URL)
# -------------------------------------------------------------------
def getty_search(query: str) -> dict:
    """
    Call your existing Getty helper/search endpoint.
    Expected to return JSON with a list of assets.
    """
    if not GETTY_URL:
        raise HTTPException(status_code=500, detail="GETTY_URL is not configured")

    url = f"{GETTY_URL.rstrip('/')}/search"
    resp = requests.get(url, params={"q": query}, timeout=60)

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Getty search failed: {resp.text[:1000]}",
        )

    return resp.json()


def getty_download(asset_id: str) -> tuple[int, Optional[str]]:
    """
    Call your existing Getty download helper.
    Expected to:
      - fetch from Getty delivery URL
      - store into GCS
      - return JSON like { "media_url": "gs://bucket/getty/<id>.mp4" }
    We also surface the raw status/body for debugging.
    """
    if not GETTY_URL:
        return 500, "GETTY_URL is not configured"

    url = f"{GETTY_URL.rstrip('/')}/download"
    resp = requests.post(url, json={"asset_id": asset_id}, timeout=300)

    return resp.status_code, resp.text


# -------------------------------------------------------------------
# Health
# -------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "services": {
            "getty": GETTY_URL,
            "pipeline": PIPELINE_URL,
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# -------------------------------------------------------------------
# Search + run pipeline
# -------------------------------------------------------------------
@app.get("/search-and-run")
def search_and_run(q: str):
    """
    1. Search Getty for the query.
    2. Pick the first asset.
    3. Ask Getty helper to download it (into GCS).
    4. Trigger the start-pipeline /run_all endpoint with media_url + asset_id.
    5. Return a compact summary including pipeline status + preview.

    Audio is treated as fully optional and is handled inside the pipeline;
    this service does NOT special-case audio at all.
    """

    # ---------------------------------------------------------------
    # 1. Search Getty
    # ---------------------------------------------------------------
    try:
        search_json = getty_search(q)
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={
                "stage": "search_failed",
                "query": q,
                "asset_id": None,
                "pipeline_status": None,
                "pipeline_preview": None,
                "error": str(e.detail),
                "download_attempt_status": None,
                "download_attempt_body": None,
            },
        )

    # Adapt to your actual search response shape
    assets = search_json.get("assets") or search_json.get("results") or []
    if not assets:
        return {
            "stage": "no_results",
            "query": q,
            "asset_id": None,
            "pipeline_status": None,
            "pipeline_preview": None,
            "error": "No Getty assets found for query.",
            "download_attempt_status": None,
            "download_attempt_body": None,
        }

    first = assets[0]
    asset_id = str(first.get("id") or first.get("asset_id") or "unknown")

    # ---------------------------------------------------------------
    # 2. Download via Getty helper
    # ---------------------------------------------------------------
    download_status, download_body = getty_download(asset_id)

    download_attempt_status = download_status
    download_attempt_body = download_body

    if download_status != 200:
        return {
            "stage": "download_failed",
            "query": q,
            "asset_id": asset_id,
            "pipeline_status": None,
            "pipeline_preview": None,
            "error": "Getty download helper failed.",
            "download_attempt_status": download_attempt_status,
            "download_attempt_body": download_attempt_body,
        }

    # Expect the helper to return JSON with media_url
    try:
        download_json = json.loads(download_body)
    except json.JSONDecodeError:
        return {
            "stage": "download_failed",
            "query": q,
            "asset_id": asset_id,
            "pipeline_status": None,
            "pipeline_preview": None,
            "error": "Download helper returned non-JSON body.",
            "download_attempt_status": download_attempt_status,
            "download_attempt_body": download_attempt_body,
        }

    media_url = download_json.get("media_url")
    if not media_url:
        return {
            "stage": "download_failed",
            "query": q,
            "asset_id": asset_id,
            "pipeline_status": None,
            "pipeline_preview": None,
            "error": "Download helper did not return media_url.",
            "download_attempt_status": download_attempt_status,
            "download_attempt_body": download_attempt_body,
        }

    # ---------------------------------------------------------------
    # 3. Trigger pipeline (run_all)
    # ---------------------------------------------------------------
    if not PIPELINE_URL:
        return {
            "stage": "pipeline_misconfigured",
            "query": q,
            "asset_id": asset_id,
            "pipeline_status": None,
            "pipeline_preview": None,
            "error": "PIPELINE_URL is not configured.",
            "download_attempt_status": download_attempt_status,
            "download_attempt_body": download_attempt_body,
        }

    pipeline_payload = {
        "media_url": media_url,
        "asset_id": asset_id,
        "source": "getty",
        "media_type": "video",  # adjust if you later support images/audio here
    }

    try:
        pipeline_resp = call_cloud_run(PIPELINE_URL, "run_all", pipeline_payload)
    except HTTPException as e:
        return {
            "stage": "pipeline_error",
            "query": q,
            "asset_id": asset_id,
            "pipeline_status": e.status_code,
            "pipeline_preview": str(e.detail)[:2000],
            "error": "Pipeline call failed.",
            "download_attempt_status": download_attempt_status,
            "download_attempt_body": download_attempt_body,
        }

    pipeline_status = pipeline_resp.status_code

    try:
        pipeline_json = pipeline_resp.json()
        pipeline_preview = json.dumps(pipeline_json)[:2000]
    except Exception:
        pipeline_preview = pipeline_resp.text[:2000]

    error = None
    stage = "complete"
    if pipeline_status != 200:
        error = "Pipeline returned non-200 status."
        stage = "pipeline_error"

    return {
        "stage": stage,
        "query": q,
        "asset_id": asset_id,
        "pipeline_status": pipeline_status,
        "pipeline_preview": pipeline_preview,
        "error": error,
        "download_attempt_status": download_attempt_status,
        "download_attempt_body": download_attempt_body,
    }
