import os
import json
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime

import google.auth.transport.requests
import google.oauth2.id_token

app = FastAPI()

# -------------------------------------------------------------------
# Environment
# -------------------------------------------------------------------
# Getty helper service that can search + download (your existing service or proxy)
GETTY_HELPER_URL = os.getenv("GETTY_HELPER_URL")  # e.g. https://getty-helper-service-...run.app
# Start-pipeline service
PIPELINE_URL = os.getenv("PIPELINE_URL")          # e.g. https://start-pipeline-...run.app/run_all

if not GETTY_HELPER_URL or not PIPELINE_URL:
    raise RuntimeError("GETTY_HELPER_URL and PIPELINE_URL must be set.")


# -------------------------------------------------------------------
# Authenticated Cloud Run call
# -------------------------------------------------------------------
def call_cloud_run(url: str, json_payload: dict) -> requests.Response:
    """
    Call a Cloud Run service with an identity token.
    Returns the raw Response so caller can inspect status/text/json.
    """
    auth_req = google.auth.transport.requests.Request()
    token = google.oauth2.id_token.fetch_id_token(auth_req, url)

    resp = requests.post(
        url,
        json=json_payload,
        headers={"Authorization": f"Bearer {token}"},
        timeout=300,
    )
    return resp


# -------------------------------------------------------------------
# Getty helper calls
# -------------------------------------------------------------------
def getty_search(query: str) -> dict:
    """
    Call your Getty helper to search for assets.
    Expected to return JSON with at least one asset and an asset_id.
    """
    url = f"{GETTY_HELPER_URL}/search"
    resp = requests.get(url, params={"q": query}, timeout=60)

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Getty search failed: {resp.text[:1000]}",
        )

    data = resp.json()
    return data


def getty_download(asset_id: str) -> tuple[int, Optional[str]]:
    """
    Call your Getty helper to download a specific asset.
    Expected to:
      - fetch from Getty delivery URL
      - store into GCS
      - return something like { "media_url": "gs://bucket/getty/<id>.mp4" }
    We also surface the raw download attempt status/body for debugging.
    """
    url = f"{GETTY_HELPER_URL}/download"
    resp = requests.post(url, json={"asset_id": asset_id}, timeout=300)

    status = resp.status_code
    body = resp.text

    if status != 200:
        # We still return status/body so the ingestor response can show it
        return status, body

    return status, body


# -------------------------------------------------------------------
# Health
# -------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "services": {
            "getty_helper": GETTY_HELPER_URL,
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

    # You can adapt this to your actual search response shape
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

    # We surface these regardless of success/failure
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
    pipeline_payload = {
        "media_url": media_url,
        "asset_id": asset_id,
        "source": "getty",
        "media_type": "video",  # Getty clips here are video; adjust if needed
    }

    pipeline_resp = call_cloud_run(PIPELINE_URL, pipeline_payload)
    pipeline_status = pipeline_resp.status_code

    # We do NOT special-case audio at all here.
    # Audio is fully optional and handled inside the pipeline/transcoder.
    # We just surface whatever the pipeline returns.
    try:
        pipeline_json = pipeline_resp.json()
        pipeline_preview = json.dumps(pipeline_json)[:2000]
    except Exception:
        pipeline_preview = pipeline_resp.text[:2000]

    error = None
    if pipeline_status != 200:
        error = "Pipeline returned non-200 status."

    return {
        "stage": "complete" if pipeline_status == 200 else "pipeline_error",
        "query": q,
        "asset_id": asset_id,
        "pipeline_status": pipeline_status,
        "pipeline_preview": pipeline_preview,
        "error": error,
        "download_attempt_status": download_attempt_status,
        "download_attempt_body": download_attempt_body,
    }
