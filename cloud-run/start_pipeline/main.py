import os
import requests
from fastapi import FastAPI, Request, HTTPException
from datetime import datetime
import google.auth.transport.requests
import google.oauth2.id_token

app = FastAPI()

# Environment variables: service URLs
GETTY_URL = os.getenv("GETTY_URL")
TRANSCODE_URL = os.getenv("TRANSCODE_URL")
TRANSCRIBE_URL = os.getenv("TRANSCRIBE_URL")
FRAMES_URL = os.getenv("FRAMES_URL")
QWEN_URL = os.getenv("QWEN_URL")
STORE_URL = os.getenv("STORE_URL")

def call_service(url: str, endpoint: str, payload: dict) -> dict:
    """
    Call a downstream Cloud Run service with ID token authentication.
    """
    if not url:
        raise HTTPException(status_code=500, detail=f"Missing service URL for {endpoint}")
    request = google.auth.transport.requests.Request()
    id_token = google.oauth2.id_token.fetch_id_token(request, url)
    resp = requests.post(
        f"{url}/{endpoint}",
        json=payload,
        headers={"Authorization": f"Bearer {id_token}"}
    )
    resp.raise_for_status()
    return resp.json()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "services": {
            "getty": GETTY_URL,
            "transcode": TRANSCODE_URL,
            "transcribe": TRANSCRIBE_URL,
            "frames": FRAMES_URL,
            "qwen": QWEN_URL,
            "store": STORE_URL
        }
    }

@app.post("/run_all")
async def run_all(req: Request):
    """
    Orchestrates the full pipeline.
    Input:
      Getty: {"asset_id":"getty-123456"}
      Local: {"file_name":"raw/CE_025_0.mp4","bucket":"df-films-assets-euw1"}
    """
    data = await req.json()

    # Mode 1: Getty asset_id
    if "asset_id" in data:
        asset_id = data["asset_id"]
        getty_json = call_service(GETTY_URL, "validate", {"asset_id": asset_id})
        payload = getty_json

    # Mode 2: Local file_name + bucket
    elif "file_name" in data and "bucket" in data:
        file_name = data["file_name"]
        bucket = data["bucket"]
        payload = {"file_name": file_name, "bucket": bucket}

    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either asset_id (Getty) OR file_name+bucket (local)"
        )

    # Step 1: Transcode
    transcode_json = call_service(TRANSCODE_URL, "transcode", payload)

    # Step 2: Transcribe
    transcribe_json = call_service(TRANSCRIBE_URL, "transcribe", transcode_json)

    # Step 3: Sample frames
    frames_json = call_service(FRAMES_URL, "sample", transcribe_json)

    # Step 4: Qwen enrichment
    qwen_json = call_service(QWEN_URL, "enrich", frames_json)

    # Step 5: Store metadata
    stored_json = call_service(STORE_URL, "store", qwen_json)

    return {
        "pipeline": "complete",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "result": stored_json
    }
