import os
import requests
from fastapi import FastAPI, Request, HTTPException
from datetime import datetime
import google.auth.transport.requests
import google.oauth2.id_token

app = FastAPI()

# Downstream service URLs
GETTY_URL = os.getenv("GETTY_URL")
TRANSCODE_URL = os.getenv("TRANSCODE_URL")
TRANSCRIBE_URL = os.getenv("TRANSCRIBE_URL")
FRAMES_URL = os.getenv("FRAMES_URL")
QWEN_URL = os.getenv("QWEN_URL")
STORE_URL = os.getenv("STORE_URL")


# ---------------------------------------------------------
# AUTHENTICATED CLOUD RUN CALL
# ---------------------------------------------------------
def call_service(url: str, endpoint: str, payload: dict) -> dict:
    """Call a Cloud Run service using ID token authentication."""
    if not url:
        raise HTTPException(status_code=500, detail=f"Missing service URL for {endpoint}")

    auth_req = google.auth.transport.requests.Request()
    token = google.oauth2.id_token.fetch_id_token(auth_req, url)

    resp = requests.post(
        f"{url}/{endpoint}",
        json=payload,
        headers={"Authorization": f"Bearer {token}"}
    )

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    return resp.json()


# ---------------------------------------------------------
# INPUT NORMALISATION LAYER
# ---------------------------------------------------------
def build_initial_payload(data: dict) -> dict:
    """
    Normalise input into a unified payload for the pipeline.

    Supported input sources:
    - Getty: {"asset_id": "getty-123456"}
    - Local/GCS: {"file_name": "...", "bucket": "..."}
    - Upload: {"file_name": "...", "bucket": "...", "source": "upload"}
    - Google Drive: {"file_name": "...", "bucket": "...", "source": "drive"}
    - Camera/Camcorder: {"file_name": "...", "bucket": "...", "source": "camera"}
    - Other APIs: {"file_name": "...", "bucket": "...", "source": "other_api"}
    """

    # -----------------------------
    # MODE 1: GETTY INGESTION
    # -----------------------------
    if "asset_id" in data:
        asset_id = data["asset_id"]
        return call_service(GETTY_URL, "validate", {"asset_id": asset_id})

    # -----------------------------
    # MODE 2: GENERIC GCS INGESTION
    # (upload, drive, camera, other APIs)
    # -----------------------------
    if "file_name" in data and "bucket" in data:
        file_name = data["file_name"]
        bucket = data["bucket"]

        # Default to "local" if no explicit source is provided
        source = data.get("source", "local")

        # Derive asset_id from filename
        asset_id = os.path.splitext(os.path.basename(file_name))[0]

        return {
            "asset_id": asset_id,
            "file_name": file_name,
            "bucket": bucket,
            "source": source,  # upload | drive | camera | other_api | local
            "paths": {
                "raw": f"gs://{bucket}/{file_name}"
            }
        }

    # -----------------------------
    # INVALID INPUT
    # -----------------------------
    raise HTTPException(
        status_code=400,
        detail=(
            "Invalid input. Provide either:\n"
            "- asset_id (Getty mode), OR\n"
            "- file_name + bucket (local/upload/drive/camera/other_api)"
        )
    )


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# FULL PIPELINE ORCHESTRATION
# ---------------------------------------------------------
@app.post("/run_all")
async def run_all(req: Request):
    """Full pipeline orchestration."""
    data = await req.json()

    # Step 0: Normalise input
    payload = build_initial_payload(data)

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
