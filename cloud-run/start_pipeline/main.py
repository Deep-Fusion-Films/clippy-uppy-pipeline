import os
import requests
from fastapi import FastAPI, Request, HTTPException
from datetime import datetime
import google.auth.transport.requests
import google.oauth2.id_token
import copy
import base64

app = FastAPI()

# Downstream service URLs
GETTY_URL = os.getenv("GETTY_URL")
TRANSCODE_URL = os.getenv("TRANSCODE_URL")
TRANSCRIBE_URL = os.getenv("TRANSCRIBE_URL")
FRAMES_URL = os.getenv("FRAMES_URL")
ENRICHER_URL = os.getenv("ENRICHER_URL")  # renamed from QWEN_URL
STORE_URL = os.getenv("STORE_URL")


# ---------------------------------------------------------
# ASSET TYPE DETECTION
# ---------------------------------------------------------
def detect_asset_type_from_filename(file_name: str) -> str:
    ext = os.path.splitext(file_name.lower())[1]
    if ext in [".mp4", ".mov", ".mkv", ".avi"]:
        return "video"
    if ext in [".jpg", ".jpeg", ".png", ".webp", ".tiff"]:
        return "image"
    if ext in [".mp3", ".wav", ".aac"]:
        return "audio"
    return "unknown"


# ---------------------------------------------------------
# AUTHENTICATED CLOUD RUN CALL
# ---------------------------------------------------------
def call_service(url: str, endpoint: str, payload: dict) -> dict:
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
# DEEP MERGE UTILITY
# ---------------------------------------------------------
def deep_merge(a: dict, b: dict) -> dict:
    result = copy.deepcopy(a)
    for key, value in b.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------
# INPUT NORMALISATION (MULTI‑INPUT)
# ---------------------------------------------------------
def build_initial_payload(data: dict) -> dict:
    """
    Supports:
    - Getty ingestion (media_bytes + media_type + getty_metadata)
    - GCS ingestion (file_name + bucket)
    - Direct upload (media_bytes + media_type)
    - URL ingestion (url)
    """

    # -----------------------------------------------------
    # 1. Getty ingestion (raw bytes + metadata)
    # -----------------------------------------------------
    if "media_bytes" in data and data.get("source") == "getty":
        return {
            "asset_id": data.get("asset_id", "getty_asset"),
            "source": "getty",
            "media_type": data["media_type"],
            "media_bytes": data["media_bytes"],
            "getty_metadata": data.get("getty_metadata", {}),
            "asset_type": data["media_type"],  # image or video
        }

    # -----------------------------------------------------
    # 2. Direct upload (raw bytes)
    # -----------------------------------------------------
    if "media_bytes" in data and "media_type" in data:
        return {
            "asset_id": data.get("asset_id", "direct_upload"),
            "source": "upload",
            "media_type": data["media_type"],
            "media_bytes": data["media_bytes"],
            "asset_type": data["media_type"],
        }

    # -----------------------------------------------------
    # 3. GCS-based ingestion (existing behavior)
    # -----------------------------------------------------
    if "file_name" in data and "bucket" in data:
        file_name = data["file_name"]
        bucket = data["bucket"]
        source = data.get("source", "local")
        asset_id = os.path.splitext(os.path.basename(file_name))[0]

        asset_type = detect_asset_type_from_filename(file_name)

        return {
            "asset_id": asset_id,
            "file_name": file_name,
            "bucket": bucket,
            "source": source,
            "asset_type": asset_type,
            "paths": {
                "raw": f"gs://{bucket}/{file_name}"
            }
        }

    # -----------------------------------------------------
    # 4. URL ingestion (optional)
    # -----------------------------------------------------
    if "url" in data:
        return {
            "asset_id": data.get("asset_id", "url_asset"),
            "source": "url",
            "url": data["url"],
            "asset_type": "unknown"
        }

    raise HTTPException(
        status_code=400,
        detail="Invalid input: provide media_bytes OR file_name+bucket OR url"
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
            "enricher": ENRICHER_URL,
            "store": STORE_URL
        }
    }


# ---------------------------------------------------------
# FULL PIPELINE ORCHESTRATION
# ---------------------------------------------------------
@app.post("/run_all")
async def run_all(req: Request):
    data = await req.json()

    # Step 0: Normalize input
    payload = build_initial_payload(data)
    asset_type = payload.get("asset_type")

    merged = payload

    # Step 1: Transcode (video only, and only if using GCS or URL)
    if asset_type == "video" and "media_bytes" not in payload:
        transcode_json = call_service(TRANSCODE_URL, "transcode", merged)
        merged = deep_merge(merged, transcode_json)

    # Step 2: Transcribe (video or audio)
    if asset_type in ["video", "audio"]:
        transcribe_json = call_service(TRANSCRIBE_URL, "transcribe", merged)
        merged = deep_merge(merged, transcribe_json)

    # Step 3: Sample frames (video only, and only if using GCS or URL)
    if asset_type == "video" and "media_bytes" not in payload:
        frames_json = call_service(FRAMES_URL, "sample", merged)
        merged = deep_merge(merged, frames_json)

    # Step 4: Enrichment (always runs)
    enrich_json = call_service(ENRICHER_URL, "enrich", merged)
    merged["analysis"] = enrich_json

    # Step 5: Store metadata (always runs)
    stored_json = call_service(STORE_URL, "store", merged)

    return {
        "pipeline": "complete",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "result": stored_json
    }
