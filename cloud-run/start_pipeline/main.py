import os
import copy
import requests
from datetime import datetime
from typing import Tuple
from fastapi import FastAPI, Request, HTTPException
import google.auth.transport.requests
import google.oauth2.id_token

app = FastAPI()

# Downstream service URLs
GETTY_URL = os.getenv("GETTY_URL")
TRANSCODE_URL = os.getenv("TRANSCODE_URL")
TRANSCRIBE_URL = os.getenv("TRANSCRIBE_URL")
FRAMES_URL = os.getenv("FRAMES_URL")
ENRICHER_URL = os.getenv("ENRICHER_URL")
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
# GS URL PARSING (Python 3.9 compatible)
# ---------------------------------------------------------
def parse_gs_url(gs_url: str) -> Tuple[str, str]:
    """
    Parse gs://bucket/object into (bucket, object_name)
    """
    if not gs_url.startswith("gs://"):
        raise HTTPException(status_code=400, detail=f"Invalid GCS URL: {gs_url}")

    without_scheme = gs_url[len("gs://"):]
    parts = without_scheme.split("/", 1)
    if len(parts) != 2:
        raise HTTPException(status_code=400, detail=f"Invalid GCS URL: {gs_url}")

    bucket, object_name = parts
    if not bucket or not object_name:
        raise HTTPException(status_code=400, detail=f"Invalid GCS URL: {gs_url}")

    return bucket, object_name


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
# INPUT NORMALISATION
# ---------------------------------------------------------
def build_initial_payload(data: dict) -> dict:
    """
    Supports:
    1. Getty/GCS ingestion (media_url = gs://...)
    2. Getty ingestion (media_bytes + metadata)
    3. Direct upload (media_bytes)
    4. GCS ingestion (file_name + bucket)
    5. URL ingestion
    """

    # -----------------------------------------------------
    # 1. Getty / GCS ingestion via media_url
    # -----------------------------------------------------
    if "media_url" in data:
        media_url = data["media_url"]
        media_type = data.get("media_type", "unknown")
        getty_metadata = data.get("getty_metadata", {})
        source = data.get("source", "getty")
        asset_id = data.get("asset_id", "getty_asset")

        bucket, object_name = parse_gs_url(media_url)
        file_name = object_name

        asset_type = (
            media_type
            if media_type in ["image", "video", "audio"]
            else detect_asset_type_from_filename(file_name)
        )

        # Always trust media_url as the raw path
        raw_path = media_url

        payload = {
            "asset_id": asset_id,
            "source": source,
            "media_type": media_type,
            "asset_type": asset_type,
            "bucket": bucket,
            "file_name": file_name,
            "paths": {
                "raw": raw_path
            }
        }

        if getty_metadata:
            payload["getty_metadata"] = getty_metadata

        return payload

    # -----------------------------------------------------
    # 2. Getty ingestion (raw bytes + metadata)
    # -----------------------------------------------------
    if "media_bytes" in data and data.get("source") == "getty":
        return {
            "asset_id": data.get("asset_id", "getty_asset"),
            "source": "getty",
            "media_type": data["media_type"],
            "media_bytes": data["media_bytes"],
            "getty_metadata": data.get("getty_metadata", {}),
            "asset_type": data["media_type"],
        }

    # -----------------------------------------------------
    # 3. Direct upload (raw bytes)
    # -----------------------------------------------------
    if "media_bytes" in data and "media_type" in data:
        return {
            "asset_id": data.get("asset_id", "direct_upload"),
            "source": data.get("source", "upload"),
            "media_type": data["media_type"],
            "media_bytes": data["media_bytes"],
            "asset_type": data["media_type"],
        }

    # -----------------------------------------------------
    # 4. GCS ingestion
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
    # 5. URL ingestion
    # -----------------------------------------------------
    if "url" in data:
        return {
            "asset_id": data.get("asset_id", "url_asset"),
            "source": data.get("source", "url"),
            "url": data["url"],
            "asset_type": "unknown"
        }

    raise HTTPException(
        status_code=400,
        detail="Invalid input: provide media_url OR media_bytes OR file_name+bucket OR url"
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

    payload = build_initial_payload(data)
    asset_type = payload.get("asset_type")

    merged = payload
    audio_status = None  # track audio status from transcoder for video assets

    # Step 1: Transcode (video only, non-byte Getty/GCS/local)
    if asset_type == "video" and "media_bytes" not in payload:
        transcode_json = call_service(TRANSCODE_URL, "transcode", merged)
        merged = deep_merge(merged, transcode_json)

        # Try to read audio status from transcoder response
        audio_status = (
            merged.get("status", {})
            .get("audio")
        )

    # Step 2: Transcribe
    # - For pure audio assets: always attempt transcription
    # - For video assets: only if audio was actually extracted
    should_transcribe = False

    if asset_type == "audio":
        should_transcribe = True
    elif asset_type == "video":
        # Only transcribe if transcoder reported usable audio
        if audio_status == "extracted":
            should_transcribe = True

    if should_transcribe:
        transcribe_json = call_service(TRANSCRIBE_URL, "transcribe", merged)
        merged = deep_merge(merged, transcribe_json)

    # Step 3: Sample frames (video only, non-byte Getty/GCS/local)
    if asset_type == "video" and "media_bytes" not in payload:
        frames_json = call_service(FRAMES_URL, "sample", merged)
        merged = deep_merge(merged, frames_json)

    # Step 4: Enrich
    enrich_json = call_service(ENRICHER_URL, "enrich", merged)
    merged["analysis"] = enrich_json

    # Step 5: Store
    stored_json = call_service(STORE_URL, "store", merged)

    return {
        "pipeline": "complete",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "result": stored_json
    }
