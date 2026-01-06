import os
import requests
from fastapi import FastAPI, Request, HTTPException
from datetime import datetime
import google.auth.transport.requests
import google.oauth2.id_token
import copy

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
    """Recursively merge dict b into dict a."""
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
    if "asset_id" in data:
        return call_service(GETTY_URL, "validate", {"asset_id": data["asset_id"]})

    if "file_name" in data and "bucket" in data:
        file_name = data["file_name"]
        bucket = data["bucket"]
        source = data.get("source", "local")
        asset_id = os.path.splitext(os.path.basename(file_name))[0]

        return {
            "asset_id": asset_id,
            "file_name": file_name,
            "bucket": bucket,
            "source": source,
            "paths": {
                "raw": f"gs://{bucket}/{file_name}"
            }
        }

    raise HTTPException(
        status_code=400,
        detail="Invalid input: provide asset_id OR file_name + bucket"
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
    data = await req.json()

    # Step 0: Normalize input
    payload = build_initial_payload(data)

    # Step 1: Transcode
    transcode_json = call_service(TRANSCODE_URL, "transcode", payload)
    merged = deep_merge(payload, transcode_json)

    # Step 2: Transcribe
    transcribe_json = call_service(TRANSCRIBE_URL, "transcribe", merged)
    merged = deep_merge(merged, transcribe_json)

    # Step 3: Sample frames
    frames_json = call_service(FRAMES_URL, "sample", merged)
    merged = deep_merge(merged, frames_json)

    # Step 4: Qwen enrichment (send full merged payload)
    qwen_json = call_service(QWEN_URL, "enrich", merged)
    merged["qwen"] = qwen_json

    # Step 5: Store metadata (send full merged payload)
    stored_json = call_service(STORE_URL, "store", merged)

    return {
        "pipeline": "complete",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "result": stored_json
    }
