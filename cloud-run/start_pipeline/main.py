import os
import requests
from fastapi import FastAPI, Request, HTTPException
from datetime import datetime

app = FastAPI()

# Environment variables: service URLs
GETTY_URL = os.getenv("GETTY_URL")
TRANSCODE_URL = os.getenv("TRANSCODE_URL")
TRANSCRIBE_URL = os.getenv("TRANSCRIBE_URL")
FRAMES_URL = os.getenv("FRAMES_URL")
QWEN_URL = os.getenv("QWEN_URL")
STORE_URL = os.getenv("STORE_URL")

def call_service(url: str, endpoint: str, payload: dict) -> dict:
    if not url:
        raise HTTPException(status_code=500, detail=f"Missing service URL for {endpoint}")
    resp = requests.post(f"{url}/{endpoint}", json=payload)
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
    Input: {"asset_id":"getty-123456"}
    Orchestrates the full pipeline:
      1. Getty-enricher
      2. Transcode
      3. Transcribe
      4. Sample-frames
      5. Qwen-enricher
      6. Store-metadata
    Returns: unified JSON with all blocks
    """
    data = await req.json()
    asset_id = data.get("asset_id")
    if not asset_id:
        raise HTTPException(status_code=400, detail="asset_id required")

    # Step 1: Getty
    getty_json = call_service(GETTY_URL, "validate", {"asset_id": asset_id})

    # Step 2: Transcode
    transcode_json = call_service(TRANSCODE_URL, "transcode", getty_json)

    # Step 3: Transcribe
    transcribe_json = call_service(TRANSCRIBE_URL, "transcribe", transcode_json)

    # Step 4: Sample frames
    frames_json = call_service(FRAMES_URL, "sample", transcribe_json)

    # Step 5: Qwen enrichment
    qwen_json = call_service(QWEN_URL, "enrich", frames_json)

    # Step 6: Store metadata
    stored_json = call_service(STORE_URL, "store", qwen_json)

    # Final unified JSON
    return {
        "asset_id": asset_id,
        "pipeline": "complete",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "result": stored_json
    }
