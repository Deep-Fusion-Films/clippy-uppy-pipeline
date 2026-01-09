import os
import base64
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

GETTY_API_KEY = os.getenv("GETTY_API_KEY")
GETTY_API_SECRET = os.getenv("GETTY_API_SECRET")

START_PIPELINE_URL = os.getenv(
    "START_PIPELINE_URL",
    "https://start-pipeline-863025961248.europe-west1.run.app/run_all"
)

# -----------------------------
# Request models
# -----------------------------
class GettySearchRequest(BaseModel):
    query: str
    page: int = 1
    page_size: int = 1


class GettyAssetRequest(BaseModel):
    asset_id: str


# -----------------------------
# Helper: Getty API auth header
# -----------------------------
def get_getty_headers():
    return {
        "Api-Key": GETTY_API_KEY,
        "Accept": "application/json"
    }


# -----------------------------
# Helper: Download media bytes
# -----------------------------
def download_media(url: str) -> bytes:
    resp = requests.get(url)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download media from Getty: {resp.text}"
        )
    return resp.content


# -----------------------------
# Helper: Forward to pipeline
# -----------------------------
def forward_to_pipeline(media_bytes: bytes, media_type: str, metadata: dict):
    encoded = base64.b64encode(media_bytes).decode("utf-8")

    payload = {
        "source": "getty",
        "media_type": media_type,
        "media_bytes": encoded,
        "getty_metadata": metadata
    }

    resp = requests.post(START_PIPELINE_URL, json=payload)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {resp.text}"
        )

    return resp.json()


# -----------------------------
# Endpoint: Search Getty
# -----------------------------
@app.post("/search")
def search_getty(req: GettySearchRequest):
    url = (
        f"https://api.gettyimages.com/v3/search/images?"
        f"phrase={req.query}&page={req.page}&page_size={req.page_size}"
    )

    resp = requests.get(url, headers=get_getty_headers())
    if resp.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"Getty search failed: {resp.text}"
        )

    data = resp.json()
    if not data.get("images"):
        raise HTTPException(status_code=404, detail="No Getty results found")

    return data["images"]


# -----------------------------
# Endpoint: Process Getty asset
# -----------------------------
@app.post("/process")
def process_getty_asset(req: GettyAssetRequest):
    # 1. Fetch metadata for the asset
    meta_url = f"https://api.gettyimages.com/v3/images/{req.asset_id}"
    meta_resp = requests.get(meta_url, headers=get_getty_headers())

    if meta_resp.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch Getty metadata: {meta_resp.text}"
        )

    meta_json = meta_resp.json()
    if "images" not in meta_json or len(meta_json["images"]) == 0:
        raise HTTPException(status_code=404, detail="Getty asset not found")

    asset = meta_json["images"][0]

    # 2. Determine media type
    media_type = "image"
    if asset.get("media_type") == "video":
        media_type = "video"

    # 3. Get download URL
    sizes = asset.get("display_sizes", [])
    if not sizes:
        raise HTTPException(status_code=500, detail="No download URL available")

    download_url = sizes[0]["uri"]

    # 4. Download media bytes
    media_bytes = download_media(download_url)

    # 5. Forward to your pipeline
    enriched = forward_to_pipeline(
        media_bytes=media_bytes,
        media_type=media_type,
        metadata=asset
    )

    return {
        "asset_id": req.asset_id,
        "status": "processed",
        "metadata": enriched
    }
