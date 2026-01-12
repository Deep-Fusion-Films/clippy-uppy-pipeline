import os
import uuid
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage

app = FastAPI()

GETTY_API_KEY = os.environ["GETTY_API_KEY"]
GETTY_API_SECRET = os.environ["GETTY_API_SECRET"]
START_PIPELINE_URL = os.environ["START_PIPELINE_URL"]
TEMP_BUCKET = "clippyuppy-temp-ingestor"


# -----------------------------
# Request Models
# -----------------------------
class ProcessRequest(BaseModel):
    query: str | None = None
    asset_id: str | None = None
    media_type: str | None = "image"


# -----------------------------
# Getty API Helpers
# -----------------------------
def get_getty_access_token():
    url = "https://api.gettyimages.com/oauth2/token"
    data = {
        "client_id": GETTY_API_KEY,
        "client_secret": GETTY_API_SECRET,
        "grant_type": "client_credentials"
    }
    r = requests.post(url, data=data)
    r.raise_for_status()
    return r.json()["access_token"]


def get_getty_search_results(query: str):
    token = get_getty_access_token()
    url = "https://api.gettyimages.com/v3/search/images"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"phrase": query, "page_size": 1}

    r = requests.get(url, headers=headers, params=params)
    r.raise_for_status()
    results = r.json().get("images", [])

    if not results:
        raise HTTPException(status_code=404, detail="No Getty results found")

    return results[0]  # first result


def get_getty_download_url(asset_id: str):
    token = get_getty_access_token()
    url = f"https://api.gettyimages.com/v3/downloads/{asset_id}"
    headers = {"Authorization": f"Bearer {token}"}

    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        raise HTTPException(
            status_code=400,
            detail=f"Getty download failed: {r.text}"
        )

    data = r.json()
    download_url = data.get("uri")

    if not download_url:
        raise HTTPException(
            status_code=400,
            detail=f"Getty did not return a download URL: {data}"
        )

    return download_url


# -----------------------------
# GCS Streaming Upload
# -----------------------------
def upload_getty_media_to_gcs(download_url: str) -> str:
    object_name = f"getty/{uuid.uuid4()}"
    storage_client = storage.Client()
    bucket = storage_client.bucket(TEMP_BUCKET)
    blob = bucket.blob(object_name)

    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        with blob.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    return f"gs://{TEMP_BUCKET}/{object_name}"


# -----------------------------
# Main Processing Endpoint
# -----------------------------
@app.post("/process")
def process_asset(req: ProcessRequest):
    # 1. Resolve asset ID
    if req.asset_id:
        asset_id = req.asset_id
        metadata = {"asset_id": asset_id}
    elif req.query:
        result = get_getty_search_results(req.query)
        asset_id = result["id"]
        metadata = result
    else:
        raise HTTPException(status_code=400, detail="Provide query or asset_id")

    # 2. Get Getty download URL
    download_url = get_getty_download_url(asset_id)

    # 3. Stream into GCS
    gcs_url = upload_getty_media_to_gcs(download_url)

    # 4. Send tiny payload to start_pipeline
    payload = {
        "media_url": gcs_url,
        "media_type": req.media_type,
        "getty_metadata": metadata
    }

    r = requests.post(START_PIPELINE_URL, json=payload)
    if r.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"start_pipeline returned error: {r.text}"
        )

    return {"status": "ok", "pipeline_response": r.json()}


# -----------------------------
# Health Check
# -----------------------------
@app.get("/health")
def health():
    return {"status": "healthy"}
