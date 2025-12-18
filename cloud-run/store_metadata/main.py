import os
import json
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from google.cloud import storage, firestore

app = FastAPI()

# Environment variables
METADATA_BUCKET = os.getenv("METADATA_BUCKET")  # e.g. gs://df-films-metadata-euw1/json
FIRESTORE_PROJECT_ID = os.getenv("FIRESTORE_PROJECT_ID")
FIRESTORE_DATABASE_ID = os.getenv("FIRESTORE_DATABASE_ID", "(default)")

storage_client = storage.Client()
firestore_client = firestore.Client(project=FIRESTORE_PROJECT_ID)

def bucket_name_from_uri(uri: str) -> str:
    return uri.replace("gs://", "").split("/")[0]

def prefix_from_uri(uri: str) -> str:
    parts = uri.replace("gs://", "").split("/", 1)
    return parts[1] if len(parts) > 1 else ""

def validate_schema(asset_json: dict) -> None:
    # Require asset_id always
    if "asset_id" not in asset_json:
        raise HTTPException(status_code=400, detail="Missing required key: asset_id")

    # Require either paths.raw (Getty) OR file_name+bucket (local)
    if "paths" in asset_json and "raw" in asset_json["paths"]:
        return
    elif "file_name" in asset_json and "bucket" in asset_json:
        return
    else:
        raise HTTPException(
            status_code=400,
            detail="Missing required input: provide paths.raw (Getty) OR file_name+bucket (local)"
        )

def build_metadata_blob_path(asset_id: str, base_uri: str) -> str:
    bucket = bucket_name_from_uri(base_uri)
    prefix = prefix_from_uri(base_uri).rstrip("/")
    blob_path = f"{prefix}/{asset_id}.json" if prefix else f"{asset_id}.json"
    return bucket, blob_path

def write_json_to_gcs(asset_id: str, asset_json: dict, base_uri: str) -> str:
    bucket_name, blob_path = build_metadata_blob_path(asset_id, base_uri)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    payload = json.dumps(asset_json, indent=2)
    blob.upload_from_string(payload, content_type="application/json")
    return f"gs://{bucket_name}/{blob_path}"

def write_json_to_firestore(asset_id: str, asset_json: dict) -> dict:
    doc_ref = firestore_client.collection("assets").document(asset_id)
    doc_ref.set(asset_json)
    return {"collection": "assets", "doc_id": asset_id}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "METADATA_BUCKET": METADATA_BUCKET,
        "FIRESTORE_PROJECT_ID": FIRESTORE_PROJECT_ID,
        "FIRESTORE_DATABASE_ID": FIRESTORE_DATABASE_ID
    }

@app.post("/store")
async def store(req: Request):
    """
    Input: unified JSON (from pipeline) containing technical, transcript, frames, qwen, paths.
    Behavior:
      - Validates schema (Getty or local mode)
      - Ensures metadata path in 'paths.metadata'
      - Writes JSON to GCS and mirrors to Firestore
      - Returns confirmation with canonical paths
    """
    asset_json = await req.json()
    validate_schema(asset_json)

    asset_id = asset_json["asset_id"]

    # Normalize status/timestamp
    asset_json["status"] = "complete"
    asset_json["timestamp"] = datetime.utcnow().isoformat() + "Z"

    # Ensure paths.metadata is consistent with METADATA_BUCKET
    metadata_uri = write_json_to_gcs(asset_id, asset_json, METADATA_BUCKET)
    asset_json.setdefault("paths", {})
    asset_json["paths"]["metadata"] = metadata_uri

    # Mirror to Firestore
    fs_info = write_json_to_firestore(asset_id, asset_json)

    return {
        "asset_id": asset_id,
        "paths": asset_json["paths"],
        "firestore": fs_info,
        "status": "stored",
        "timestamp": asset_json["timestamp"]
    }
