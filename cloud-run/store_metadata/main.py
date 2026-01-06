import os
import json
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from google.cloud import storage, firestore

app = FastAPI()

# --------------------------------------------------------------------
# Environment variables
# --------------------------------------------------------------------
# IMPORTANT:
#   METADATA_BUCKET = "gs://df-films-metadata-euw1"
#
# Folder prefixes (e.g. json/) are handled IN CODE.
# --------------------------------------------------------------------
METADATA_BUCKET = os.getenv("METADATA_BUCKET")  # e.g. gs://df-films-metadata-euw1
FIRESTORE_PROJECT_ID = os.getenv("FIRESTORE_PROJECT_ID")
FIRESTORE_DATABASE_ID = os.getenv("FIRESTORE_DATABASE_ID", "(default)")

if not METADATA_BUCKET:
    raise RuntimeError("METADATA_BUCKET environment variable must be set.")

if not FIRESTORE_PROJECT_ID:
    raise RuntimeError("FIRESTORE_PROJECT_ID environment variable must be set.")

storage_client = storage.Client()
firestore_client = firestore.Client(project=FIRESTORE_PROJECT_ID)


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def normalize_bucket(uri: str) -> str:
    """Convert gs://bucket[/prefix] → bucket."""
    if uri.startswith("gs://"):
        return uri[len("gs://"):].split("/")[0]
    return uri.split("/")[0]


def extract_prefix(uri: str) -> str:
    """Extract optional prefix from gs://bucket/prefix."""
    if uri.startswith("gs://"):
        parts = uri[len("gs://"):].split("/", 1)
    else:
        parts = uri.split("/", 1)
    return parts[1] if len(parts) > 1 else ""


def validate_schema(asset_json: dict) -> None:
    """
    Validates that the incoming JSON is a valid pipeline output.

    Requirements:
      - asset_id must exist
      - paths must exist (pipeline always produces this)
    """
    if "asset_id" not in asset_json:
        raise HTTPException(status_code=400, detail="Missing required key: asset_id")

    if "paths" not in asset_json:
        raise HTTPException(
            status_code=400,
            detail="Invalid schema: expected pipeline output containing 'paths'."
        )


def build_metadata_blob_path(asset_id: str, base_uri: str) -> tuple[str, str]:
    """
    Build the GCS blob path for metadata JSON.

    Example:
      base_uri = "gs://df-films-metadata-euw1"
      → bucket = "df-films-metadata-euw1"
      → blob_path = "metadata/<asset_id>.json"
    """
    bucket = normalize_bucket(base_uri)
    prefix = extract_prefix(base_uri).rstrip("/")

    # Always store metadata under metadata/<asset_id>.json
    metadata_prefix = "metadata"
    if prefix:
        blob_path = f"{prefix}/{metadata_prefix}/{asset_id}.json"
    else:
        blob_path = f"{metadata_prefix}/{asset_id}.json"

    return bucket, blob_path


def write_json_to_gcs(asset_id: str, asset_json: dict, base_uri: str) -> str:
    """Write metadata JSON to GCS."""
    bucket_name, blob_path = build_metadata_blob_path(asset_id, base_uri)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    payload = json.dumps(asset_json, indent=2)
    blob.upload_from_string(payload, content_type="application/json")

    return f"gs://{bucket_name}/{blob_path}"


def write_json_to_firestore(asset_id: str, asset_json: dict) -> dict:
    """Mirror metadata JSON to Firestore."""
    doc_ref = firestore_client.collection("assets").document(asset_id)
    doc_ref.set(asset_json)
    return {"collection": "assets", "doc_id": asset_id}


# --------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "METADATA_BUCKET": METADATA_BUCKET,
        "FIRESTORE_PROJECT_ID": FIRESTORE_PROJECT_ID,
        "FIRESTORE_DATABASE_ID": FIRESTORE_DATABASE_ID,
    }


@app.post("/store")
async def store(req: Request):
    """
    Input: unified JSON from the pipeline containing:
      - asset_id
      - paths (transcoded, audio, transcript, frames, metadata)
      - technical
      - transcript
      - frames
      - qwen

    Behavior:
      - Validates schema
      - Normalizes status/timestamp
      - Writes JSON to GCS under metadata/<asset_id>.json
      - Mirrors JSON to Firestore
      - Returns canonical paths
    """
    asset_json = await req.json()
    validate_schema(asset_json)

    asset_id = asset_json["asset_id"]

    # Normalize status/timestamp
    asset_json["status"] = "complete"
    asset_json["timestamp"] = datetime.utcnow().isoformat() + "Z"

    # Ensure paths.metadata exists and is correct
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
        "timestamp": asset_json["timestamp"],
    }
