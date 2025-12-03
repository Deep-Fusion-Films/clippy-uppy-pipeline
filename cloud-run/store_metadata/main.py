from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict
import os
import logging
from google.cloud import firestore, storage

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Initialize Firestore client
db = firestore.Client()

# Optionally initialize Cloud Storage client
storage_client = storage.Client()
bucket_name = os.getenv("METADATA_BUCKET")

# Request schema
class MetadataRequest(BaseModel):
    asset_id: str                 # Unique ID for the video/image
    metadata: Dict[str, str]      # Arbitrary metadata fields
    store_in_bucket: Optional[bool] = False  # Whether to also persist JSON in GCS

# Response schema
class MetadataResponse(BaseModel):
    status: str
    firestore_doc: Optional[str]
    gcs_blob: Optional[str]

# Health check endpoint (required for Cloud Run)
@app.get("/")
def health():
    return {"status": "healthy"}

# API endpoint
@app.post("/store", response_model=MetadataResponse)
async def store_metadata(request: MetadataRequest):
    try:
        # Store in Firestore
        doc_ref = db.collection("assets").document(request.asset_id)
        doc_ref.set(request.metadata)
        logging.info("Stored metadata for asset %s in Firestore", request.asset_id)

        gcs_blob = None
        if request.store_in_bucket and bucket_name:
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(f"{request.asset_id}.json")
            blob.upload_from_string(str(request.metadata))
            gcs_blob = blob.name
            logging.info("Stored metadata for asset %s in GCS bucket %s", request.asset_id, bucket_name)

        return MetadataResponse(
            status="success",
            firestore_doc=request.asset_id,
            gcs_blob=gcs_blob
        )
    except Exception as e:
        logging.error("Metadata storage failed: %s", e)
        return MetadataResponse(status="error", firestore_doc=None, gcs_blob=None)

