from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import logging
import os

from google.cloud import firestore

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Initialize Firestore client
db = firestore.Client()

# Request schema
class MetadataRequest(BaseModel):
    asset_id: str
    title: str
    caption: str
    summary: str
    tags: List[str]
    licensing_flags: List[str]
    embedding: List[float]

# Response schema
class MetadataResponse(BaseModel):
    status: str
    document_path: str

# Metadata storage endpoint
@app.post("/store", response_model=MetadataResponse)
async def store_metadata(request: MetadataRequest):
    doc_ref = db.collection("media_metadata").document(request.asset_id)

    metadata = {
        "title": request.title,
        "caption": request.caption,
        "summary": request.summary,
        "tags": request.tags,
        "licensing_flags": request.licensing_flags,
        "embedding": request.embedding
    }

    try:
        doc_ref.set(metadata)
        logging.info("Stored metadata for asset_id: %s", request.asset_id)
        return MetadataResponse(status="success", document_path=doc_ref.path)
    except Exception as e:
        logging.error("Failed to store metadata: %s", e)
        return MetadataResponse(status="error", document_path="")

