from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import os
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Request schema
class GettyRequest(BaseModel):
    asset_id: str                # Getty video/image ID
    fields: Optional[List[str]]  # Which metadata fields to fetch (e.g. title, caption, keywords)

# Response schema
class GettyResponse(BaseModel):
    status: str
    metadata: Optional[dict]

# Health check endpoint (required for Cloud Run)
@app.get("/")
def health():
    return {"status": "healthy"}

# API endpoint
@app.post("/validate", response_model=GettyResponse)
async def validate_metadata(request: GettyRequest):
    client_id = os.getenv("GETTY_CLIENT_ID")
    client_secret = os.getenv("GETTY_CLIENT_SECRET")

    if not client_id or not client_secret:
        logging.error("Getty API credentials not set")
        return GettyResponse(status="error", metadata=None)

    try:
        # Example Getty API call (Search/Metadata endpoint)
        url = f"https://api.gettyimages.com/v3/assets/{request.asset_id}"
        headers = {
            "Api-Key": client_id,
            "Authorization": f"Bearer {client_secret}"
        }

        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        # Filter metadata fields if requested
        if request.fields:
            filtered = {field: data.get(field) for field in request.fields}
            return GettyResponse(status="success", metadata=filtered)

        return GettyResponse(status="success", metadata=data)

    except Exception as e:
        logging.error("Getty API call failed: %s", e)
        return GettyResponse(status="error", metadata=None)
