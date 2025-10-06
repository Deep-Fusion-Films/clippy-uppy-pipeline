from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import logging
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Load Getty taxonomy (assumes local JSON file)
TAXONOMY_PATH = os.getenv("TAXONOMY_PATH", "taxonomy.json")
try:
    with open(TAXONOMY_PATH, "r") as f:
        getty_tags = set(json.load(f))
    logging.info("Getty taxonomy loaded with %d tags", len(getty_tags))
except Exception as e:
    logging.error("Failed to load Getty taxonomy: %s", e)
    getty_tags = set()

# Request schema
class EnrichRequest(BaseModel):
    tags: List[str]

# Response schema
class EnrichResponse(BaseModel):
    filtered_tags: List[str]

# Enrichment endpoint
@app.post("/enrich", response_model=EnrichResponse)
async def enrich_tags(request: EnrichRequest):
    input_tags = [tag.strip().lower() for tag in request.tags]
    matched_tags = [tag for tag in input_tags if tag in getty_tags]

    logging.info("Received %d tags, matched %d Getty tags", len(input_tags), len(matched_tags))
    return EnrichResponse(filtered_tags=matched_tags)

