from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import logging
import os

from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Load embedding model
MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-small-en-v1.5")
model = SentenceTransformer(MODEL_NAME)

# Request schema
class EmbedRequest(BaseModel):
    title: str
    caption: str
    summary: str
    tags: List[str]

# Response schema
class EmbedResponse(BaseModel):
    embedding: List[float]

# Embedding endpoint
@app.post("/embed", response_model=EmbedResponse)
async def generate_embedding(request: EmbedRequest):
    # Combine fields into a single semantic string
    text = f"{request.title}. {request.caption}. {request.summary}. Tags: {', '.join(request.tags)}"

    embedding = model.encode(text, normalize_embeddings=True)
    logging.info("Generated embedding for input: %s", text)

    return EmbedResponse(embedding=embedding.tolist())

