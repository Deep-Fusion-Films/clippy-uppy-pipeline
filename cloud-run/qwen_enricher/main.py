import os
import json
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from google import genai
from google.cloud import storage, firestore

app = FastAPI()

# -------------------------------------------------------------------
# Vertex AI Gemini client (Cloud Run service account auth)
# -------------------------------------------------------------------
# IMPORTANT: This uses Cloud Run's service account credentials.
# Make sure the service account has:
# - roles/aiplatform.publisherModelUser
# - roles/storage.objectAdmin (or similar for GCS)
# - roles/datastore.user (for Firestore)
client = genai.Client(
    vertexai=True,
    project="deepfusion-clippyuppy-pipeline",
    location="europe-west1",
)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Cloud Storage + Firestore clients
storage_client = storage.Client()
firestore_client = firestore.Client()

# Bucket where enriched metadata JSON will be stored
METADATA_BUCKET = os.getenv("METADATA_BUCKET", "df-films-metadata-euw1")


# -------------------------------------------------------------------
# Utility: load image bytes from Cloud Storage
# -------------------------------------------------------------------
def load_image_from_gcs(bucket: str, file_name: str) -> bytes:
    try:
        bucket_obj = storage_client.bucket(bucket)
        blob = bucket_obj.blob(file_name)

        if not blob.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Image not found in GCS: gs://{bucket}/{file_name}",
            )

        return blob.download_as_bytes()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load image from GCS: {e}",
        )


# -------------------------------------------------------------------
# Utility: strip Markdown fences from model output
# -------------------------------------------------------------------
def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # Remove leading ``` or ```json
        parts = text.split("```", 1)
        if len(parts) > 1:
            text = parts[1]
        # Remove trailing ```
        if "```" in text:
            text = text.rsplit("```", 1)[0]
    return text.strip()


# -------------------------------------------------------------------
# Schema block (kept consistent with your previous design)
# -------------------------------------------------------------------
SCHEMA_BLOCK = """
Schema (types):
{
  "description_long": string,
  "entities": {
    "people": [
      {
        "name": string|null,
        "role": string|null,
        "clothing": string[],
        "age_range": string|null,
        "facial_expression": string|null,
        "pose": string|null
      }
    ],
    "places": [
      {
        "name": string|null,
        "sublocation": string|null,
        "indoor_outdoor": "indoor"|"outdoor"|null
      }
    ],
    "orgs": [
      {
        "name": string,
        "evidence": string|null
      }
    ]
  },
  "time": {
    "year": number|null,
    "month": number|null,
    "day": number|null,
    "confidence": number,
    "hint_text": string|null
  },
  "objects": [
    { "label": string, "salience": number }
  ],
  "activities": [
    { "label": string, "confidence": number, "who": string|null }
  ],
  "themes": [ string ],
  "composition": {
    "camera_angle": string|null,
    "focal_length_est": string|null,
    "depth_of_field": string|null,
    "lighting": string|null,
    "color_palette": string|null,
    "contrast_style": string|null,
    "orientation": string|null
  },
  "text_in_image": [ string ],
  "distinguishing_features": [ string ],
  "story_use": [
    "opener" | "bridge" | "chapter_art" | "context" | "climax" | "reveal"
  ],
  "safety": {
    "sensitive": boolean,
    "notes": string|null
  }
}
"""


# -------------------------------------------------------------------
# Prompt builder – strongly prioritises the actual image
# -------------------------------------------------------------------
def build_prompt(asset_json: dict) -> str:
    template = """
You are an image analyst for a factual documentary.

You are given:
- A REAL image (primary source of truth)
- Optional metadata (secondary, may be noisy or wrong)

Your job:
- Describe and analyse ONLY what is actually visible in the image.
- Use metadata ONLY as a weak hint, and IGNORE it if it conflicts with the image.
- Never invent people, locations, or objects that are not clearly visible.

Output STRICT JSON only, matching the schema below.
Focus on composition, lighting, micro-differences, and discriminative details that would
distinguish this image from near-duplicates.

{schema}

Rules:
- Trust the IMAGE over metadata. If metadata contradicts the image, follow the image.
- If uncertain, use null or [] and lower confidence.
- Do NOT wrap the JSON in markdown fences.
- Do NOT include any extra fields beyond the schema.
- The output must be a single JSON object.

### Provided Metadata (may be incomplete or partially wrong)
{metadata}
"""
    return template.format(
        schema=SCHEMA_BLOCK,
        metadata=json.dumps(asset_json, indent=2),
    ).strip()


# -------------------------------------------------------------------
# Extract text safely from Gemini response
# -------------------------------------------------------------------
def extract_text(response) -> str:
    # New genai client exposes .text for simple responses
    if hasattr(response, "text") and isinstance(response.text, str):
        return response.text

    # Fallback for more structured responses
    try:
        candidates = getattr(response, "candidates", None)
        if candidates:
            parts = candidates[0].content.parts
            for part in parts:
                if hasattr(part, "text") and isinstance(part.text, str):
                    return part.text
    except Exception:
        pass

    raise HTTPException(
        status_code=500,
        detail="Model response did not contain text output.",
    )


# -------------------------------------------------------------------
# Gemini multimodal inference (image + prompt)
# -------------------------------------------------------------------
def run_gemini(prompt: str, image_bytes: bytes) -> dict:
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                {"text": prompt},
                {"image": image_bytes},
            ],
        )
    except Exception as e:
        # Surface the error so you see it clearly in FastAPI
        raise HTTPException(
            status_code=500,
            detail=f"Error calling Gemini model: {e}",
        )

    text = extract_text(response)
    text = strip_markdown_fences(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail=f"Model returned invalid JSON: {text}",
        )


# -------------------------------------------------------------------
# Write metadata to Cloud Storage
# -------------------------------------------------------------------
def write_metadata_to_gcs(asset_id: str, data: dict):
    try:
        bucket = storage_client.bucket(METADATA_BUCKET)
        blob = bucket.blob(f"{asset_id}.json")
        blob.upload_from_string(
            json.dumps(data, indent=2),
            content_type="application/json",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write metadata to GCS: {e}",
        )


# -------------------------------------------------------------------
# Write metadata to Firestore
# -------------------------------------------------------------------
def write_metadata_to_firestore(asset_id: str, data: dict):
    try:
        doc_ref = firestore_client.collection("assets").document(asset_id)
        doc_ref.set(data)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write metadata to Firestore: {e}",
        )


# -------------------------------------------------------------------
# Enrichment endpoint
# -------------------------------------------------------------------
@app.post("/enrich")
async def enrich(req: Request):
    asset_json = await req.json()

    asset_id = asset_json.get("asset_id")
    bucket = asset_json.get("bucket")
    file_name = asset_json.get("file_name")

    if not asset_id or not bucket or not file_name:
        raise HTTPException(
            status_code=400,
            detail="asset_id, bucket, and file_name are required",
        )

    # Load the real image
    image_bytes = load_image_from_gcs(bucket, file_name)

    # Build prompt with metadata (image is primary truth)
    prompt = build_prompt(asset_json)

    # Run Gemini multimodal
    enriched = run_gemini(prompt, image_bytes)

    # Attach analysis + pipeline status
    asset_json["analysis"] = enriched
    asset_json["status"] = "enriched"
    asset_json["timestamp"] = datetime.utcnow().isoformat() + "Z"

    # Persist metadata
    write_metadata_to_gcs(asset_id, asset_json)
    write_metadata_to_firestore(asset_id, asset_json)

    return asset_json
