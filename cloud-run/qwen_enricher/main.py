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
# Requires service account with:
# - roles/aiplatform.user
# - roles/storage.objectAdmin (or similar)
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
# Load image bytes from GCS
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

        image_bytes = blob.download_as_bytes()
        if not image_bytes:
            raise HTTPException(
                status_code=500,
                detail=f"Image in GCS is empty: gs://{bucket}/{file_name}",
            )

        return image_bytes

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load image from GCS: {e}",
        )


# -------------------------------------------------------------------
# Strip markdown fences from model output
# -------------------------------------------------------------------
def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # remove leading ``` or ```json
        parts = text.split("```", 1)
        if len(parts) > 1:
            text = parts[1]
        # remove trailing ```
        if "```" in text:
            text = text.rsplit("```", 1)[0]
    return text.strip()


# -------------------------------------------------------------------
# Strip leading 'json' token from model output
# -------------------------------------------------------------------
def strip_leading_json_token(text: str) -> str:
    cleaned = text.strip()
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    return cleaned


# -------------------------------------------------------------------
# Schema block
# -------------------------------------------------------------------
SCHEMA_BLOCK = """
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
# Prompt builder (image is primary truth)
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
Do NOT wrap the JSON in markdown fences.
Do NOT include any keys not defined in the schema.
Do NOT prepend language tags like 'json' or explanations.

Schema:
{schema}

Metadata (may be incomplete or wrong):
{metadata}
"""
    return template.format(
        schema=SCHEMA_BLOCK,
        metadata=json.dumps(asset_json, indent=2),
    ).strip()


# -------------------------------------------------------------------
# Extract text from Gemini response
# -------------------------------------------------------------------
def extract_text(response) -> str:
    # Simple path: response.text
    if hasattr(response, "text") and isinstance(response.text, str):
        return response.text

    # Fallback: first candidate, first text part
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
# Gemini multimodal inference with correct inline_data usage
# -------------------------------------------------------------------
def run_gemini(prompt: str, image_bytes: bytes) -> dict:
    try:
        image_part = {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": image_bytes,
            }
        }

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                {"text": prompt},
                image_part,
            ],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calling Gemini model: {e}",
        )

    text = extract_text(response)
    text = strip_markdown_fences(text)
    text = strip_leading_json_token(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail=f"Model returned invalid JSON: {text}",
        )


# -------------------------------------------------------------------
# Write metadata to GCS
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

    image_bytes = load_image_from_gcs(bucket, file_name)
    prompt = build_prompt(asset_json)
    enriched = run_gemini(prompt, image_bytes)

    asset_json["analysis"] = enriched
    asset_json["status"] = "enriched"
    asset_json["timestamp"] = datetime.utcnow().isoformat() + "Z"

    write_metadata_to_gcs(asset_id, asset_json)
    write_metadata_to_firestore(asset_id, asset_json)

    return asset_json
