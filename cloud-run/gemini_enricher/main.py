import os
import json
import base64
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from google import genai
from google.cloud import storage, firestore

app = FastAPI()

# -------------------------------------------------------------------
# Vertex AI Gemini client (Cloud Run service account auth)
# -------------------------------------------------------------------
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
# Load bytes from GCS
# -------------------------------------------------------------------
def load_from_gcs(bucket: str, file_name: str) -> bytes:
    try:
        bucket_obj = storage_client.bucket(bucket)
        blob = bucket_obj.blob(file_name)

        if not blob.exists():
            raise HTTPException(
                status_code=404,
                detail=f"File not found in GCS: gs://{bucket}/{file_name}",
            )

        file_bytes = blob.download_as_bytes()
        if not file_bytes:
            raise HTTPException(
                status_code=500,
                detail=f"File in GCS is empty: gs://{bucket}/{file_name}",
            )

        return file_bytes

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load file from GCS: {e}",
        )


# -------------------------------------------------------------------
# Strip markdown fences
# -------------------------------------------------------------------
def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```", 1)
        if len(parts) > 1:
            text = parts[1]
        if "```" in text:
            text = text.rsplit("```", 1)[0]
    return text.strip()


# -------------------------------------------------------------------
# Strip leading 'json'
# -------------------------------------------------------------------
def strip_leading_json_token(text: str) -> str:
    cleaned = text.strip()
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    return cleaned


# -------------------------------------------------------------------
# CLEANED SCHEMA BLOCK (Option B)
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
        "indoor_outdoor": "indoor" | "outdoor" | null
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
# Prompt builder
# -------------------------------------------------------------------
def build_prompt(asset_json: dict, media_type: str) -> str:
    media_line = (
        "You are given a REAL video."
        if media_type == "video"
        else "You are given a REAL image."
    )

    template = f"""
You are a forensic visual analyst.

{media_line}
Your task is to extract ONLY information that is directly visible in the media.
Metadata may be incomplete, noisy, or partially incorrect. Treat it as optional context, not fact.

STRICT RULES:
- Describe ONLY what is visually present.
- Do NOT invent or assume anything that cannot be clearly confirmed.
- Do NOT use metadata to add details that are not visible.
- Do NOT repeat the same information across multiple fields.
- Be specific, concrete, and observational.
- If something is unclear or partially visible, state the uncertainty.
- If a field has no visible evidence, return null or an empty list.
- Avoid generic statements; focus on precise, observable attributes.
- Output MUST be valid JSON following the schema exactly.
- No markdown, no commentary, no extra keys.

TIME SENSITIVITY RULES:
- Use Getty metadata (e.g., date_created, era, collection, caption) as contextual hints.
- Cross-check metadata against visible evidence such as clothing, vehicles, architecture, technology, hairstyles, film grain, color grading, and lighting style.
- If metadata suggests a specific year or decade, treat it as a hypothesis and verify visually.
- If the visual evidence contradicts metadata, note the contradiction explicitly.
- Estimate time of day, season, and era based on shadows, foliage, weather, clothing, and lighting.
- If the media appears archival, identify the approximate decade based on film texture, aspect ratio, and color profile.
- If the media appears modern, identify the approximate year range based on camera quality, resolution, and color science.
- Always return a confidence score.

Your goal is to produce the most detailed, accurate, non‑fictional, non‑redundant analysis possible based solely on what the media shows. List any recognisable people or landmarks, species, object, sub species. Be very specific and be descriptive about everything thats seen. Fully in depth observation 

Schema:
{{schema}}

Metadata (weak hints only):
{{metadata}}
"""

    return template.format(
        schema=SCHEMA_BLOCK,
        metadata=json.dumps(asset_json, indent=2),
    ).strip()



# -------------------------------------------------------------------
# Extract text from Gemini response
# -------------------------------------------------------------------
def extract_text(response) -> str:
    if hasattr(response, "text") and isinstance(response.text, str):
        return response.text

    try:
        candidates = getattr(response, "candidates", None)
        if candidates:
            parts = candidates[0].content.parts
            for part in parts:
                if hasattr(part, "text"):
                    return part.text
    except Exception:
        pass

    raise HTTPException(
        status_code=500,
        detail="Model response did not contain text output.",
    )


# -------------------------------------------------------------------
# Gemini inference (image or video)
# -------------------------------------------------------------------
def run_gemini(prompt: str, media_bytes: bytes, media_type: str) -> dict:
    try:
        mime = "video/mp4" if media_type == "video" else "image/jpeg"

        media_part = {
            "inline_data": {
                "mime_type": mime,
                "data": media_bytes,
            }
        }

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                {"text": prompt},
                media_part,
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
# Enrichment endpoint (multi‑input aware)
# -------------------------------------------------------------------
@app.post("/enrich")
async def enrich(req: Request):
    asset_json = await req.json()

    asset_id = asset_json.get("asset_id")
    media_type = asset_json.get("media_type", "image")

    if not asset_id:
        raise HTTPException(400, "asset_id is required")

    # ---------------------------------------------------------
    # 1. Getty or direct upload (media_bytes)
    # ---------------------------------------------------------
    if "media_bytes" in asset_json:
        try:
            media_bytes = base64.b64decode(asset_json["media_bytes"])
        except Exception:
            raise HTTPException(400, "Invalid base64 media_bytes")

    # ---------------------------------------------------------
    # 2. GCS mode (existing)
    # ---------------------------------------------------------
    elif "bucket" in asset_json and "file_name" in asset_json:
        media_bytes = load_from_gcs(asset_json["bucket"], asset_json["file_name"])

    else:
        raise HTTPException(
            400,
            "Invalid input: provide media_bytes OR bucket+file_name"
        )

    # ---------------------------------------------------------
    # Build prompt + run Gemini
    # ---------------------------------------------------------
    prompt = build_prompt(asset_json, media_type)
    enriched = run_gemini(prompt, media_bytes, media_type)

    # ---------------------------------------------------------
    # Merge + store metadata
    # ---------------------------------------------------------
    asset_json["analysis"] = enriched
    asset_json["status"] = "enriched"
    asset_json["timestamp"] = datetime.utcnow().isoformat() + "Z"

    write_metadata_to_gcs(asset_id, asset_json)
    write_metadata_to_firestore(asset_id, asset_json)

    return asset_json
