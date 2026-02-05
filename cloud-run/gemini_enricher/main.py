import os
import json
import base64
import subprocess
import tempfile
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from google import genai
from google.cloud import storage, firestore

app = FastAPI()

# -------------------------------------------------------------------
# Gemini client
# -------------------------------------------------------------------
client = genai.Client(
    vertexai=True,
    project="deepfusion-clippyuppy-pipeline",
    location="europe-west1",
)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

storage_client = storage.Client()
firestore_client = firestore.Client()

METADATA_BUCKET = os.getenv("METADATA_BUCKET", "df-films-metadata-euw1")

# Threshold: above this, we use frame sampling
MAX_VIDEO_BYTES = 20 * 1024 * 1024  # 20 MB


# -------------------------------------------------------------------
# Warmup
# -------------------------------------------------------------------
@app.get("/warmup")
def warmup():
    try:
        _ = client is not None
        return {
            "status": "warm",
            "message": "Gemini-enricher is ready",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        raise HTTPException(500, f"Warmup failed: {e}")


# -------------------------------------------------------------------
# Downscale video
# -------------------------------------------------------------------
def downscale_video(media_bytes: bytes, resolution: str = "720") -> bytes:
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(media_bytes)
            input_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            output_path = f.name

        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-vf", f"scale=-2:{resolution}",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "28",
            "-c:a", "aac",
            output_path,
        ]

        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        with open(output_path, "rb") as f:
            return f.read()

    except Exception as e:
        raise HTTPException(500, f"Video downscaling failed: {e}")


# -------------------------------------------------------------------
# Extract frames at 5 FPS
# -------------------------------------------------------------------
def extract_frames(media_bytes: bytes, fps: int = 5) -> list:
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(media_bytes)
            input_path = f.name

        output_dir = tempfile.mkdtemp()
        output_pattern = os.path.join(output_dir, "frame_%04d.jpg")

        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-vf", f"fps={fps}",
            output_pattern,
            "-hide_banner",
            "-loglevel", "error",
        ]

        subprocess.run(cmd, check=True)

        frames = []
        for fname in sorted(os.listdir(output_dir)):
            if fname.endswith(".jpg"):
                with open(os.path.join(output_dir, fname), "rb") as f:
                    frames.append(f.read())

        if not frames:
            raise HTTPException(500, "Frame extraction produced no frames")

        return frames

    except Exception as e:
        raise HTTPException(500, f"Frame extraction failed: {e}")


# -------------------------------------------------------------------
# Load from GCS
# -------------------------------------------------------------------
def load_from_gcs(bucket: str, file_name: str) -> bytes:
    try:
        blob = storage_client.bucket(bucket).blob(file_name)

        if not blob.exists():
            raise HTTPException(404, f"File not found: gs://{bucket}/{file_name}")

        data = blob.download_as_bytes()
        if not data:
            raise HTTPException(500, f"File empty: gs://{bucket}/{file_name}")

        return data

    except Exception as e:
        raise HTTPException(500, f"Failed to load file from GCS: {e}")


# -------------------------------------------------------------------
# Strip helpers
# -------------------------------------------------------------------
def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 1)[1]
        if "```" in text:
            text = text.rsplit("```", 1)[0]
    return text.strip()


def strip_leading_json_token(text: str) -> str:
    cleaned = text.strip()
    if cleaned.lower().startswith("json"):
        return cleaned[4:].strip()
    return cleaned


# -------------------------------------------------------------------
# Schema (unchanged)
# -------------------------------------------------------------------
SCHEMA_BLOCK = """{
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
  "themes": [string],
  "composition": {
    "camera_angle": string|null,
    "focal_length_est": string|null,
    "depth_of_field": string|null,
    "lighting": string|null,
    "color_palette": string|null,
    "contrast_style": string|null,
    "orientation": string|null
  },
  "text_in_image": [string],
  "distinguishing_features": [string],
  "story_use": ["opener"|"bridge"|"chapter_art"|"context"|"climax"|"reveal"],
  "safety": {
    "sensitive": boolean,
    "notes": string|null
  },
  "time_analysis": {
    "era": string|null,
    "decade_estimate": string|null,
    "season_estimate": string|null,
    "time_of_day": string|null,
    "lighting_context": string|null,
    "metadata_date": string|null,
    "visual_date_estimate": string|null,
    "alignment": "match"|"partial_match"|"contradiction",
    "notes": string|null
  },
  "timeline": [
    {
      "timestamp": string|null,
      "description": string,
      "entities_involved": [string],
      "objects_involved": [string],
      "actions": [string],
      "scene_change": boolean
    }
  ]
}"""


# -------------------------------------------------------------------
# Prompt builder (unchanged)
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

TEMPORAL ANALYSIS RULES (VIDEO ONLY):
- Treat the video as a sequence of evolving moments, not a single frame.
- Identify events in chronological order.
- Describe what changes over time: people, objects, actions, lighting, weather, camera movement.
- Note when new elements enter or leave the scene.
- Identify cause-and-effect relationships between events.
- Identify scene boundaries, transitions, and shifts in tone or activity.
- Capture micro-events (gestures, reactions, movements) and macro-events (scene changes, major actions).
- If the video contains multiple shots, describe each shot separately and in order.
- If the video contains continuous action, describe the progression clearly.
- Always anchor descriptions to the timeline: “At the beginning…”, “Midway…”, “Toward the end…”.
- Do not invent events that are not visible.

Your goal is to produce the most detailed, accurate, non‑fictional, non‑redundant analysis possible based solely on what the media shows.

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
# Extract text
# -------------------------------------------------------------------
def extract_text(response) -> str:
    if hasattr(response, "text") and isinstance(response.text, str):
        return response.text

    try:
        parts = response.candidates[0].content.parts
        for p in parts:
            if hasattr(p, "text"):
                return p.text
    except Exception:
        pass

    raise HTTPException(500, "Model response did not contain text output")


# -------------------------------------------------------------------
# Gemini single-media
# -------------------------------------------------------------------
def run_gemini(prompt: str, media_bytes: bytes, media_type: str) -> dict:
    try:
        mime = "video/mp4" if media_type == "video" else "image/jpeg"

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                {"text": prompt},
                {"inline_data": {"mime_type": mime, "data": media_bytes}},
            ],
        )

    except Exception as e:
        raise HTTPException(500, f"Error calling Gemini: {e}")

    text = strip_leading_json_token(strip_markdown_fences(extract_text(response)))

    try:
        return json.loads(text)
    except:
        raise HTTPException(500, f"Invalid JSON from model: {text}")


# -------------------------------------------------------------------
# Gemini multi-image (frames)
# -------------------------------------------------------------------
def run_gemini_multi(prompt: str, frames: list) -> dict:
    try:
        contents = [{"text": prompt}]
        for frame in frames:
            contents.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": frame,
                }
            })

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
        )

    except Exception as e:
        raise HTTPException(500, f"Error calling Gemini (multi): {e}")

    text = strip_leading_json_token(strip_markdown_fences(extract_text(response)))

    try:
        return json.loads(text)
    except:
        raise HTTPException(500, f"Invalid JSON from model: {text}")


# -------------------------------------------------------------------
# Write metadata
# -------------------------------------------------------------------
def write_metadata_to_gcs(asset_id: str, data: dict):
    try:
        blob = storage_client.bucket(METADATA_BUCKET).blob(f"{asset_id}.json")
        blob.upload_from_string(json.dumps(data, indent=2), content_type="application/json")
    except Exception as e:
        raise HTTPException(500, f"Failed to write metadata to GCS: {e}")


def write_metadata_to_firestore(asset_id: str, data: dict):
    try:
        firestore_client.collection("assets").document(asset_id).set(data)
    except Exception as e:
        raise HTTPException(500, f"Failed to write metadata to Firestore: {e}")


# -------------------------------------------------------------------
# Enrich endpoint
# -------------------------------------------------------------------
@app.post("/enrich")
async def enrich(req: Request):
    asset_json = await req.json()

    asset_id = asset_json.get("asset_id")
    media_type = asset_json.get("media_type", "image")

    print("MEDIA_TYPE:", media_type)

    if not asset_id:
        raise HTTPException(400, "asset_id is required")

    # Load media
    if "media_bytes" in asset_json:
        media_bytes = base64.b64decode(asset_json["media_bytes"])
    elif "bucket" in asset_json and "file_name" in asset_json:
        media_bytes = load_from_gcs(asset_json["bucket"], asset_json["file_name"])
    else:
        raise HTTPException(400, "Provide media_bytes OR bucket+file_name")

    # ---------------------------------------------------------
    # VIDEO LOGIC
    # ---------------------------------------------------------
    if media_type == "video":
        original_size = len(media_bytes)
        print("ORIGINAL VIDEO SIZE:", original_size)

        if original_size > MAX_VIDEO_BYTES:
            print("VIDEO TOO LARGE → USING FRAME SAMPLING (5 FPS)")
            frames = extract_frames(media_bytes, fps=5)
            print("EXTRACTED FRAMES:", len(frames))

            prompt = build_prompt(asset_json, media_type)
            print("PROMPT SIZE:", len(prompt))
            print("METADATA SIZE:", len(json.dumps(asset_json)))

            enriched = run_gemini_multi(prompt, frames)

            asset_json["analysis"] = enriched
            asset_json["status"] = "enriched"
            asset_json["timestamp"] = datetime.utcnow().isoformat() + "Z"

            write_metadata_to_gcs(asset_id, asset_json)
            write_metadata_to_firestore(asset_id, asset_json)

            return asset_json

        else:
            print("VIDEO SMALL ENOUGH → DOWNSCALING")
            media_bytes = downscale_video(media_bytes, resolution="720")
            print("DOWNSCALED VIDEO SIZE:", len(media_bytes))

    # ---------------------------------------------------------
    # IMAGE or DOWNSCALED VIDEO
    # ---------------------------------------------------------
    prompt = build_prompt(asset_json, media_type)

    print("MEDIA BYTES:", len(media_bytes))
    print("PROMPT SIZE:", len(prompt))
    print("METADATA SIZE:", len(json.dumps(asset_json)))

    enriched = run_gemini(prompt, media_bytes, media_type)

    asset_json["analysis"] = enriched
    asset_json["status"] = "enriched"
    asset_json["timestamp"] = datetime.utcnow().isoformat() + "Z"

    write_metadata_to_gcs(asset_id, asset_json)
    write_metadata_to_firestore(asset_id, asset_json)

    return asset_json
