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
# Schema (upgraded with explicit AI‑artifact fields)
# -------------------------------------------------------------------
SCHEMA_BLOCK = """{
  "brief_summary": "string",
  "verbose_summary": "string",

  "people": {
    "present": "boolean",
    "count": "number | null",
    "details": [
      {
        "age": "string | null",
        "gen": "string | null",
        "role": "string | null",
        "act": ["string"],
        "pos": "string | null",
        "clo": ["string"],
        "vis": "string | null"
      }
    ]
  },

  "animals": [
    {
      "type": "string",
      "cnt": "number | null",
      "beh": ["string"],
      "col": "string | null",
      "pos": "string | null",
      "int": ["string"]
    }
  ],

  "objects": [
    {
      "lbl": "string",
      "cnt": "number | null",
      "sal": "number | null",
      "pos": "string | null",
      "use": "string | null"
    }
  ],

  "brand_ip": {
    "logos": ["string"],
    "other": ["string"],
    "ctx": "string | null"
  },

  "celebrities": {
    "detected": ["string"],
    "ctx": "string | null"
  },

  "camera": {
    "mov": "string | null",
    "shake": "string | null",
    "pan": "boolean | null",
    "tilt": "boolean | null",
    "zoom": "boolean | null",
    "framing": "string | null",
    "exp": "string | null",
    "focus": "string | null",
    "tempo": "string | null"
  },

  "environment": {
    "tod": "string | null",
    "loc": "string | null",
    "lit": "string | null",
    "wth": "string | null",
    "surf": "string | null",
    "bg": "string | null",
    "depth": "string | null"
  },

  "audio": {
    "speech": "string | null",
    "lang": "string | null",
    "events": ["string"],
    "noise": ["string"],
    "mood": "string | null"
  },

  "text_overlays": {
    "present": "boolean",
    "texts": ["string"],
    "pos": ["string"],
    "lang": ["string"]
  },

  "quick_edits": {
    "present": "boolean",
    "types": ["string"],
    "freq": "string | null"
  },

  "timeline": [
    {
      "ts": "string",
      "desc": "string",
      "hum": ["string"],
      "ani": ["string"],
      "objs": ["string"],
      "cam": ["string"],
      "aud": ["string"],
      "scn_change": "boolean"
    }
  ],

  "recognizable": {
    "people": ["string"],
    "places": ["string"],
    "events": ["string"]
  },

  "historical_context": {
    "period": "string | null",
    "evidence": ["string"]
  },

  "tags": ["string"],

  "ai_artifacts": {
    "vis": "boolean",
    "aud": "boolean",
    "vis_signs": ["string"],
    "aud_signs": ["string"],
    "notes": "string"
  }
}"""


# -------------------------------------------------------------------
# Prompt builder (updated with new instructions)
# -------------------------------------------------------------------
def build_prompt(asset_json: dict, media_type: str) -> str:
    media_line = (
        "You are analyzing a REAL video clip for a stock media library."
        if media_type == "video"
        else "You are analyzing a REAL image for a stock media library."
    )

    template = f"""
You are a constrained forensic video-analysis system. Your output must be strictly factual, concise, and fully aligned with the schema provided. Do not speculate, infer intent, or add information not directly observable in the media.

{media_line} You may be shown multiple frames extracted at equally spaced intervals throughout the video. These frames reveal the action, movement, and changes that occur over time. Your task is to extract maximum factual detail to help users find this video when searching for specific content.

STRICT RULES:
1. If uncertain, return null, false, or empty arrays.
2. Never guess identities, brands, demographics, locations, or AI-generation indicators.
3. Describe only what is visually or audibly present; no hidden motives or invented narrative.
4. Be specific, concrete, and observational. Identify species, object types, clothing, colors, lighting, and visible behaviours.
5. Follow the schema exactly. Do not add or remove fields.
6. Use consistent terminology across all fields.
7. Only report people, animals, objects, brands, or text that are clearly visible. Include all relevant observable details.
8. Timeline entries must be concrete, observable events tied to approximate timestamps.
9. Audio descriptions must reflect actual audible content (speech, events, noise, mood if clearly signalled).
10. Camera analysis must reflect observable motion, framing, shake, exposure, and focus behaviour.
11. Environment analysis must reflect visible lighting, surfaces, depth, and location type.
12. Human and animal behaviour must be strictly based on visible actions and interactions.

AI-ARTIFACT DETECTION RULES:
Check for the following visual indicators of AI generation:
- Anatomical distortions (hands, limbs, faces, eyes, teeth)
- Texture repetition or unnatural patterns
- Inconsistent lighting or shadows
- Physically impossible reflections or highlights
- Warped geometry, bending objects, or melting edges
- Temporal inconsistencies between frames (objects changing shape, position, or texture without cause)
- Incorrect motion blur or missing motion blur
- Overly smooth surfaces or plastic-like skin
- Flickering details or unstable fine textures
- Mismatched depth of field or inconsistent focus planes

Check for the following audio indicators of AI generation:
- Robotic or metallic voice timbre
- Unnatural prosody, pacing, or breath patterns
- Abrupt cutoffs or unnatural transitions
- Repetitive or looping background noise
- Phasey, warbling, or “underwater” artifacts
- Inconsistent room acoustics or mismatched reverb
- Audio that does not match visible actions or timing

Only report AI artifacts when they are clearly visible or audible.

ADDITIONAL DETAIL REQUIREMENTS:
- Describe progression, changes, continuity, and evolving action across frames.
- Identify changes in composition, lighting, subject position, and camera behaviour.
- Identify recognizable people, places, or events only if visually confirmed.
- If historical context is visually evident (clothing, technology, architecture), include it; otherwise return null.
- Extract all relevant visual and audible cues that would help a user search for this video in a stock library.

DEFINITIONS:
- “Brief Summary”: 1–2 sentences describing the core content.
- “Verbose Summary”: A detailed description of the sequence, context, and progression.
- “Movement Type”: steady, handheld, shaky, static, tracking, panning.
- “Timeline ts”: approximate time markers like 00:00, 00:05, 00:10.
- “Audio Events”: footsteps, traffic, wind, speech, animal noises.
- “Scene Change”: a clear shift in camera angle, location, or composition.

Return only valid JSON that conforms to the schema.

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
    except Exception:
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
    except Exception:
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

    # Prefer explicit media_type, but fall back to asset_type
    media_type = asset_json.get("media_type")
    asset_type = asset_json.get("asset_type")

    if not media_type:
        if asset_type == "video":
            media_type = "video"
        elif asset_type == "image":
            media_type = "image"
        else:
            media_type = "image"

    print("MEDIA_TYPE:", media_type)
    print("ASSET_TYPE:", asset_type)

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
