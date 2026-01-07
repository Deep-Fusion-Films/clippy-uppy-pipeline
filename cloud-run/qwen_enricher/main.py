import os
import json
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
import requests

app = FastAPI()

# -------------------------------------------------------------------
# Environment variables (Groq)
# -------------------------------------------------------------------
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY must be set")


# -------------------------------------------------------------------
# Health check
# -------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": GROQ_MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE
    }


# -------------------------------------------------------------------
# Prompt builder (MAXIMUM METADATA VERSION)
# -------------------------------------------------------------------
def build_prompt(asset_json: dict) -> str:
    """Build a rich editorial prompt using all available metadata."""

    # Getty metadata (optional)
    title = asset_json.get("getty", {}).get("title", "")
    caption = asset_json.get("getty", {}).get("caption", "")
    keywords = asset_json.get("getty", {}).get("keywords", [])

    # Technical metadata
    tech = asset_json.get("technical", {})
    codec = tech.get("codec")
    resolution = tech.get("resolution")
    duration = tech.get("duration")
    bitrate = tech.get("bitrate")
    frame_rate = tech.get("frame_rate")

    # Transcript
    transcript_text = asset_json.get("transcript", {}).get("text", "")
    transcript_excerpt = transcript_text[:1000] if transcript_text else ""

    # Frames
    frames = asset_json.get("frames", {})
    objects = frames.get("objects_detected", [])
    faces = frames.get("faces_detected", [])
    dominant_colors = frames.get("dominant_colors", [])
    scene_boundaries = frames.get("scene_boundaries", [])

    prompt = f"""
You are an editorial enrichment assistant. Use ALL provided metadata to produce a factual, grounded, non-sensational enrichment.

### Getty Metadata
Title: {title}
Caption: {caption}
Keywords: {", ".join(keywords)}

### Technical Metadata
Codec: {codec}
Resolution: {resolution}
Duration: {duration}
Bitrate: {bitrate}
Frame rate: {frame_rate}

### Transcript (excerpt)
{transcript_excerpt}

### Frame Analysis
Objects detected: {", ".join(objects)}
Faces detected: {faces}
Dominant colors: {", ".join(dominant_colors)}
Scene boundaries (frame indices): {scene_boundaries}

### Task
Return ONLY valid JSON with the following fields:

- "summary": A 3–4 sentence editorial summary describing the content, context, and visual narrative.
- "tags": 12–18 concise tags (lowercase, hyphenated if multiword).
- "tone": A single descriptive word (e.g., "informative", "documentary", "cinematic").
- "entities": list of named entities inferred from transcript or visuals (people, places, objects).
- "themes": list of thematic concepts (e.g., "rural life", "travel", "nature", "solitude").
- "visual_style": description of the visual style inferred from frames (e.g., "wide-shot", "color-rich", "natural light").
- "audio_style": description of the audio tone inferred from transcript (e.g., "calm narration", "ambient sound").
- "scene_summary": 1–2 sentence summary of what happens in the key scene(s).
- "confidence": a numeric confidence score (0–1) for the enrichment.

Return ONLY JSON. No commentary, no markdown.
"""

    return prompt.strip()


# -------------------------------------------------------------------
# Groq inference
# -------------------------------------------------------------------
def run_groq(prompt: str) -> dict:
    """Call Groq's OpenAI-compatible API and return parsed JSON."""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are an editorial enrichment assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "response_format": {"type": "json_object"}
    }

    resp = requests.post(GROQ_API_URL, json=payload, headers=headers)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    content = resp.json()["choices"][0]["message"]["content"]

    # Parse JSON returned by the model
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail=f"Model returned invalid JSON: {content}"
        )


# -------------------------------------------------------------------
# Enrichment endpoint
# -------------------------------------------------------------------
@app.post("/enrich")
async def enrich(req: Request):
    asset_json = await req.json()

    asset_id = asset_json.get("asset_id")
    if not asset_id:
        raise HTTPException(status_code=400, detail="asset_id required")

    # Build prompt from full metadata
    prompt = build_prompt(asset_json)

    # Run Groq model
    enriched = run_groq(prompt)

    # Attach enrichment
    asset_json["qwen"] = enriched  # keep same field name for pipeline compatibility

    asset_json["status"] = "enriched"
    asset_json["timestamp"] = datetime.utcnow().isoformat() + "Z"

    return asset_json
