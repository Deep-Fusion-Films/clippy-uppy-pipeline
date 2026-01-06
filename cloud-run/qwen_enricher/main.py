import os
import json
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
import requests

app = FastAPI()

# -------------------------------------------------------------------
# Environment variables
# -------------------------------------------------------------------
QWEN_API_URL = os.getenv("QWEN_API_URL")  # e.g. https://api.openai.com/v1/chat/completions
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen2.5")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

if not QWEN_API_URL:
    raise RuntimeError("QWEN_API_URL must be set")
if not QWEN_API_KEY:
    raise RuntimeError("QWEN_API_KEY must be set")


# -------------------------------------------------------------------
# Health check
# -------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "QWEN_MODEL": QWEN_MODEL,
        "MAX_TOKENS": MAX_TOKENS,
        "TEMPERATURE": TEMPERATURE
    }


# -------------------------------------------------------------------
# Prompt builder
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

    # Transcript
    transcript_text = asset_json.get("transcript", {}).get("text", "")
    transcript_excerpt = transcript_text[:500] if transcript_text else ""

    # Frames
    frames = asset_json.get("frames", {})
    objects = frames.get("objects_detected", [])
    dominant_colors = frames.get("dominant_colors", [])

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

### Transcript (excerpt)
{transcript_excerpt}

### Frame Analysis
Objects detected: {", ".join(objects)}
Dominant colors: {", ".join(dominant_colors)}

### Task
Return ONLY valid JSON with the following fields:

- "summary": A 3–4 sentence editorial summary describing the content, context, and visual narrative.
- "tags": 8–12 concise tags (lowercase, hyphenated if multiword).
- "tone": A single descriptive word (e.g., "informative", "documentary", "cinematic").

Return ONLY JSON. No commentary, no markdown.
"""

    return prompt.strip()


# -------------------------------------------------------------------
# Qwen inference
# -------------------------------------------------------------------
def run_qwen(prompt: str) -> dict:
    """Call an OpenAI-compatible API endpoint and return parsed JSON."""

    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": QWEN_MODEL,
        "messages": [
            {"role": "system", "content": "You are an editorial enrichment assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE
    }

    resp = requests.post(QWEN_API_URL, json=payload, headers=headers)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    content = resp.json()["choices"][0]["message"]["content"]

    # The model returns JSON as a string → parse it
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

    # Run model
    qwen_out = run_qwen(prompt)

    # Attach enrichment
    asset_json["qwen"] = {
        "summary": qwen_out.get("summary"),
        "tags": qwen_out.get("tags", []),
        "tone": qwen_out.get("tone")
    }

    asset_json["status"] = "enriched"
    asset_json["timestamp"] = datetime.utcnow().isoformat() + "Z"

    return asset_json
