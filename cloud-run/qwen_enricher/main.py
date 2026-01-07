import os
import json
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from google import genai

app = FastAPI()

# -------------------------------------------------------------------
# Environment variables (Gemini Flash)
# -------------------------------------------------------------------
VERTEX_API_KEY = os.getenv("VERTEX_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

if not VERTEX_API_KEY:
    raise RuntimeError("VERTEX_API_KEY must be set")

client = genai.Client(api_key=VERTEX_API_KEY)


# -------------------------------------------------------------------
# Health check
# -------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": GEMINI_MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE
    }


# -------------------------------------------------------------------
# Schema block (verbatim, safe — NOT inside an f-string)
# -------------------------------------------------------------------
SCHEMA_BLOCK = """
Schema (types):
{
  "description_long": string,                       // 2–4 sentences; unique details that distinguish this image
  "entities": {
    "people": [
      { "name": string|null, "role": string|null, "clothing": string[], "age_range": string|null, "facial_expression": string|null, "pose": string|null }
    ],
    "places": [
      { "name": string|null, "sublocation": string|null, "indoor_outdoor": "indoor"|"outdoor"|null }
    ],
    "orgs": [
      { "name": string, "evidence": string|null }
    ]
  },
  "time": { "year": number|null, "month": number|null, "day": number|null, "confidence": number, "hint_text": string|null },
  "objects": [ { "label": string, "salience": number } ],
  "activities": [ { "label": string, "confidence": number, "who": string|null } ],
  "themes": [ string ],
  "composition": {
    "camera_angle": string|null,                   // e.g., high/low/eye-level, over-shoulder
    "focal_length_est": string|null,               // e.g., wide/normal/telephoto look
    "depth_of_field": string|null,                 // shallow/deep/moderate
    "lighting": string|null,                       // direction, quality, time-of-day cues
    "color_palette": string|null,                  // dominant hues; warm/cool; saturation
    "contrast_style": string|null,                 // low/medium/high; soft/hard shadows
    "orientation": string|null                     // landscape/portrait/square
  },
  "text_in_image": [ string ],
  "distinguishing_features": [ string ],
  "story_use": [ "opener" | "bridge" | "chapter_art" | "context" | "climax" | "reveal" ],
  "safety": { "sensitive": boolean, "notes": string|null }
}
"""


# -------------------------------------------------------------------
# Prompt builder (NO f-string braces inside schema)
# -------------------------------------------------------------------
def build_prompt(asset_json: dict) -> str:
    # Use .format() to safely insert schema + metadata
    template = """
You are an image/video analyst for a factual documentary. Produce nuanced, discriminative analyses that can distinguish between hundreds of near-identical images.

Output STRICT JSON only, matching the schema below. Prefer concrete details (composition, lighting, micro-differences) over generic tags.

{schema}

Rules:
- Use only what is visible plus supplied tags; do not invent facts.
- If uncertain, use null/[] and reduce confidence.
- Keep output as a single JSON object with the exact keys above.

### Provided Metadata
{metadata}
"""
    return template.format(
        schema=SCHEMA_BLOCK,
        metadata=json.dumps(asset_json, indent=2)
    ).strip()


# -------------------------------------------------------------------
# Gemini Flash inference
# -------------------------------------------------------------------
def run_gemini(prompt: str) -> dict:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        generation_config={
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_TOKENS,
            "response_mime_type": "application/json"
        }
    )

    try:
        return json.loads(response.text)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail=f"Model returned invalid JSON: {response.text}"
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

    prompt = build_prompt(asset_json)
    enriched = run_gemini(prompt)

    asset_json["analysis"] = enriched
    asset_json["status"] = "enriched"
    asset_json["timestamp"] = datetime.utcnow().isoformat() + "Z"

    return asset_json
