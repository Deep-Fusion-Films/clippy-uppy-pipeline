import os
import json
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from google import genai

app = FastAPI()

# -------------------------------------------------------------------
# Environment variables
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
# Schema block (safe literal string)
# -------------------------------------------------------------------
SCHEMA_BLOCK = """
Schema (types):
{
  "description_long": string,
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
  "story_use": [ "opener" | "bridge" | "chapter_art" | "context" | "climax" | "reveal" ],
  "safety": { "sensitive": boolean, "notes": string|null }
}
"""


# -------------------------------------------------------------------
# Prompt builder
# -------------------------------------------------------------------
def build_prompt(asset_json: dict) -> str:
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
# Extract text safely from google‑genai response
# -------------------------------------------------------------------
def extract_text(response) -> str:
    # Preferred: response.text
    if hasattr(response, "text") and isinstance(response.text, str):
        return response.text

    # Fallback: candidates[0].content.parts[0].text
    try:
        candidates = getattr(response, "candidates", None)
        if candidates:
            parts = candidates[0].content.parts
            if parts and hasattr(parts[0], "text"):
                return parts[0].text
    except Exception:
        pass

    # Last resort: try JSON
    try:
        data = json.loads(response.to_json())
        if isinstance(data, dict) and "text" in data:
            return data["text"]
    except Exception:
        pass

    raise HTTPException(
        status_code=500,
        detail="Model response did not contain text output."
    )


# -------------------------------------------------------------------
# Gemini Flash inference (correct API signature)
# -------------------------------------------------------------------
def run_gemini(prompt: str) -> dict:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_TOKENS,
        response_mime_type="application/json"
    )

    text = extract_text(response)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail=f"Model returned invalid JSON: {text}"
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
