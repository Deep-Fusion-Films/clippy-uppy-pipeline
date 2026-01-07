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
# Prompt builder (Documentary Image Analysis)
# -------------------------------------------------------------------
def build_prompt(asset_json: dict) -> str:
    """
    Build the documentary‑grade image/video analysis prompt.
    """

    return f"""
You are an image/video analyst for a factual documentary. Your task is to produce highly discriminative, non-generic analyses that can reliably distinguish between hundreds of visually similar images. Base all conclusions strictly on visible evidence and supplied metadata. Do not invent facts.

Return ONLY valid JSON matching the schema below:

{{
  "description_long": string,
  "entities": {{
    "people": [
      {{
        "name": string|null,
        "role": string|null,
        "clothing": string[],
        "age_range": string|null,
        "facial_expression": string|null,
        "pose": string|null
      }}
    ],
    "places": [
      {{
        "name": string|null,
        "sublocation": string|null,
        "indoor_outdoor": "indoor"|"outdoor"|null
      }}
    ],
    "orgs": [
      {{
        "name": string,
        "evidence": string|null
      }}
    ]
  }},
  "time": {{
    "year": number|null,
    "month": number|null,
    "day": number|null,
    "confidence": number,
    "hint_text": string|null
  }},
  "objects": [
    {{ "label": string, "salience": number }}
  ],
  "activities": [
    {{ "label": string, "confidence": number, "who": string|null }}
  ],
  "themes": [ string ],
  "composition": {{
    "camera_angle": string|null,
    "focal_length_est": string|null,
    "depth_of_field": string|null,
    "lighting": string|null,
    "color_palette": string|null,
    "contrast_style": string|null,
    "orientation": string|null
  }},
  "text_in_image": [ string ],
  "distinguishing_features": [ string ],
  "story_use": [ "opener" | "bridge" | "chapter_art" | "context" | "climax" | "reveal" ],
  "safety": {{ "sensitive": boolean, "notes": string|null }}
}}

Guidelines:
- "description_long" must be 2–4 sentences focusing on unique, concrete, differentiating details.
- Use null or empty arrays when uncertain; reduce confidence accordingly.
- Do not infer identities, locations, or organizations without visible evidence.
- Prefer micro-details (lighting direction, clothing textures, object wear, spatial relationships) over generic descriptors.
- Output must be a single JSON object with no commentary, no markdown, and no surrounding text.

Now analyze the provided image metadata and produce the JSON response.

### Provided Metadata
{json.dumps(asset_json, indent=2)}
""".strip()


# -------------------------------------------------------------------
# Gemini Flash inference
# -------------------------------------------------------------------
def run_gemini(prompt: str) -> dict:
    """Call Gemini Flash and return parsed JSON."""

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
    except json.JSONDecodeError:
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
