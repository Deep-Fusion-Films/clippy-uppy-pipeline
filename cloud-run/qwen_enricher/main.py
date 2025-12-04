import os
import json
from datetime import datetime
from fastapi import FastAPI, Request

app = FastAPI()

# Environment variables (configure model settings as needed)
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen2.5")   # placeholder name for your chosen local/open-source model
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

@app.get("/health")
def health():
    return {
        "status": "ok",
        "QWEN_MODEL": QWEN_MODEL,
        "MAX_TOKENS": MAX_TOKENS,
        "TEMPERATURE": TEMPERATURE
    }

def build_prompt(asset_json: dict) -> str:
    """Create a deterministic prompt from prior blocks."""
    title = asset_json.get("getty", {}).get("title", "")
    caption = asset_json.get("getty", {}).get("caption", "")
    keywords = asset_json.get("getty", {}).get("keywords", [])
    tech = asset_json.get("technical", {})
    transcript_text = asset_json.get("transcript", {}).get("text", "")
    frames = asset_json.get("frames", {})
    objects = frames.get("objects_detected", [])
    dominant_colors = frames.get("dominant_colors", [])

    prompt = f"""
You are an editorial enrichment assistant.
Asset title: {title}
Caption: {caption}
Keywords: {", ".join(keywords)}
Technical: codec={tech.get("codec")}, resolution={tech.get("resolution")}, duration={tech.get("duration")}
Transcript (truncated): {transcript_text[:300]}
Objects detected: {", ".join(objects)}
Dominant colors: {", ".join(dominant_colors)}

Produce JSON with:
- "summary": 3-4 sentence editorial summary, factual, non-sensational.
- "tags": 8-12 concise tags (lowercase, hyphenated if multiword).
- "tone": single word descriptor ("informative", "documentary", "cinematic", etc.).

Return only JSON, no extra text.
"""
    return prompt.strip()

def run_qwen(prompt: str) -> dict:
    """
    Placeholder inference function.
    Replace with your local model call or API client.
    Must return a dict with keys: summary, tags, tone.
    """
    # Deterministic stub for pipeline integration; swap with real model call.
    return {
        "summary": "A grounded editorial overview of the asset’s content, context, and visual narrative.",
        "tags": [
            "documentary", "cinematic", "travel", "outdoor",
            "nature", "landscape", "wide-shot", "color-rich"
        ],
        "tone": "informative"
    }

@app.post("/enrich")
async def enrich(req: Request):
    """
    Input: unified JSON (partial) containing getty, technical, transcript, frames, paths.
    Output: same JSON plus 'qwen' block with summary, tags, tone.
    """
    asset_json = await req.json()
    asset_id = asset_json.get("asset_id")

    prompt = build_prompt(asset_json)
    qwen_out = run_qwen(prompt)

    asset_json["qwen"] = {
        "summary": qwen_out.get("summary"),
        "tags": qwen_out.get("tags", []),
        "tone": qwen_out.get("tone")
    }
    asset_json["status"] = "enriched"
    asset_json["timestamp"] = datetime.utcnow().isoformat() + "Z"

    return asset_json
