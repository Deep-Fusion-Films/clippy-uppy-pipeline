from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import os
import logging
import transformers

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Load Qwen model and tokenizer
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen-7B-Chat")
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Request schema
class CaptionRequest(BaseModel):
    scene_description: str
    transcript: str

# Response schema
class CaptionResponse(BaseModel):
    title: str
    caption: str
    summary: str
    tags: List[str]
    licensing_flags: List[str]

# Prompt template
def build_prompt(scene: str, transcript: str) -> str:
    return (
        f"Generate a Getty-style editorial caption for a video scene.\n"
        f"Scene: {scene}\n"
        f"Transcript: {transcript}\n"
        f"Return: title, caption, summary, tags, licensing_flags\n"
        f"Tone: editorial, descriptive, non-commercial"
    )

# Caption endpoint
@app.post("/caption", response_model=CaptionResponse)
async def generate_caption(request: CaptionRequest):
    prompt = build_prompt(request.scene_description, request.transcript)

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=512)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Parse output (placeholder logic – replace with structured parsing)
    # You may use regex, JSON parsing, or structured delimiters
    return CaptionResponse(
        title="Generated Title",
        caption="Generated Caption",
        summary="Generated Summary",
        tags=["Tag1", "Tag2"],
        licensing_flags=["Editorial"]
    )

