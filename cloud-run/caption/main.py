from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
import logging
import transformers

from prompt_templates import build_editorial_prompt

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

# Caption endpoint
@app.post("/caption", response_model=CaptionResponse)
async def generate_caption(request: CaptionRequest):
    prompt = build_editorial_prompt(
        scene_description=request.scene_description,
        transcript=request.transcript
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=512)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # TODO: Replace with structured parsing logic
    logging.info("Raw model output:\n%s", decoded)

    return CaptionResponse(
        title="Generated Title",
        caption="Generated Caption",
        summary="Generated Summary",
        tags=["Tag1", "Tag2"],
        licensing_flags=["Editorial"]
    )

