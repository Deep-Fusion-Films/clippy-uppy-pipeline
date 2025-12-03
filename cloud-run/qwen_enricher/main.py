from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Global model/tokenizer reference (lazy load)
model = None
tokenizer = None
generator = None

# Request schema
class EnrichRequest(BaseModel):
    text: str                     # Input text (caption, transcript, etc.)
    tasks: Optional[List[str]] = ["caption", "summary", "tags", "readme"]

# Response schema
class EnrichResponse(BaseModel):
    status: str
    outputs: dict

# Health check endpoint (required for Cloud Run)
@app.get("/")
def health():
    return {"status": "healthy"}

# Lazy model loader
def load_qwen_model():
    global model, tokenizer, generator
    if model is None or tokenizer is None or generator is None:
        model_name = os.getenv("MODEL_NAME", "Qwen/Qwen1.5-7B-Chat")
        logging.info(f"Loading Qwen model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# API endpoint
@app.post("/enrich", response_model=EnrichResponse)
async def enrich(request: EnrichRequest):
    try:
        load_qwen_model()
        outputs = {}

        for task in request.tasks:
            prompt = ""
            if task == "caption":
                prompt = f"Rewrite this caption to be editorially strong:\n{request.text}"
            elif task == "summary":
                prompt = f"Summarize the following transcript:\n{request.text}"
            elif task == "tags":
                prompt = f"Suggest 5 editorial tags for:\n{request.text}"
            elif task == "readme":
                prompt = f"Draft a README-style description for:\n{request.text}"
            else:
                continue

            logging.info(f"Running Qwen task: {task}")
            result = generator(prompt, max_length=256, do_sample=True, top_p=0.9)[0]["generated_text"]
            outputs[task] = result.strip()

        return EnrichResponse(status="success", outputs=outputs)

    except Exception as e:
        logging.error("Qwen enrichment failed: %s", e)
        return EnrichResponse(status="error", outputs={})

