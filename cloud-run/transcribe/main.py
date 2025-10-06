from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import subprocess
import os
import logging

import whisper

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Load Whisper model (base for speed, change to "medium" or "large" if needed)
model = whisper.load_model("base")

# Request schema
class TranscribeRequest(BaseModel):
    file_name: str
    bucket: str
    language: Optional[str] = None  # Optional override

# Response schema
class TranscribeResponse(BaseModel):
    status: str
    transcript: Optional[str]

# Audio extraction logic
def extract_audio(input_path: str, output_path: str) -> bool:
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            output_path
        ]
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error("Audio extraction failed: %s", e)
        return False

# API endpoint
@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(request: TranscribeRequest):
    input_path = f"/videos/{request.file_name}"
    audio_path = f"/audio/{request.file_name.replace('.', '_')}.wav"

    os.makedirs("/audio", exist_ok=True)

    if not extract_audio(input_path, audio_path):
        return TranscribeResponse(status="error", transcript=None)

    try:
        result = model.transcribe(audio_path, language=request.language)
        transcript = result.get("text", "").strip()
        logging.info("Transcription complete for %s", request.file_name)
        return TranscribeResponse(status="success", transcript=transcript)
    except Exception as e:
        logging.error("Transcription failed: %s", e)
        return TranscribeResponse(status="error", transcript=None)

