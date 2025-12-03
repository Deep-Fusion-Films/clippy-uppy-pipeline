from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import subprocess
import os
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Health check endpoint for Cloud Run
@app.get("/")
def health():
    return {"status": "healthy"}

class TranscodeRequest(BaseModel):
    file_name: str
    bucket: str
    target_format: Optional[str] = "mp4"

class TranscodeResponse(BaseModel):
    status: str
    output_path: Optional[str]

def transcode_video(input_path: str, output_path: str) -> bool:
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            output_path
        ]
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error("FFmpeg failed: %s", e)
        return False

@app.post("/transcode", response_model=TranscodeResponse)
async def transcode(request: TranscodeRequest):
    # In Cloud Run, you’ll likely need to pull the file from GCS here.
    input_path = f"/videos/{request.file_name}"
    base_name = os.path.splitext(request.file_name)[0]
    output_path = f"/transcoded/{base_name}.{request.target_format}"

    os.makedirs("/transcoded", exist_ok=True)

    success = transcode_video(input_path, output_path)
    if success:
        logging.info("Transcoded %s to %s", request.file_name, output_path)
        return TranscodeResponse(status="success", output_path=output_path)
    else:
        return TranscodeResponse(status="error", output_path=None)
