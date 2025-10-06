from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import subprocess
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Request schema
class TranscodeRequest(BaseModel):
    file_name: str
    bucket: str
    target_format: Optional[str] = "mp4"

# Response schema
class TranscodeResponse(BaseModel):
    status: str
    output_path: Optional[str]

# Transcoding logic
def transcode_video(input_path: str, output_path: str) -> bool:
    try:
        cmd = [
            "ffmpeg",
            "-y",  # overwrite output
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

# API endpoint
@app.post("/transcode", response_model=TranscodeResponse)
async def transcode(request: TranscodeRequest):
    input_path = f"/videos/{request.file_name}"  # Assumes mounted or downloaded
    base_name = os.path.splitext(request.file_name)[0]
    output_path = f"/transcoded/{base_name}.{request.target_format}"

    os.makedirs("/transcoded", exist_ok=True)

    success = transcode_video(input_path, output_path)
    if success:
        logging.info("Transcoded %s to %s", request.file_name, output_path)
        return TranscodeResponse(status="success", output_path=output_path)
    else:
        return TranscodeResponse(status="error", output_path=None)

