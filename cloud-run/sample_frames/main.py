from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import cv2
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Request schema
class FrameRequest(BaseModel):
    file_name: str
    bucket: str

# Response schema
class FrameResponse(BaseModel):
    frame_paths: List[str]

# Frame extraction logic
def extract_key_frames(video_path: str, output_dir: str, interval_sec: int = 5) -> List[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_sec)
    frame_count = 0
    saved_frames = []

    os.makedirs(output_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_frames.append(frame_path)

        frame_count += 1

    cap.release()
    return saved_frames

# API endpoint
@app.post("/sample", response_model=FrameResponse)
async def sample_frames(request: FrameRequest):
    video_path = f"/videos/{request.file_name}"  # Assumes mounted or downloaded
    output_dir = f"/frames/{request.file_name.replace('.', '_')}"

    try:
        frames = extract_key_frames(video_path, output_dir)
        logging.info("Extracted %d frames from %s", len(frames), request.file_name)
        return FrameResponse(frame_paths=frames)
    except Exception as e:
        logging.error("Frame extraction failed: %s", e)
        return FrameResponse(frame_paths=[])

