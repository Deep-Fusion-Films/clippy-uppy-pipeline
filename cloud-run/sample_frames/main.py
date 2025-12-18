import os
import cv2
import json
from typing import List, Tuple
from fastapi import FastAPI, Request, HTTPException
from google.cloud import storage
from datetime import datetime

app = FastAPI()

# Environment variables
TRANSCODED_BUCKET = os.getenv("TRANSCODED_BUCKET")  # e.g. gs://df-films-assets-euw1/transcoded
FRAMES_BUCKET = os.getenv("FRAMES_BUCKET")          # e.g. gs://df-films-assets-euw1/frames

storage_client = storage.Client()

def _bucket_name(uri: str) -> str:
    return uri.replace("gs://", "")

def download_from_gcs(bucket_uri: str, blob_name: str, local_path: str) -> str:
    bucket = storage_client.bucket(_bucket_name(bucket_uri))
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    return local_path

def upload_to_gcs(bucket_uri: str, blob_name: str, local_path: str) -> str:
    bucket = storage_client.bucket(_bucket_name(bucket_uri))
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path, content_type="image/jpeg")
    return f"gs://{_bucket_name(bucket_uri)}/{blob_name}"

def sample_keyframes(local_video_path: str, max_frames: int = 6, stride_seconds: float = 10.0) -> List[Tuple[int, int]]:
    cap = cv2.VideoCapture(local_video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video for sampling")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride_frames = int(fps * stride_seconds)
    indices = list(range(0, total_frames, max(stride_frames, 1)))[:max_frames]

    samples = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, _ = cap.read()
        if ok:
            timestamp_ms = int((idx / fps) * 1000)
            samples.append((idx, timestamp_ms))
        if len(samples) >= max_frames:
            break
    cap.release()
    return samples

def save_frame(local_video_path: str, frame_index: int, out_path: str) -> None:
    cap = cv2.VideoCapture(local_video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to reopen video to save frame")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame at index {frame_index}")
    cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

@app.get("/health")
def health():
    return {
        "status": "ok",
        "TRANSCODED_BUCKET": TRANSCODED_BUCKET,
        "FRAMES_BUCKET": FRAMES_BUCKET
    }

@app.post("/sample")
async def sample(req: Request):
    """
    Input JSON (two modes):
      Getty:
        {
          "asset_id": "getty-123456",
          "paths": {
            "transcoded": "gs://bucket/transcoded/getty-123456_normalized.mp4"
          },
          "config": { "max_frames": 6, "stride_seconds": 10.0 }
        }
      Local:
        {
          "file_name": "raw/CE_025_0.mp4",
          "bucket": "df-films-assets-euw1",
          "config": { "max_frames": 6, "stride_seconds": 10.0 }
        }
    """
    data = await req.json()

    # Mode 1: Getty asset_id + paths.transcoded
    if "asset_id" in data and "paths" in data and "transcoded" in data["paths"]:
        asset_id = data["asset_id"]
        transcoded_path = data["paths"]["transcoded"]
        transcoded_blob = f"{asset_id}_normalized.mp4"

    # Mode 2: Local file_name + bucket
    elif "file_name" in data and "bucket" in data:
        file_name = data["file_name"]  # e.g. raw/CE_025_0.mp4
        bucket = data["bucket"]
        asset_id = os.path.splitext(os.path.basename(file_name))[0]
        transcoded_path = f"gs://{bucket}/transcoded/{asset_id}_normalized.mp4"
        transcoded_blob = f"{asset_id}_normalized.mp4"

    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either asset_id+paths.transcoded (Getty) OR file_name+bucket (local)"
        )

    config = data.get("config", {})
    max_frames = int(config.get("max_frames", 6))
    stride_seconds = float(config.get("stride_seconds", 10.0))

    local_video = f"/tmp/{asset_id}_normalized.mp4"
    frames_prefix = f"{asset_id}/"

    # Download video
    download_from_gcs(TRANSCODED_BUCKET, transcoded_blob, local_video)

    # Sample frames
    samples = sample_keyframes(local_video_path=local_video, max_frames=max_frames, stride_seconds=stride_seconds)

    keyframe_paths = []
    scene_boundaries = []
    for i, (frame_idx, ts_ms) in enumerate(samples, start=1):
        local_frame = f"/tmp/{asset_id}_kf_{i:04}.jpg"
        save_frame(local_video, frame_idx, local_frame)
        gcs_frame_blob = f"{frames_prefix}{asset_id}_kf_{i:04}.jpg"
        gcs_frame_path = upload_to_gcs(FRAMES_BUCKET, gcs_frame_blob, local_frame)
        keyframe_paths.append(gcs_frame_path)
        scene_boundaries.append(ts_ms)

    return {
        "asset_id": asset_id,
        "paths": {
            "transcoded": transcoded_path,
            "frames": keyframe_paths
        },
        "frames": {
            "scene_boundaries": scene_boundaries,
            "keyframe_paths": keyframe_paths,
            "objects_detected": [],
            "faces_detected": [],
            "dominant_colors": []
        },
        "status": "frames_sampled",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
