import os
import subprocess
import json
from fastapi import FastAPI, Request, HTTPException
from google.cloud import storage
from datetime import datetime

app = FastAPI()

# Environment variables
ASSETS_BUCKET = os.getenv("ASSETS_BUCKET")       # e.g. gs://df-films-assets-euw1
TRANSCODED_BUCKET = os.getenv("TRANSCODED_BUCKET")  # e.g. gs://df-films-assets-euw1/transcoded

storage_client = storage.Client()

def download_from_gcs(bucket_uri: str, blob_name: str, local_path: str):
    """Download file from GCS to local /tmp for processing."""
    bucket_name = bucket_uri.replace("gs://", "")
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    return local_path

def upload_to_gcs(bucket_uri: str, blob_name: str, local_path: str):
    """Upload file from local /tmp back to GCS."""
    bucket_name = bucket_uri.replace("gs://", "")
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{blob_name}"

def probe_metadata(local_path: str):
    """Use ffprobe to extract technical metadata."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration,size:stream=codec_name,bit_rate,width,height,r_frame_rate",
         "-of", "json", local_path],
        capture_output=True, text=True, check=True
    )
    meta = json.loads(probe.stdout)
    stream = next((st for st in meta["streams"] if st.get("codec_name")), meta["streams"][0])
    return {
        "codec": stream.get("codec_name"),
        "bitrate": stream.get("bit_rate"),
        "frame_rate": stream.get("r_frame_rate"),
        "resolution": f"{stream.get('width')}x{stream.get('height')}",
        "duration": meta["format"].get("duration"),
        "file_size": meta["format"].get("size")
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "ASSETS_BUCKET": ASSETS_BUCKET,
        "TRANSCODED_BUCKET": TRANSCODED_BUCKET
    }

@app.post("/transcode")
async def transcode(req: Request):
    """Normalize video and return technical metadata."""
    data = await req.json()

    # Mode 1: Getty asset_id
    if "asset_id" in data and "paths" in data and "raw" in data["paths"]:
        asset_id = data["asset_id"]
        raw_path = data["paths"]["raw"]  # e.g. gs://bucket/raw/getty-123456.mp4
        local_in = f"/tmp/{asset_id}.mp4"
        local_out = f"/tmp/{asset_id}_normalized.mp4"
        blob_name = f"raw/{asset_id}.mp4"

    # Mode 2: Local file_name + bucket
    elif "file_name" in data and "bucket" in data:
        file_name = data["file_name"]  # e.g. raw/CE_025_0.mp4
        bucket = data["bucket"]        # e.g. df-films-assets-euw1
        asset_id = os.path.splitext(os.path.basename(file_name))[0]
        raw_path = f"gs://{bucket}/{file_name}"
        local_in = f"/tmp/{asset_id}.mp4"
        local_out = f"/tmp/{asset_id}_normalized.mp4"
        blob_name = file_name

    else:
        raise HTTPException(status_code=400,
                            detail="Provide either asset_id+paths.raw (Getty) OR file_name+bucket (local)")

    # Download raw video
    download_from_gcs(ASSETS_BUCKET, blob_name, local_in)

    # Run FFmpeg normalization
    subprocess.run([
        "ffmpeg", "-y", "-i", local_in,
        "-c:v", "libx264", "-preset", "fast", "-c:a", "aac",
        local_out
    ], check=True)

    # Upload transcoded video
    gcs_out = upload_to_gcs(TRANSCODED_BUCKET, f"{asset_id}_normalized.mp4", local_out)

    # Extract metadata
    technical = probe_metadata(local_out)

    # Build JSON block
    result = {
        "asset_id": asset_id,
        "paths": {
            "raw": raw_path,
            "transcoded": gcs_out
        },
        "technical": technical,
        "status": "transcoded",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    return result
