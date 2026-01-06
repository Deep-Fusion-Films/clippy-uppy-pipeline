import os
import subprocess
import json
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from google.cloud import storage

app = FastAPI()

# --------------------------------------------------------------------
# Environment variables
# --------------------------------------------------------------------
# IMPORTANT:
#   ASSETS_BUCKET      = "gs://df-films-assets-euw1"
#   TRANSCODED_BUCKET  = "gs://df-films-assets-euw1"
#
# Folder prefixes (raw/, transcoded/) are handled IN CODE, not in env vars.
# --------------------------------------------------------------------
ASSETS_BUCKET = os.getenv("ASSETS_BUCKET")          # e.g. gs://df-films-assets-euw1
TRANSCODED_BUCKET = os.getenv("TRANSCODED_BUCKET")  # e.g. gs://df-films-assets-euw1

if not ASSETS_BUCKET or not TRANSCODED_BUCKET:
    raise RuntimeError("ASSETS_BUCKET and TRANSCODED_BUCKET environment variables must be set.")

storage_client = storage.Client()


def normalize_bucket(bucket_uri: str) -> str:
    """
    Normalize a bucket URI or name to a bare bucket name.

    Accepts:
      - "gs://my-bucket"
      - "my-bucket"

    Returns:
      - "my-bucket"
    """
    if bucket_uri.startswith("gs://"):
        return bucket_uri[len("gs://"):]
    return bucket_uri


def download_from_gcs(bucket_uri: str, blob_name: str, local_path: str) -> str:
    """Download file from GCS to local /tmp for processing."""
    bucket_name = normalize_bucket(bucket_uri)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        raise HTTPException(
            status_code=404,
            detail=f"GCS object not found: gs://{bucket_name}/{blob_name}"
        )

    blob.download_to_filename(local_path)
    return local_path


def upload_to_gcs(bucket_uri: str, blob_name: str, local_path: str) -> str:
    """Upload file from local /tmp back to GCS."""
    bucket_name = normalize_bucket(bucket_uri)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{blob_name}"


def probe_metadata(local_path: str) -> dict:
    """Use ffprobe to extract technical metadata."""
    probe = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries",
            "format=duration,size:stream=codec_name,bit_rate,width,height,r_frame_rate",
            "-of", "json",
            local_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    meta = json.loads(probe.stdout)

    if "streams" not in meta or not meta["streams"]:
        raise HTTPException(status_code=500, detail="ffprobe returned no streams.")

    stream = next((st for st in meta["streams"] if st.get("codec_name")), meta["streams"][0])

    return {
        "codec": stream.get("codec_name"),
        "bitrate": stream.get("bit_rate"),
        "frame_rate": stream.get("r_frame_rate"),
        "resolution": f"{stream.get('width')}x{stream.get('height')}",
        "duration": meta.get("format", {}).get("duration"),
        "file_size": meta.get("format", {}).get("size"),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ASSETS_BUCKET": normalize_bucket(ASSETS_BUCKET),
        "TRANSCODED_BUCKET": normalize_bucket(TRANSCODED_BUCKET),
    }


@app.post("/transcode")
async def transcode(req: Request):
    """
    Normalize video and return technical metadata.

    Supports two modes:

    1) Getty mode:
       {
         "asset_id": "...",
         "paths": {
           "raw": "gs://df-films-assets-euw1/raw/<asset_id>.mp4"
         }
       }

    2) Local GCS mode:
       {
         "file_name": "raw/CE_025_0.mp4",
         "bucket": "df-films-assets-euw1"
       }
    """
    data = await req.json()

    # ----------------------------------------------------------------
    # Mode 1: Getty asset_id + paths.raw
    # ----------------------------------------------------------------
    if "asset_id" in data and "paths" in data and "raw" in data["paths"]:
        asset_id = data["asset_id"]
        raw_path = data["paths"]["raw"]  # full gs://.../raw/<asset_id>.mp4

        # We assume ASSETS_BUCKET points to the same bucket as raw_path's bucket.
        # Blob name is the object path under that bucket.
        # e.g. raw/<asset_id>.mp4
        blob_name = f"raw/{asset_id}.mp4"

        local_in = f"/tmp/{asset_id}.mp4"
        local_out = f"/tmp/{asset_id}_normalized.mp4"

    # ----------------------------------------------------------------
    # Mode 2: Local file_name + bucket
    # ----------------------------------------------------------------
    elif "file_name" in data and "bucket" in data:
        file_name = data["file_name"]      # e.g. "raw/CE_025_0.mp4"
        bucket = data["bucket"]            # e.g. "df-films-assets-euw1"

        asset_id = os.path.splitext(os.path.basename(file_name))[0]

        # Construct full GCS raw path
        raw_path = f"gs://{bucket}/{file_name}"

        # For download, we still use ASSETS_BUCKET as the bucket,
        # and file_name as the blob name (e.g. "raw/CE_025_0.mp4")
        blob_name = file_name

        local_in = f"/tmp/{asset_id}.mp4"
        local_out = f"/tmp/{asset_id}_normalized.mp4"

    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either asset_id+paths.raw (Getty) OR file_name+bucket (local)",
        )

    # ----------------------------------------------------------------
    # Download raw video from ASSETS bucket
    # ----------------------------------------------------------------
    download_from_gcs(ASSETS_BUCKET, blob_name, local_in)

    # ----------------------------------------------------------------
    # CLOUD RUN SAFE FFMPEG COMMAND
    # ----------------------------------------------------------------
    # We avoid depending on codecs that might be missing in some builds.
    # mpeg4 + libmp3lame are widely supported in Debian ffmpeg.
    try:
        ffmpeg_proc = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", local_in,
                "-c:v", "mpeg4",
                "-qscale:v", "2",
                "-c:a", "libmp3lame",
                "-qscale:a", "2",
                local_out,
            ],
            capture_output=True,
            text=True,
        )
        if ffmpeg_proc.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"FFmpeg failed with code {ffmpeg_proc.returncode}: {ffmpeg_proc.stderr}",
            )
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="ffmpeg binary not found in container.")

    # ----------------------------------------------------------------
    # Upload transcoded video to TRANSCODED bucket under transcoded/
    # ----------------------------------------------------------------
    transcoded_blob_name = f"transcoded/{asset_id}_normalized.mp4"
    gcs_out = upload_to_gcs(TRANSCODED_BUCKET, transcoded_blob_name, local_out)

    # ----------------------------------------------------------------
    # Extract technical metadata from the transcoded file
    # ----------------------------------------------------------------
    technical = probe_metadata(local_out)

    # ----------------------------------------------------------------
    # Build JSON response
    # ----------------------------------------------------------------
    result = {
        "asset_id": asset_id,
        "paths": {
            "raw": raw_path,
            "transcoded": gcs_out,
        },
        "technical": technical,
        "status": "transcoded",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    return result
