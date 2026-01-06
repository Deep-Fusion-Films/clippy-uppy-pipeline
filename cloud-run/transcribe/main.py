import os
import subprocess
import json
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from google.cloud import storage
import whisper  # OpenAI Whisper model

app = FastAPI()

# --------------------------------------------------------------------
# Environment variables
# --------------------------------------------------------------------
# IMPORTANT:
#   TRANSCODED_BUCKET  = "gs://df-films-assets-euw1"
#   AUDIO_BUCKET       = "gs://df-films-assets-euw1"
#   TRANSCRIPTS_BUCKET = "gs://df-films-metadata-euw1"
#
# Folder prefixes (transcoded/, audio/, transcripts/) are handled IN CODE.
# --------------------------------------------------------------------
TRANSCODED_BUCKET = os.getenv("TRANSCODED_BUCKET")   # e.g. gs://df-films-assets-euw1
AUDIO_BUCKET = os.getenv("AUDIO_BUCKET")             # e.g. gs://df-films-assets-euw1
TRANSCRIPTS_BUCKET = os.getenv("TRANSCRIPTS_BUCKET") # e.g. gs://df-films-metadata-euw1
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

if not TRANSCODED_BUCKET or not AUDIO_BUCKET or not TRANSCRIPTS_BUCKET:
    raise RuntimeError(
        "TRANSCODED_BUCKET, AUDIO_BUCKET, and TRANSCRIPTS_BUCKET environment variables must be set."
    )

storage_client = storage.Client()
model = whisper.load_model(WHISPER_MODEL)


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
            detail=f"GCS object not found: gs://{bucket_name}/{blob_name}",
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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "TRANSCODED_BUCKET": normalize_bucket(TRANSCODED_BUCKET),
        "AUDIO_BUCKET": normalize_bucket(AUDIO_BUCKET),
        "TRANSCRIPTS_BUCKET": normalize_bucket(TRANSCRIPTS_BUCKET),
        "WHISPER_MODEL": WHISPER_MODEL,
    }


@app.post("/transcribe")
async def transcribe(req: Request):
    """
    Transcribe audio from a transcoded video.

    Supports two modes:

    1) Getty mode:
       {
         "asset_id": "...",
         "paths": {
           "transcoded": "gs://df-films-assets-euw1/transcoded/<asset_id>_normalized.mp4"
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
    # Mode 1: Getty asset_id + paths.transcoded
    # ----------------------------------------------------------------
    if "asset_id" in data and "paths" in data and "transcoded" in data["paths"]:
        asset_id = data["asset_id"]
        transcoded_path = data["paths"]["transcoded"]  # full gs://.../transcoded/<asset_id>_normalized.mp4

        # We assume TRANSCODED_BUCKET points to the same bucket as transcoded_path's bucket.
        # Blob name is the object path under that bucket.
        # e.g. transcoded/<asset_id>_normalized.mp4
        transcoded_blob_name = f"transcoded/{asset_id}_normalized.mp4"

    # ----------------------------------------------------------------
    # Mode 2: Local file_name + bucket
    # ----------------------------------------------------------------
    elif "file_name" in data and "bucket" in data:
        file_name = data["file_name"]      # e.g. "raw/CE_025_0.mp4"
        bucket = data["bucket"]            # e.g. "df-films-assets-euw1"

        asset_id = os.path.splitext(os.path.basename(file_name))[0]

        # Construct full GCS transcoded path based on a convention:
        # transcoded/<asset_id>_normalized.mp4
        transcoded_blob_name = f"transcoded/{asset_id}_normalized.mp4"
        transcoded_path = f"gs://{bucket}/{transcoded_blob_name}"

    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either asset_id+paths.transcoded (Getty) OR file_name+bucket (local)",
        )

    # Local paths
    local_video = f"/tmp/{asset_id}_normalized.mp4"
    local_audio = f"/tmp/{asset_id}.wav"
    local_transcript = f"/tmp/{asset_id}.json"

    # ----------------------------------------------------------------
    # Download transcoded video from TRANSCODED bucket
    # ----------------------------------------------------------------
    download_from_gcs(TRANSCODED_BUCKET, transcoded_blob_name, local_video)

    # ----------------------------------------------------------------
    # Extract audio with ffmpeg
    # ----------------------------------------------------------------
    try:
        ffmpeg_proc = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", local_video,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                local_audio,
            ],
            capture_output=True,
            text=True,
        )
        if ffmpeg_proc.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"FFmpeg audio extraction failed with code {ffmpeg_proc.returncode}: {ffmpeg_proc.stderr}",
            )
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="ffmpeg binary not found in container.")

    # ----------------------------------------------------------------
    # Upload audio to AUDIO bucket under audio/
    # ----------------------------------------------------------------
    audio_blob_name = f"audio/{asset_id}.wav"
    gcs_audio = upload_to_gcs(AUDIO_BUCKET, audio_blob_name, local_audio)

    # ----------------------------------------------------------------
    # Run Whisper transcription locally
    # ----------------------------------------------------------------
    result = model.transcribe(local_audio)

    transcript_json = {
        "text": result.get("text", ""),
        "language": result.get("language", "en"),
        "segments": result.get("segments", []),
        "confidence": None,  # Whisper doesn’t provide confidence
    }

    # ----------------------------------------------------------------
    # Save transcript JSON to GCS under transcripts/
    # ----------------------------------------------------------------
    with open(local_transcript, "w") as f:
        json.dump(transcript_json, f, indent=2)

    transcript_blob_name = f"transcripts/{asset_id}.json"
    gcs_transcript = upload_to_gcs(TRANSCRIPTS_BUCKET, transcript_blob_name, local_transcript)

    # ----------------------------------------------------------------
    # Build response JSON
    # ----------------------------------------------------------------
    return {
        "asset_id": asset_id,
        "paths": {
            "transcoded": transcoded_path,
            "audio": gcs_audio,
            "transcript": gcs_transcript,
        },
        "transcript": transcript_json,
        "status": "transcribed",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
