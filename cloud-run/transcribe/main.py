import os
import subprocess
import json
import logging
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from google.cloud import storage
import whisper

app = FastAPI()

logger = logging.getLogger("uvicorn.error")

# --------------------------------------------------------------------
# Environment variables
# --------------------------------------------------------------------
TRANSCODED_BUCKET = os.getenv("TRANSCODED_BUCKET")
AUDIO_BUCKET = os.getenv("AUDIO_BUCKET")
TRANSCRIPTS_BUCKET = os.getenv("TRANSCRIPTS_BUCKET")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

if not TRANSCODED_BUCKET or not AUDIO_BUCKET or not TRANSCRIPTS_BUCKET:
    raise RuntimeError(
        "TRANSCODED_BUCKET, AUDIO_BUCKET, and TRANSCRIPTS_BUCKET must be set."
    )

storage_client = storage.Client()
model = None  # loaded at startup


# --------------------------------------------------------------------
# Load Whisper ONCE at container startup
# --------------------------------------------------------------------
@app.on_event("startup")
def load_whisper_model():
    global model
    logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
    model = whisper.load_model(WHISPER_MODEL)
    logger.info("Whisper model loaded successfully.")


# --------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------
def normalize_bucket(bucket_uri: str) -> str:
    if bucket_uri.startswith("gs://"):
        return bucket_uri[len("gs://"):]
    return bucket_uri


def download_from_gcs(bucket_uri: str, blob_name: str, local_path: str) -> str:
    bucket_name = normalize_bucket(bucket_uri)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    logger.info(f"DOWNLOAD: gs://{bucket_name}/{blob_name} -> {local_path}")

    if not blob.exists():
        msg = f"GCS object not found: gs://{bucket_name}/{blob_name}"
        logger.error(msg)
        raise HTTPException(status_code=404, detail=msg)

    blob.download_to_filename(local_path)
    return local_path


def upload_to_gcs(bucket_uri: str, blob_name: str, local_path: str) -> str:
    bucket_name = normalize_bucket(bucket_uri)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    logger.info(f"UPLOAD: {local_path} -> gs://{bucket_name}/{blob_name}")

    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{blob_name}"


# --------------------------------------------------------------------
# Health check
# --------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "TRANSCODED_BUCKET": normalize_bucket(TRANSCODED_BUCKET),
        "AUDIO_BUCKET": normalize_bucket(AUDIO_BUCKET),
        "TRANSCRIPTS_BUCKET": normalize_bucket(TRANSCRIPTS_BUCKET),
        "WHISPER_MODEL": WHISPER_MODEL,
    }


# --------------------------------------------------------------------
# Main transcription endpoint
# --------------------------------------------------------------------
@app.post("/transcribe")
async def transcribe(req: Request):
    try:
        data = await req.json()
        logger.info(f"REQUEST BODY: {data}")

        # ------------------------------------------------------------
        # Mode 1: Getty
        # ------------------------------------------------------------
        if "asset_id" in data and "paths" in data and "transcoded" in data["paths"]:
            asset_id = data["asset_id"]
            transcoded_blob_name = f"transcoded/{asset_id}_normalized.mp4"
            logger.info(f"MODE: Getty | asset_id={asset_id}")

        # ------------------------------------------------------------
        # Mode 2: Local GCS
        # ------------------------------------------------------------
        elif "file_name" in data and "bucket" in data:
            file_name = data["file_name"]
            bucket = data["bucket"]
            asset_id = os.path.splitext(os.path.basename(file_name))[0]
            transcoded_blob_name = f"transcoded/{asset_id}_normalized.mp4"
            logger.info(
                f"MODE: Local | file_name={file_name} bucket={bucket} asset_id={asset_id}"
            )

        else:
            msg = "Provide either asset_id+paths.transcoded OR file_name+bucket."
            logger.error(f"BAD REQUEST: {msg} | data={data}")
            raise HTTPException(status_code=400, detail=msg)

        # Local temp paths
        local_video = f"/tmp/{asset_id}_normalized.mp4"
        local_audio = f"/tmp/{asset_id}.wav"
        local_transcript = f"/tmp/{asset_id}.json"

        # ------------------------------------------------------------
        # Download transcoded video
        # ------------------------------------------------------------
        logger.info("STEP: download_from_gcs")
        download_from_gcs(TRANSCODED_BUCKET, transcoded_blob_name, local_video)

        # ------------------------------------------------------------
        # Extract audio with ffmpeg
        # ------------------------------------------------------------
        logger.info("STEP: ffmpeg extract audio")
        try:
            ffmpeg_proc = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    local_video,
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    local_audio,
                ],
                capture_output=True,
                text=True,
            )
            if ffmpeg_proc.returncode != 0:
                msg = (
                    f"FFmpeg failed with code {ffmpeg_proc.returncode}: "
                    f"{ffmpeg_proc.stderr}"
                )
                logger.error(msg)
                raise HTTPException(status_code=500, detail=msg)
        except FileNotFoundError:
            msg = "ffmpeg binary not found in container."
            logger.error(msg)
            raise HTTPException(status_code=500, detail=msg)

        # ------------------------------------------------------------
        # Upload audio
        # ------------------------------------------------------------
        logger.info("STEP: upload audio")
        audio_blob_name = f"audio/{asset_id}.wav"
        gcs_audio = upload_to_gcs(AUDIO_BUCKET, audio_blob_name, local_audio)

        # ------------------------------------------------------------
        # Whisper transcription
        # ------------------------------------------------------------
        logger.info("STEP: whisper transcribe")
        result = model.transcribe(local_audio)

        transcript_json = {
            "text": result.get("text", ""),
            "language": result.get("language", "en"),
            "segments": result.get("segments", []),
            "confidence": None,
        }

        # ------------------------------------------------------------
        # Save transcript JSON
        # ------------------------------------------------------------
        logger.info("STEP: save transcript JSON")
        with open(local_transcript, "w") as f:
            json.dump(transcript_json, f, indent=2)

        transcript_blob_name = f"transcripts/{asset_id}.json"
        logger.info("STEP: upload transcript JSON")
        gcs_transcript = upload_to_gcs(
            TRANSCRIPTS_BUCKET, transcript_blob_name, local_transcript
        )

        # ------------------------------------------------------------
        # Response
        # ------------------------------------------------------------
        logger.info(f"SUCCESS: asset_id={asset_id}")
        return {
            "asset_id": asset_id,
            "paths": {
                "audio": gcs_audio,
                "transcript": gcs_transcript,
            },
            "transcript": transcript_json,
            "status": "transcribed",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    except HTTPException:
        # Already logged with detail above
        raise
    except Exception as e:
        logger.exception(f"UNEXPECTED ERROR for request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
