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


def split_gs_uri(gs_uri: str):
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gs_uri}")
    without = gs_uri[len("gs://"):]
    bucket, _, blob = without.partition("/")
    return bucket, blob


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


def download_from_gcs_bucket_name(bucket_name: str, blob_name: str, local_path: str):
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
        # Determine asset_id
        # ------------------------------------------------------------
        if "asset_id" in data:
            asset_id = data["asset_id"]
        elif "file_name" in data:
            asset_id = os.path.splitext(os.path.basename(data["file_name"]))[0]
        else:
            msg = "Missing asset_id or file_name."
            logger.error(msg)
            raise HTTPException(status_code=400, detail=msg)

        # Local temp paths
        local_video = f"/tmp/{asset_id}_normalized.mp4"
        local_audio = f"/tmp/{asset_id}.wav"
        local_transcript = f"/tmp/{asset_id}.json"

        # ------------------------------------------------------------
        # 1) Prefer upstream audio if provided
        # ------------------------------------------------------------
        audio_source_local = None
        paths = data.get("paths") or {}
        transcoded = paths.get("transcoded")

        if isinstance(transcoded, dict) and "audio" in transcoded:
            audio_uri = transcoded["audio"]
            logger.info(f"MODE: using provided audio file {audio_uri}")

            bucket_name, blob_name = split_gs_uri(audio_uri)
            audio_source_local = f"/tmp/{asset_id}_source_audio"
            download_from_gcs_bucket_name(bucket_name, blob_name, audio_source_local)

        else:
            # ------------------------------------------------------------
            # 2) Fall back to extracting audio from video
            # ------------------------------------------------------------
            logger.info("MODE: extracting audio from video")

            transcoded_blob_name = f"transcoded/{asset_id}_normalized.mp4"
            download_from_gcs(TRANSCODED_BUCKET, transcoded_blob_name, local_video)

            logger.info("STEP: ffmpeg extract audio")
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

                if ffmpeg_proc.returncode == 0:
                    audio_source_local = local_audio
                else:
                    # No audio stream → treat as silent
                    logger.warning(
                        f"No audio stream detected for asset {asset_id}. "
                        f"FFmpeg stderr: {ffmpeg_proc.stderr}"
                    )
                    audio_source_local = None

            except FileNotFoundError:
                msg = "ffmpeg binary not found in container."
                logger.error(msg)
                raise HTTPException(status_code=500, detail=msg)

        # ------------------------------------------------------------
        # Handle silent videos
        # ------------------------------------------------------------
        if audio_source_local is None:
            logger.info(f"ASSET {asset_id} HAS NO AUDIO — returning empty transcript")

            transcript_json = {
                "text": "",
                "language": "en",
                "segments": [],
                "confidence": None,
                "note": "No audio detected in source video",
            }

            with open(local_transcript, "w") as f:
                json.dump(transcript_json, f, indent=2)

            transcript_blob_name = f"transcripts/{asset_id}.json"
            gcs_transcript = upload_to_gcs(
                TRANSCRIPTS_BUCKET, transcript_blob_name, local_transcript
            )

            return {
                "asset_id": asset_id,
                "paths": {
                    "audio": None,
                    "transcript": gcs_transcript,
                },
                "transcript": transcript_json,
                "status": "no_audio",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

        # ------------------------------------------------------------
        # Upload audio (wav or mp3)
        # ------------------------------------------------------------
        ext = os.path.splitext(audio_source_local)[1]
        audio_blob_name = f"audio/{asset_id}{ext}"
        gcs_audio = upload_to_gcs(AUDIO_BUCKET, audio_blob_name, audio_source_local)

        # ------------------------------------------------------------
        # Whisper transcription
        # ------------------------------------------------------------
        logger.info("STEP: whisper transcribe")
        result = model.transcribe(audio_source_local)

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
        raise
    except Exception as e:
        logger.exception(f"UNEXPECTED ERROR for request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
