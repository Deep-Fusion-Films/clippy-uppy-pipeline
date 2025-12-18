import os
import subprocess
import json
from fastapi import FastAPI, Request, HTTPException
from google.cloud import storage
from datetime import datetime
import whisper  # OpenAI Whisper model

app = FastAPI()

# Environment variables
TRANSCODED_BUCKET = os.getenv("TRANSCODED_BUCKET")   # e.g. gs://df-films-assets-euw1/transcoded
AUDIO_BUCKET = os.getenv("AUDIO_BUCKET")             # e.g. gs://df-films-assets-euw1/audio
TRANSCRIPTS_BUCKET = os.getenv("TRANSCRIPTS_BUCKET") # e.g. gs://df-films-metadata-euw1/transcripts
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

storage_client = storage.Client()
model = whisper.load_model(WHISPER_MODEL)

def download_from_gcs(bucket_uri: str, blob_name: str, local_path: str):
    bucket_name = bucket_uri.replace("gs://", "")
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    return local_path

def upload_to_gcs(bucket_uri: str, blob_name: str, local_path: str):
    bucket_name = bucket_uri.replace("gs://", "")
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{blob_name}"

@app.get("/health")
def health():
    return {
        "status": "ok",
        "TRANSCODED_BUCKET": TRANSCODED_BUCKET,
        "AUDIO_BUCKET": AUDIO_BUCKET,
        "TRANSCRIPTS_BUCKET": TRANSCRIPTS_BUCKET,
        "WHISPER_MODEL": WHISPER_MODEL
    }

@app.post("/transcribe")
async def transcribe(req: Request):
    data = await req.json()

    # Mode 1: Getty asset_id + paths.transcoded
    if "asset_id" in data and "paths" in data and "transcoded" in data["paths"]:
        asset_id = data["asset_id"]
        transcoded_path = data["paths"]["transcoded"]  # e.g. gs://bucket/transcoded/getty-123456_normalized.mp4
        local_video = f"/tmp/{asset_id}_normalized.mp4"
        blob_name = f"{asset_id}_normalized.mp4"

    # Mode 2: Local file_name + bucket
    elif "file_name" in data and "bucket" in data:
        file_name = data["file_name"]  # e.g. raw/CE_025_0.mp4
        bucket = data["bucket"]        # e.g. df-films-assets-euw1
        asset_id = os.path.splitext(os.path.basename(file_name))[0]
        transcoded_path = f"gs://{bucket}/transcoded/{asset_id}_normalized.mp4"
        local_video = f"/tmp/{asset_id}_normalized.mp4"
        blob_name = f"{asset_id}_normalized.mp4"

    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either asset_id+paths.transcoded (Getty) OR file_name+bucket (local)"
        )

    # Local paths for audio and transcript
    local_audio = f"/tmp/{asset_id}.wav"
    local_transcript = f"/tmp/{asset_id}.json"

    # Download transcoded video
    download_from_gcs(TRANSCODED_BUCKET, blob_name, local_video)

    # Extract audio with ffmpeg
    subprocess.run([
        "ffmpeg", "-y", "-i", local_video,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        local_audio
    ], check=True)

    # Upload audio to GCS
    gcs_audio = upload_to_gcs(AUDIO_BUCKET, f"{asset_id}.wav", local_audio)

    # Run Whisper transcription
    result = model.transcribe(local_audio)

    transcript_json = {
        "text": result["text"],
        "language": result.get("language", "en"),
        "segments": result.get("segments", []),
        "confidence": None  # Whisper doesn’t provide confidence
    }

    # Save transcript JSON to GCS
    with open(local_transcript, "w") as f:
        json.dump(transcript_json, f, indent=2)
    gcs_transcript = upload_to_gcs(TRANSCRIPTS_BUCKET, f"{asset_id}.json", local_transcript)

    # Build response JSON
    return {
        "asset_id": asset_id,
        "paths": {
            "transcoded": transcoded_path,
            "audio": gcs_audio,
            "transcript": gcs_transcript
        },
        "transcript": transcript_json,
        "status": "transcribed",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
