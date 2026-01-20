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
ASSETS_BUCKET = os.getenv("ASSETS_BUCKET")          # e.g. gs://df-films-assets-euw1
TRANSCODED_BUCKET = os.getenv("TRANSCODED_BUCKET")  # e.g. gs://df-films-assets-euw1

if not ASSETS_BUCKET or not TRANSCODED_BUCKET:
    raise RuntimeError("ASSETS_BUCKET and TRANSCODED_BUCKET must be set.")

storage_client = storage.Client()


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def normalize_bucket(bucket_uri: str) -> str:
    """Convert gs://bucket → bucket."""
    return bucket_uri.replace("gs://", "")


def extract_blob_name(full_gs_url: str) -> str:
    """
    Convert:
        gs://bucket/getty/123.mp4
    →   getty/123.mp4
    """
    if not full_gs_url.startswith("gs://"):
        raise HTTPException(status_code=400, detail=f"Invalid GCS URL: {full_gs_url}")

    without_scheme = full_gs_url[len("gs://"):]
    parts = without_scheme.split("/", 1)
    if len(parts) != 2:
        raise HTTPException(status_code=400, detail=f"Invalid GCS URL: {full_gs_url}")

    return parts[1]  # object path only


def download_from_gcs(bucket_uri: str, blob_name: str, local_path: str) -> str:
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
    bucket_name = normalize_bucket(bucket_uri)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{blob_name}"


def probe_video_metadata(local_path: str) -> dict:
    """
    Extract technical metadata for video from a (presumably valid) media file.
    Raises HTTPException only if ffprobe itself is missing or misbehaves.
    """
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries",
                "stream=codec_name,bit_rate,width,height,r_frame_rate,"
                "format=duration,size",
                "-of", "json",
                local_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="ffprobe binary not found.")
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail=f"ffprobe failed on transcoded video: {e.stderr}"
        )

    meta = json.loads(probe.stdout or "{}")
    streams = meta.get("streams", [])
    if not streams:
        raise HTTPException(status_code=500, detail="ffprobe returned no video streams.")

    stream = streams[0]
    fmt = meta.get("format", {})

    return {
        "codec": stream.get("codec_name"),
        "bitrate": stream.get("bit_rate"),
        "frame_rate": stream.get("r_frame_rate"),
        "resolution": f"{stream.get('width')}x{stream.get('height')}",
        "duration": fmt.get("duration"),
        "file_size": fmt.get("size"),
    }


def probe_audio_metadata(local_path: str) -> dict | None:
    """
    Extract technical metadata for audio, if present.
    Returns dict if audio stream exists and is readable, otherwise None.
    Never raises for asset-level issues; only for ffprobe missing.
    """
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries",
                "stream=codec_name,bit_rate,channels,sample_rate,"
                "format=duration,size",
                "-of", "json",
                local_path,
            ],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="ffprobe binary not found.")

    if proc.returncode != 0:
        # Treat as "no usable audio metadata"
        return None

    try:
        meta = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return None

    streams = meta.get("streams", [])
    if not streams:
        return None

    stream = streams[0]
    fmt = meta.get("format", {})

    return {
        "codec": stream.get("codec_name"),
        "bitrate": stream.get("bit_rate"),
        "channels": stream.get("channels"),
        "sample_rate": stream.get("sample_rate"),
        "duration": fmt.get("duration"),
        "file_size": fmt.get("size"),
    }


def safe_probe_input(local_path: str) -> tuple[bool, str | None]:
    """
    Lightweight pre-flight check: is this a valid media file ffprobe can read?
    Returns (is_valid, error_message_if_any).
    """
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json",
                local_path,
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return False, proc.stderr.strip() or "ffprobe returned non-zero exit code"
        try:
            meta = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError:
            return False, "ffprobe produced invalid JSON"
        if "format" not in meta:
            return False, "ffprobe found no format information"
        return True, None
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="ffprobe binary not found.")


def has_audio_stream(local_path: str) -> bool:
    """
    Check if the input file has at least one audio stream.
    """
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=index",
                "-of", "csv=p=0",
                local_path,
            ],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="ffprobe binary not found.")

    if proc.returncode != 0:
        return False

    # If ffprobe prints at least one line, there is an audio stream
    return bool(proc.stdout.strip())


# --------------------------------------------------------------------
# Health
# --------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "ASSETS_BUCKET": normalize_bucket(ASSETS_BUCKET),
        "TRANSCODED_BUCKET": normalize_bucket(TRANSCODED_BUCKET),
    }


# --------------------------------------------------------------------
# TRANSCODE ENDPOINT
# --------------------------------------------------------------------
@app.post("/transcode")
async def transcode(req: Request):
    data = await req.json()

    # ------------------------------------------------------------
    # MODE 1: Getty (asset_id + paths.raw)
    # ------------------------------------------------------------
    if "asset_id" in data and "paths" in data and "raw" in data["paths"]:
        asset_id = data["asset_id"]
        raw_path = data["paths"]["raw"]  # full gs://bucket/getty/<id>.mp4

        blob_name = extract_blob_name(raw_path)
        local_in = f"/tmp/{asset_id}.mp4"
        local_video_out = f"/tmp/{asset_id}_normalized.mp4"
        local_audio_out = f"/tmp/{asset_id}_audio.mp3"

    # ------------------------------------------------------------
    # MODE 2: Local GCS (file_name + bucket)
    # ------------------------------------------------------------
    elif "file_name" in data and "bucket" in data:
        file_name = data["file_name"]      # e.g. raw/CE_025_0.mp4
        bucket = data["bucket"]            # e.g. df-films-assets-euw1

        asset_id = os.path.splitext(os.path.basename(file_name))[0]
        raw_path = f"gs://{bucket}/{file_name}"

        blob_name = file_name
        local_in = f"/tmp/{asset_id}.mp4"
        local_video_out = f"/tmp/{asset_id}_normalized.mp4"
        local_audio_out = f"/tmp/{asset_id}_audio.mp3"

    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either asset_id+paths.raw (Getty) OR file_name+bucket (local)",
        )

    # ------------------------------------------------------------
    # Download original video
    # ------------------------------------------------------------
    download_from_gcs(ASSETS_BUCKET, blob_name, local_in)

    # ------------------------------------------------------------
    # Pre-flight: validate input with ffprobe
    # ------------------------------------------------------------
    is_valid, probe_error = safe_probe_input(local_in)
    if not is_valid:
        return {
            "asset_id": asset_id,
            "paths": {
                "raw": raw_path,
                "transcoded": {
                    "video": None,
                    "audio": None,
                },
            },
            "technical": {
                "video": None,
                "audio": None,
            },
            "status": {
                "video": "skipped_invalid_input",
                "audio": "skipped_invalid_input",
            },
            "reason": f"ffprobe could not read input: {probe_error}",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    # ------------------------------------------------------------
    # Transcode video using ffmpeg
    # ------------------------------------------------------------
    try:
        ffmpeg_video = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", local_in,
                "-c:v", "mpeg4",
                "-qscale:v", "2",
                "-an",  # drop audio in this pass
                local_video_out,
            ],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="ffmpeg binary not found.")

    video_status = "transcoded"
    video_reason = None

    if ffmpeg_video.returncode != 0:
        video_status = "transcode_failed"
        video_reason = ffmpeg_video.stderr[:2000]
        local_video_out = None

    # ------------------------------------------------------------
    # Audio extraction (optional, non-blocking)
    # ------------------------------------------------------------
    audio_status = "not_attempted"
    audio_reason = None
    audio_gcs_path = None
    audio_technical = None

    if has_audio_stream(local_in):
        try:
            ffmpeg_audio = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i", local_in,
                    "-vn",
                    "-acodec", "libmp3lame",
                    "-qscale:a", "2",
                    local_audio_out,
                ],
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="ffmpeg binary not found.")

        if ffmpeg_audio.returncode == 0:
            # Upload audio
            audio_blob_name = f"transcoded/{asset_id}_audio.mp3"
            audio_gcs_path = upload_to_gcs(TRANSCODED_BUCKET, audio_blob_name, local_audio_out)
            audio_status = "extracted"
            audio_technical = probe_audio_metadata(local_audio_out)
        else:
            audio_status = "audio_extraction_failed"
            audio_reason = ffmpeg_audio.stderr[:2000]
    else:
        audio_status = "no_audio_present"

    # ------------------------------------------------------------
    # Upload transcoded video (if successful)
    # ------------------------------------------------------------
    video_gcs_path = None
    video_technical = None

    if local_video_out is not None and video_status == "transcoded":
        transcoded_blob_name = f"transcoded/{asset_id}_normalized.mp4"
        video_gcs_path = upload_to_gcs(TRANSCODED_BUCKET, transcoded_blob_name, local_video_out)
        video_technical = probe_video_metadata(local_video_out)

    # ------------------------------------------------------------
    # Response (organised / nested)
    # ------------------------------------------------------------
    response = {
        "asset_id": asset_id,
        "paths": {
            "raw": raw_path,
            "transcoded": {
                "video": video_gcs_path,
                "audio": audio_gcs_path,
            },
        },
        "technical": {
            "video": video_technical,
            "audio": audio_technical,
        },
        "status": {
            "video": video_status,
            "audio": audio_status,
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    # Attach reasons only when they exist
    reasons = {}
    if video_reason:
        reasons["video_reason"] = video_reason
    if audio_reason:
        reasons["audio_reason"] = audio_reason
    if reasons:
        response["reason"] = reasons

    return response
