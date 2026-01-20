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
        raise HTTPException(status_code=400, detail=f"Invalid GCS URL: {full_gCS_url}")

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


def probe_metadata(local_path: str) -> dict:
    """
    Extract technical metadata from a (presumably valid) media file.
    Raises HTTPException only if ffprobe itself is missing or misbehaves.
    """
    try:
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
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="ffprobe binary not found.")
    except subprocess.CalledProcessError as e:
        # ffprobe couldn't parse the file – treat as invalid media
        raise HTTPException(
            status_code=500,
            detail=f"ffprobe failed on transcoded output: {e.stderr}"
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
        # If ffprobe produced no JSON or empty format, treat as invalid
        try:
            meta = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError:
            return False, "ffprobe produced invalid JSON"
        if "format" not in meta:
            return False, "ffprobe found no format information"
        return True, None
    except FileNotFoundError:
        # If ffprobe is missing, that's a service misconfig, not an asset issue
        raise HTTPException(status_code=500, detail="ffprobe binary not found.")


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

        # Extract blob name from full GCS URL
        blob_name = extract_blob_name(raw_path)

        local_in = f"/tmp/{asset_id}.mp4"
        local_out = f"/tmp/{asset_id}_normalized.mp4"

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
        local_out = f"/tmp/{asset_id}_normalized.mp4"

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
        # Do NOT blow up the pipeline – return a graceful "skipped" result
        return {
            "asset_id": asset_id,
            "paths": {
                "raw": raw_path,
                "transcoded": None,
            },
            "technical": None,
            "status": "skipped_invalid_input",
            "reason": f"ffprobe could not read input: {probe_error}",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    # ------------------------------------------------------------
    # Transcode using ffmpeg
    # ------------------------------------------------------------
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
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="ffmpeg binary not found.")

    if ffmpeg_proc.returncode != 0:
        # Again: do NOT kill the pipeline – report a soft failure
        return {
            "asset_id": asset_id,
            "paths": {
                "raw": raw_path,
                "transcoded": None,
            },
            "technical": None,
            "status": "transcode_failed",
            "reason": f"FFmpeg failed: {ffmpeg_proc.stderr[:2000]}",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    # ------------------------------------------------------------
    # Upload transcoded file
    # ------------------------------------------------------------
    transcoded_blob_name = f"transcoded/{asset_id}_normalized.mp4"
    gcs_out = upload_to_gcs(TRANSCODED_BUCKET, transcoded_blob_name, local_out)

    # ------------------------------------------------------------
    # Extract metadata from transcoded output
    # ------------------------------------------------------------
    technical = probe_metadata(local_out)

    # ------------------------------------------------------------
    # Response
    # ------------------------------------------------------------
    return {
        "asset_id": asset_id,
        "paths": {
            "raw": raw_path,
            "transcoded": gcs_out,
        },
        "technical": technical,
        "status": "transcoded",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
