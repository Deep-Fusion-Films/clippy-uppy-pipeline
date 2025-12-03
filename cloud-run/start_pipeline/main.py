from fastapi import FastAPI, Request
import os, requests, logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Cloud Run endpoint for the transcode service
TRANSCODE_URL = os.getenv("TRANSCODE_URL", "https://transcode-service-xyz.a.run.app/transcode")

# Create FastAPI app
app = FastAPI()

@app.post("/start-pipeline")
async def start_pipeline(request: Request):
    """
    HTTP endpoint for triggering the pipeline.
    Expects JSON payload with 'name' and 'bucket'.
    """
    data = await request.json()
    file_name = data.get("name")
    bucket = data.get("bucket")

    if not file_name or not bucket:
        logging.error("Missing file name or bucket in request payload.")
        return {"error": "Missing file name or bucket"}

    payload = {"file_name": file_name, "bucket": bucket}

    try:
        response = requests.post(TRANSCODE_URL, json=payload, timeout=10)
        response.raise_for_status()
        logging.info(f"Pipeline triggered for {file_name}. Status: {response.status_code}")
        return {"status": "ok"}
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to trigger transcode service: {e}")
        return {"error": str(e)}
