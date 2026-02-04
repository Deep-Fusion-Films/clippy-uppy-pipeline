# Batch Runner

Processes N random unprocessed assets from a GCS folder and sends them to the start-pipeline service.

Environment variables:
- BUCKET
- FOLDER
- START_PIPELINE_URL
- COUNT

Run locally:
docker build -t batch-runner .
docker run --rm -e BUCKET=... -e FOLDER=... -e START_PIPELINE_URL=... -e COUNT=5 batch-runner
