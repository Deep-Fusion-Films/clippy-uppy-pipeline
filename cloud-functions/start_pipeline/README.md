## Start Pipeline – Cloud Function

This module is the start point to the Clippy-Uppy pipeline. It is triggered when a video file is uploaded to the `getty-ingest-media` bucket situated on GCS (Google Cloud Storage). Once activated, it extracts the file name and bucket name, then sends a POST request to the `/transcode` Cloud Run service to begin processing.

## 🚀 Trigger Details

- **Event Type**: `google.storage.object.finalize`  
- **Trigger Bucket**: `getty-ingest-media`  
- **Payload Sent To**: `cloud-run/transcode`  
- **Payload Format**:

```json
{
  "file_name": "example_video.mov",
  "bucket": "getty-ingest-media"
}
