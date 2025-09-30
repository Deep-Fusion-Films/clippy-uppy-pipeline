This Cloud Function is the entry point for the Clippy Uppy metadata pipeline. 
It is automatically triggered when a new video file is uploaded to your Google Cloud Storage bucket (getty-ingest-media). 
The function sends the file information to the first Cloud Run service (typically /transcode) in order to begin the enrichment process.
