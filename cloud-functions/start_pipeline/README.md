This Cloud Function is the entry point for the Clippy Uppy metadata pipeline. 
It is automatically triggered when a new video file is uploaded to your Google Cloud Storage bucket (). 
The function sends the file information to the first Cloud Run service (typically ) to begin the enrichment process.
