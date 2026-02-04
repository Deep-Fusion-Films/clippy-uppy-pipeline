import os
import random
import json
import subprocess
from google.cloud import storage

BUCKET = os.getenv("BUCKET", "df-films-assets-euw1")
FOLDER = os.getenv("FOLDER", "newsflare/newsflare_upload")
START_PIPELINE_URL = os.getenv("START_PIPELINE_URL")
COUNT = int(os.getenv("COUNT", "1"))
PROCESSED_FILE = "/data/processed.txt"

storage_client = storage.Client()


def load_processed():
    if not os.path.exists(PROCESSED_FILE):
        return set()
    with open(PROCESSED_FILE, "r") as f:
        return set(line.strip() for line in f.readlines())


def save_processed(processed):
    os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
    with open(PROCESSED_FILE, "w") as f:
        for item in processed:
            f.write(item + "\n")


def list_gcs_files():
    bucket = storage_client.bucket(BUCKET)
    blobs = bucket.list_blobs(prefix=FOLDER)
    return [blob.name for blob in blobs if blob.name.endswith(".mp4")]


def run_pipeline(file_name):
    payload = {
        "file_name": file_name,
        "bucket": BUCKET,
        "source": "newsflare"
    }

    token = subprocess.getoutput("gcloud auth print-identity-token")

    cmd = [
        "curl", "-X", "POST", START_PIPELINE_URL,
        "-H", f"Authorization: Bearer {token}",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(payload)
    ]

    print(f"Processing: {file_name}")
    result = subprocess.getoutput(" ".join(cmd))
    print(result)
    print("-" * 40)


def main():
    processed = load_processed()
    all_files = list_gcs_files()

    remaining = [f for f in all_files if f not in processed]

    if not remaining:
        print("No unprocessed files left.")
        return

    to_process = random.sample(remaining, k=min(COUNT, len(remaining)))

    for file_name in to_process:
        run_pipeline(file_name)
        processed.add(file_name)

    save_processed(processed)


if __name__ == "__main__":
    main()
