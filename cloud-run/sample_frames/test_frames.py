import pytest
from fastapi.testclient import TestClient
from main import app
import os

client = TestClient(app)

def test_sample_frames_success(tmp_path):
    # Simulate a short video using OpenCV (1 black frame every second for 5 seconds)
    import cv2
    import numpy as np

    video_path = tmp_path / "test_video.mp4"
    out = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 1, (640, 480))
    for _ in range(5):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    # Create /videos mount point and copy video there
    os.makedirs("/videos", exist_ok=True)
    mounted_path = f"/videos/test_video.mp4"
    os.rename(video_path, mounted_path)

    payload = {
        "file_name": "test_video.mp4",
        "bucket": "dummy-bucket"
    }

    response = client.post("/sample", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "frame_paths" in data
    assert isinstance(data["frame_paths"], list)
    assert len(data["frame_paths"]) > 0
    assert all(path.endswith(".jpg") for path in data["frame_paths"])

def test_missing_fields():
    payload = {
        "file_name": "video.mp4"
        # Missing bucket
    }

    response = client.post("/sample", json=payload)
    assert response.status_code == 422  # Unprocessable Entity

def test_nonexistent_video():
    payload = {
        "file_name": "nonexistent.mp4",
        "bucket": "dummy-bucket"
    }

    response = client.post("/sample", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["frame_paths"] == []

