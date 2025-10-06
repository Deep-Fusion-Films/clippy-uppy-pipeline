import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from main import app

client = TestClient(app)

@pytest.fixture
def sample_payload():
    return {
        "file_name": "sample_video.mp4",
        "bucket": "test-bucket",
        "target_format": "mp4"
    }

@patch("main.transcode_video")
def test_transcode_success(mock_transcode, sample_payload):
    mock_transcode.return_value = True

    response = client.post("/transcode", json=sample_payload)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert data["output_path"].endswith(".mp4")

@patch("main.transcode_video")
def test_transcode_failure(mock_transcode, sample_payload):
    mock_transcode.return_value = False

    response = client.post("/transcode", json=sample_payload)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "error"
    assert data["output_path"] is None

def test_missing_fields():
    payload = {
        "file_name": "video.mp4"
        # Missing bucket
    }

    response = client.post("/transcode", json=payload)
    assert response.status_code == 422  # Unprocessable Entity

