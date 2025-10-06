import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import app

client = TestClient(app)

@pytest.fixture
def sample_payload():
    return {
        "file_name": "sample_video.mp4",
        "bucket": "test-bucket",
        "language": "en"
    }

@patch("main.extract_audio")
@patch("main.model")
def test_transcribe_success(mock_model, mock_extract, sample_payload):
    mock_extract.return_value = True
    mock_model.transcribe.return_value = {"text": "Hello world"}

    response = client.post("/transcribe", json=sample_payload)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert data["transcript"] == "Hello world"

@patch("main.extract_audio")
def test_audio_extraction_failure(mock_extract, sample_payload):
    mock_extract.return_value = False

    response = client.post("/transcribe", json=sample_payload)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "error"
    assert data["transcript"] is None

def test_missing_fields():
    payload = {
        "file_name": "video.mp4"
        # Missing bucket
    }

    response = client.post("/transcribe", json=payload)
    assert response.status_code == 422  # Unprocessable Entity

