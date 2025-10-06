import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import app

client = TestClient(app)

@pytest.fixture
def sample_payload():
    return {
        "asset_id": "abc123",
        "title": "Sunset Over London",
        "caption": "Editorial use only. A dramatic sunset over the Thames skyline.",
        "summary": "A visual moment capturing London's evening light.",
        "tags": ["London", "Sunset", "Editorial"],
        "licensing_flags": ["Editorial"],
        "embedding": [0.1, 0.2, 0.3, 0.4]
    }

@patch("main.db")
def test_store_metadata_success(mock_db, sample_payload):
    mock_doc = MagicMock()
    mock_doc.path = "media_metadata/abc123"
    mock_db.collection.return_value.document.return_value = mock_doc

    response = client.post("/store", json=sample_payload)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert data["document_path"] == "media_metadata/abc123"

def test_missing_fields():
    payload = {
        "asset_id": "xyz789",
        "title": "Missing Fields"
        # Missing caption, summary, tags, licensing_flags, embedding
    }

    response = client.post("/store", json=payload)
    assert response.status_code == 422  # Unprocessable Entity

@patch("main.db")
def test_firestore_failure(mock_db, sample_payload):
    mock_doc = MagicMock()
    mock_doc.set.side_effect = Exception("Firestore write failed")
    mock_doc.path = ""
    mock_db.collection.return_value.document.return_value = mock_doc

    response = client.post("/store", json=sample_payload)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "error"
    assert data["document_path"] == ""

