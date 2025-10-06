import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_generate_embedding_success():
    payload = {
        "title": "Street Food Vendor in Bangkok",
        "caption": "Editorial use only. A vendor prepares traditional dishes at a lively night market in Bangkok.",
        "summary": "A cultural snapshot of Thailand’s street food scene.",
        "tags": ["Travel", "Culture", "Food"]
    }

    response = client.post("/embed", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "embedding" in data
    assert isinstance(data["embedding"], list)
    assert len(data["embedding"]) > 0
    assert all(isinstance(x, float) for x in data["embedding"])

def test_missing_fields():
    payload = {
        "title": "Quiet Forest Trail",
        # Missing caption, summary, tags
    }

    response = client.post("/embed", json=payload)
    assert response.status_code == 422  # Unprocessable Entity

def test_empty_input():
    payload = {
        "title": "",
        "caption": "",
        "summary": "",
        "tags": []
    }

    response = client.post("/embed", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "embedding" in data
    assert isinstance(data["embedding"], list)

