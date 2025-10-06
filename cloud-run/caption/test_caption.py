import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_generate_caption_success():
    payload = {
        "scene_description": "A woman walks through a bright sunflower field wearing a straw hat.",
        "transcript": "She reflects on her childhood summers and the joy of being outdoors."
    }

    response = client.post("/caption", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "title" in data
    assert "caption" in data
    assert "summary" in data
    assert isinstance(data["tags"], list)
    assert isinstance(data["licensing_flags"], list)
    assert "Editorial" in data["licensing_flags"]

def test_missing_fields():
    payload = {
        "scene_description": "A quiet forest trail in autumn."
        # transcript is missing
    }

    response = client.post("/caption", json=payload)
    assert response.status_code == 422  # Unprocessable Entity

def test_empty_input():
    payload = {
        "scene_description": "",
        "transcript": ""
    }

    response = client.post("/caption", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["title"] != ""
    assert data["caption"] != ""

