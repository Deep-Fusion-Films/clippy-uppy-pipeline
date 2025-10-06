import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_enrich_tags_success():
    payload = {
        "tags": ["Travel", "Culture", "Food", "Sunset", "Unicorn"]
    }

    response = client.post("/enrich", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "filtered_tags" in data
    assert isinstance(data["filtered_tags"], list)

    # Assuming "Unicorn" is not in Getty taxonomy
    assert "Unicorn" not in data["filtered_tags"]
    assert "Travel" in data["filtered_tags"]

def test_empty_tags():
    payload = {
        "tags": []
    }

    response = client.post("/enrich", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["filtered_tags"] == []

def test_missing_tags_field():
    payload = {}  # No 'tags' key

    response = client.post("/enrich", json=payload)
    assert response.status_code == 422  # Unprocessable Entity

def test_case_insensitivity():
    payload = {
        "tags": ["travel", "CULTURE", "FoOd"]
    }

    response = client.post("/enrich", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert sorted(data["filtered_tags"]) == sorted(["travel", "culture", "food"])

