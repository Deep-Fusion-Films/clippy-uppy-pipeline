import json
import os
from jsonschema import validate, ValidationError

# Load schema once at startup
SCHEMA_PATH = os.getenv("SCHEMA_PATH", "schema.json")
try:
    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load schema: {e}")

def validate_metadata(payload: dict) -> dict:
    """
    Validates metadata payload against schema.json.
    Returns {"valid": True} or {"valid": False, "error": "..."}.
    """
    try:
        validate(instance=payload, schema=schema)
        return {"valid": True}
    except ValidationError as e:
        return {"valid": False, "error": e.message}

