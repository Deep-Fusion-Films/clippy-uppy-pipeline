import pytest
from unittest.mock import patch, MagicMock
from google.cloud import storage

# Assume your upload function looks like this:
# def upload_to_gcs(bucket_name: str, source_path: str, destination_blob: str) -> bool

from gcs_upload import upload_to_gcs  # adjust import path as needed

@pytest.fixture
def sample_args():
    return {
        "bucket_name": "test-bucket",
        "source_path": "/tmp/test_file.txt",
        "destination_blob": "uploads/test_file.txt"
    }

@patch("gcs_upload.storage.Client")
def test_upload_success(mock_client, sample_args):
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_client.return_value.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    result = upload_to_gcs(**sample_args)
    assert result is True
    mock_blob.upload_from_filename.assert_called_once_with(sample_args["source_path"])

@patch("gcs_upload.storage.Client")
def test_upload_failure(mock_client, sample_args):
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.upload_from_filename.side_effect = Exception("Upload failed")
    mock_client.return_value.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    result = upload_to_gcs(**sample_args)
    assert result is False

@patch("gcs_upload.storage.Client")
def test_invalid_bucket(mock_client, sample_args):
    mock_client.return_value.bucket.side_effect = Exception("Bucket not found")

    result = upload_to_gcs(**sample_args)
    assert result is False

