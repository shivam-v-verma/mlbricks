"""Project-wide pytest fixtures."""

from typing import Any

import boto3
import pytest
from moto import mock_aws


@pytest.fixture()
def aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set dummy AWS credentials so boto3 does not look for real ones."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture()
def s3_bucket(aws_env: None) -> Any:
    """Yield a mock S3 client with an empty test-bucket created."""
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="test-bucket")
        yield client
