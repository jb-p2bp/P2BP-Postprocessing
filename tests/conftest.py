"""Shared test fixtures.

Keeps tests deterministic and isolated from the developer's environment: every
R2-related env var is cleared, and the temp-download base is redirected under
pytest's per-test `tmp_path` so nothing touches the real system temp directory.
"""

import os

import pytest

R2_ENV_VARS = (
    "P2BP_TMP_DIR",
    "R2_BUCKET",
    "CLOUDFLARE_ACCOUNT_ID",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    """Clear ambient R2 env vars and point the temp base at tmp_path."""

    for var in R2_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    # tmp_path exists, so "p2bp" is a fresh dir we create and own.
    monkeypatch.setenv("P2BP_TMP_DIR", str(tmp_path / "p2bp"))


class FakeClient:
    """Stand-in for a boto3 S3 client; records calls and writes a stub file."""

    def __init__(self, payload: bytes = b"data"):
        self.payload = payload
        self.calls: list[tuple[str, str, str]] = []

    def download_file(self, bucket: str, key: str, dest: str) -> None:
        self.calls.append((bucket, key, dest))
        with open(dest, "wb") as handle:
            handle.write(self.payload)


@pytest.fixture
def fake_client():
    return FakeClient()
