"""Utilities for downloading objects from Cloudflare R2.

R2 exposes an S3-compatible API, so we use boto3 (already a project
dependency) pointed at the account's R2 endpoint rather than the
Cloudflare SDK, which does not handle object transfers.

Required environment variables:

    CLOUDFLARE_ACCOUNT_ID   Account that owns the R2 bucket (used to build
                            the endpoint URL).
    R2_ACCESS_KEY_ID        R2 API token access key id.
    R2_SECRET_ACCESS_KEY    R2 API token secret.

Optional:

    R2_BUCKET               Default bucket, so callers can omit it.
    P2BP_TMP_DIR            Base directory for temporary downloads. Defaults
                            to "<system temp>/p2bp-tmp" (i.e. /tmp/p2bp-tmp
                            on the EC2 Linux host).
"""

import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import boto3
from botocore.client import BaseClient
from botocore.config import Config


# --- Configuration -----------------------------------------------------------


def require_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        print(f"Missing required environment variable: {name}")
        sys.exit(1)

    return value


def r2_endpoint_url(account_id: str) -> str:
    return f"https://{account_id}.r2.cloudflarestorage.com"


def create_r2_client() -> BaseClient:
    """Build an S3 client configured for this account's R2 endpoint."""

    account_id = require_env("CLOUDFLARE_ACCOUNT_ID")
    access_key_id = require_env("R2_ACCESS_KEY_ID")
    secret_access_key = require_env("R2_SECRET_ACCESS_KEY")

    return boto3.client(
        "s3",
        endpoint_url=r2_endpoint_url(account_id),
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        # R2 only supports the "auto" region and SigV4.
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


def default_bucket() -> str:
    return require_env("R2_BUCKET")


# --- Temporary download directories ------------------------------------------
#
# Layout:  <base>/r2-downloads/<unique>/<file>
#
# e.g.     /tmp/p2bp-tmp/r2-downloads/tmp8f3k2a/scan.las
#
# The per-download "<unique>" segment is created with tempfile.mkdtemp, which
# atomically guarantees a fresh, collision-free directory (no manual GUID +
# mkdir race). Grouping everything under one stable base keeps the tree tidy
# and easy to wipe wholesale if a worker dies mid-job.


def tmp_base_dir() -> Path:
    base = os.getenv("P2BP_TMP_DIR") or os.path.join(tempfile.gettempdir(), "p2bp-tmp")
    return Path(base)


def r2_downloads_dir() -> Path:
    path = tmp_base_dir() / "r2-downloads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def new_download_dir(label: Optional[str] = None) -> Path:
    """Create and return a unique directory for one download/job.

    `label` (e.g. a job or project id) is used as a readable prefix so the
    directory is easy to spot while debugging; uniqueness is still guaranteed
    by mkdtemp regardless of the label.
    """

    prefix = f"{_sanitize(label)}-" if label else ""
    return Path(tempfile.mkdtemp(prefix=prefix, dir=r2_downloads_dir()))


@contextmanager
def temp_download_dir(label: Optional[str] = None) -> Iterator[Path]:
    """Context manager yielding a unique download dir, removed on exit."""

    path = new_download_dir(label)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _sanitize(label: str) -> str:
    """Keep labels filesystem-safe and short."""

    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    return safe[:64] or "job"


# --- Downloads ---------------------------------------------------------------


def download_object(
    client: BaseClient,
    bucket: str,
    key: str,
    dest: Path,
) -> Path:
    """Download a single R2 object to `dest` (a full file path).

    Parent directories are created as needed. Returns the destination path.
    """

    dest.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(dest))
    return dest


def download_to_dir(
    client: BaseClient,
    bucket: str,
    key: str,
    directory: Path,
    filename: Optional[str] = None,
) -> Path:
    """Download an object into `directory`, returning the written file path.

    By default the filename is the last path segment of the key.
    """

    name = filename or key_basename(key)
    return download_object(client, bucket, key, directory / name)


def download_to_temp(
    client: BaseClient,
    bucket: str,
    key: str,
    label: Optional[str] = None,
) -> Path:
    """Download an object into a fresh, unique temp directory.

    The caller owns the returned file and its parent directory and is
    responsible for cleanup. Use `temp_download_dir` instead when you want
    automatic removal.
    """

    return download_to_dir(client, bucket, key, new_download_dir(label))


def key_basename(key: str) -> str:
    """Last segment of an R2 object key (keys use forward slashes)."""

    return key.rstrip("/").rsplit("/", 1)[-1] or "download"
