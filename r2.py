"""Utilities for uploading objects to and downloading objects from Cloudflare R2.

R2 exposes an S3-compatible API, so we use boto3 (already a project
dependency) pointed at the account's R2 endpoint rather than the
Cloudflare SDK, which does not handle object transfers.

Required environment variables:

    CLOUDFLARE_ACCOUNT_ID   Account that owns the R2 bucket (used to build
                            the endpoint URL).
    R2_ACCESS_KEY_ID        R2 API token access key id.
    R2_SECRET_ACCESS_KEY    R2 API token secret.

Optional:

    R2_BUCKET               Default bucket. Required unless every call passes an
                            explicit `bucket=` argument: when omitted, the
                            transfer helpers fall back to this var and raise
                            ConfigError if it too is unset.
    P2BP_TMP_DIR            Base directory for temporary downloads. Must be an
                            absolute path. Defaults to "<system temp>/p2bp-tmp"
                            (i.e. /tmp/p2bp-tmp on the EC2 Linux host).

Exceptions raised:

    ConfigError             Missing/invalid configuration: an unset required env
                            var, a non-hex account id, or a relative/parentless
                            P2BP_TMP_DIR.
    InsecureTempDirError    A temp directory failed its safety checks (symlink,
                            foreign owner, or not a directory) -- a possible
                            hijack.
    FileExistsError         A download target already exists, or an upload would
                            replace an existing R2 object, and overwrite is
                            False.
    ValueError              A key/filename yields no safe local name, or an R2
                            object key is degenerate (empty, "." or "..").
"""

import logging
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import boto3
from botocore.client import BaseClient
from botocore.config import Config
from botocore.exceptions import ClientError

from config import ConfigError, require_env

logger = logging.getLogger(__name__)

__all__ = [
    "InsecureTempDirError",
    "create_r2_client",
    "default_bucket",
    "download_object",
    "download_to_dir",
    "download_to_temp",
    "new_download_dir",
    "temp_download_dir",
    "upload_from_dir",
    "upload_object",
]


# --- Configuration -----------------------------------------------------------


_ACCOUNT_ID_RE = re.compile(r"[0-9a-f]{32}")


def _r2_endpoint_url(account_id: str) -> str:
    """Return the R2 S3-compatible endpoint URL for `account_id`.

    Raises ConfigError if the id is not a 32-char lowercase hex string.
    """

    # Account ids are interpolated into the endpoint host, so reject anything
    # that isn't a Cloudflare account id (32 lowercase hex chars) to avoid a
    # malformed value redirecting the host/path.
    if not _ACCOUNT_ID_RE.fullmatch(account_id):
        raise ConfigError(
            f"CLOUDFLARE_ACCOUNT_ID must be 32 lowercase hex characters, "
            f"got {account_id!r}"
        )
    return f"https://{account_id}.r2.cloudflarestorage.com"


def create_r2_client() -> BaseClient:
    """Build an S3 client configured for this account's R2 endpoint."""

    account_id = require_env("CLOUDFLARE_ACCOUNT_ID")
    access_key_id = require_env("R2_ACCESS_KEY_ID")
    secret_access_key = require_env("R2_SECRET_ACCESS_KEY")

    return boto3.client(
        "s3",
        endpoint_url=_r2_endpoint_url(account_id),
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        # R2 only supports the "auto" region and SigV4.
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


def default_bucket() -> str:
    """Return the bucket from R2_BUCKET, or raise ConfigError if unset."""

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


def _tmp_base_dir() -> Path:
    override = os.getenv("P2BP_TMP_DIR")
    if override:
        # A relative override would be resolved against the process's cwd at
        # call time, so the "stable base" would silently move if the worker
        # ever chdir'd. Require an absolute path rather than guess an anchor.
        if not os.path.isabs(override):
            raise ConfigError(
                f"P2BP_TMP_DIR must be an absolute path, got {override!r}"
            )
        return Path(override)

    return Path(tempfile.gettempdir()) / "p2bp-tmp"


def _r2_downloads_dir() -> Path:
    # Build the tree one level at a time so each predictable component is
    # created/verified as a private directory we own (see _ensure_private_dir).
    #
    # O_NOFOLLOW only guards the final component of each open, so this remains
    # safe only if the *ancestors* of the base are trusted -- i.e. owned by us
    # or under a sticky directory (the default /tmp is sticky). Pointing
    # P2BP_TMP_DIR under a world/group-writable, non-sticky directory voids that
    # assumption; document this requirement for operators.
    base = _ensure_private_dir(_tmp_base_dir())
    return _ensure_private_dir(base / "r2-downloads")


class InsecureTempDirError(RuntimeError):
    """Raised when a download temp directory fails its safety checks.

    Signals a symlink, foreign-owned, or non-directory path where we expected a
    private directory we own -- distinct from a ConfigError (bad configuration)
    so callers can handle a potential hijack specifically.
    """


def _ensure_private_dir(path: Path) -> Path:
    """Create `path` as a 0700 directory and verify we own it.

    The download base lives at a predictable location (e.g. /tmp/p2bp-tmp)
    under world-writable /tmp. Plain mkdir(exist_ok=True) would silently adopt
    a pre-existing symlink or another user's directory, letting a local
    attacker redirect every download.

    To avoid trusting directories we did not create and check, we do NOT create
    intermediate parents: the parent must already exist (callers build the tree
    one verified level at a time). On POSIX we then open the directory with
    O_NOFOLLOW so a symlink swap fails outright, and verify ownership / tighten
    permissions through the file descriptor so nothing is re-resolved between
    check and use. On non-POSIX platforms (local Windows dev) the ownership
    model differs, so we just ensure the directory exists.
    """

    try:
        path.mkdir(mode=0o700, exist_ok=True)  # no parents=True: see docstring
    except FileNotFoundError as error:
        raise ConfigError(
            f"parent directory of {path} does not exist; create it before use"
        ) from error

    if not hasattr(os, "getuid"):  # non-POSIX (e.g. Windows dev machines)
        return path

    # O_NOFOLLOW makes the open fail if the final component is a symlink, and
    # operating on the fd (fstat/fchmod) closes the check-then-use race.
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY)
    except OSError as error:
        raise InsecureTempDirError(
            f"refusing to use temp dir {path}: not a regular directory ({error})"
        ) from error

    try:
        info = os.fstat(fd)
        if info.st_uid != os.getuid():
            raise InsecureTempDirError(
                f"refusing to use temp dir {path}: owned by uid {info.st_uid}, "
                f"not us"
            )
        if stat.S_IMODE(info.st_mode) != 0o700:
            os.fchmod(fd, 0o700)
    finally:
        os.close(fd)

    return path


def new_download_dir(label: str | None = None) -> Path:
    """Create and return a unique directory for one download/job.

    `label` (e.g. a job or project id) is used as a readable prefix so the
    directory is easy to spot while debugging; uniqueness is still guaranteed
    by mkdtemp regardless of the label.
    """

    prefix = f"{_sanitize(label)}-" if label else ""
    return Path(tempfile.mkdtemp(prefix=prefix, dir=_r2_downloads_dir()))


@contextmanager
def temp_download_dir(label: str | None = None) -> Iterator[Path]:
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
    key: str,
    dest: Path,
    bucket: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Download a single R2 object to `dest` (a full file path).

    `bucket` defaults to the R2_BUCKET environment variable when omitted.
    Parent directories are created as needed. Returns the destination path.

    Refuses to clobber an existing file unless `overwrite=True`, so a key
    collision (two objects mapping to the same local name) surfaces as an
    error instead of silently destroying the earlier download.
    """

    bucket = bucket or default_bucket()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and dest.exists():
        raise FileExistsError(
            f"{dest} already exists; pass overwrite=True to replace it"
        )
    logger.info("Downloading r2://%s/%s -> %s", bucket, key, dest)
    client.download_file(bucket, key, str(dest))
    if logger.isEnabledFor(logging.DEBUG):
        # Read the size defensively: the download already succeeded, so a
        # concurrent removal of dest must not turn that success into an error.
        try:
            size = dest.stat().st_size
        except OSError:
            size = -1
        logger.debug("Downloaded %s (%d bytes)", dest, size)
    return dest


def download_to_dir(
    client: BaseClient,
    key: str,
    directory: Path,
    bucket: str | None = None,
    filename: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Download an object into `directory`, returning the written file path.

    `bucket` defaults to the R2_BUCKET environment variable when omitted. By
    default the filename is the last path segment of the key. Two keys sharing
    that segment would map to the same local file; the download refuses to
    overwrite an existing one unless `overwrite=True`.
    """

    dest = _safe_join(directory, filename or key)
    return download_object(client, key, dest, bucket=bucket, overwrite=overwrite)


def download_to_temp(
    client: BaseClient,
    key: str,
    bucket: str | None = None,
    label: str | None = None,
) -> Path:
    """Download an object into a fresh, unique temp directory.

    `bucket` defaults to the R2_BUCKET environment variable when omitted. The
    caller owns the returned file and its parent directory and is responsible
    for cleanup. Use `temp_download_dir` instead when you want automatic
    removal.
    """

    return download_to_dir(client, key, new_download_dir(label), bucket=bucket)


def _safe_join(directory: Path, name: str) -> Path:
    """Join `name` onto `directory`, refusing to escape it.

    `name` may be a full R2 object key (e.g. "scans/2024/a.las") or an explicit
    filename. Only its last path segment is used, so absolute paths and ".."
    traversal cannot write outside `directory`. Raises ValueError if no usable
    filename can be derived (empty key, "." or ".."), rather than silently
    inventing one.
    """

    # Normalise Windows separators so basename works the same on every OS, then
    # collapse to the final path component.
    base = os.path.basename(name.replace("\\", "/"))

    if base in ("", ".", ".."):
        raise ValueError(f"cannot derive a safe filename from {name!r}")

    dest = directory / base

    # Defense in depth: confirm the resolved path stays inside `directory`.
    if not dest.resolve().is_relative_to(directory.resolve()):
        raise ValueError(f"refusing to write {name!r} outside {directory}")

    return dest


# --- Uploads -----------------------------------------------------------------


def upload_object(
    client: BaseClient,
    source: Path,
    key: str,
    bucket: str | None = None,
    overwrite: bool = False,
) -> str:
    """Upload a single local file to R2 under `key`, returning the key.

    `bucket` defaults to the R2_BUCKET environment variable when omitted. R2
    objects are silently overwritten by default, so this refuses to replace an
    existing object unless `overwrite=True` -- a key collision (two files
    mapping to the same object) then surfaces as an error instead of quietly
    destroying the earlier upload. The existence check is a HEAD request; like
    the download-side clobber check it is racy against concurrent writers, which
    we accept for the same reason.
    """

    bucket = bucket or default_bucket()
    key = _safe_key(key)
    if not overwrite and _object_exists(client, bucket, key):
        raise FileExistsError(
            f"r2://{bucket}/{key} already exists; pass overwrite=True to replace it"
        )
    logger.info("Uploading %s -> r2://%s/%s", source, bucket, key)
    client.upload_file(str(source), bucket, key)
    if logger.isEnabledFor(logging.DEBUG):
        # Mirror the download-side debug log: the upload already succeeded, so a
        # concurrent removal of source must not turn that success into an error.
        try:
            size = source.stat().st_size
        except OSError:
            size = -1
        logger.debug("Uploaded %s (%d bytes)", source, size)
    return key


def upload_from_dir(
    client: BaseClient,
    directory: Path,
    filename: str,
    key: str,
    bucket: str | None = None,
    overwrite: bool = False,
) -> str:
    """Upload `directory`/`filename` to R2 under `key`, returning the key.

    `bucket` defaults to the R2_BUCKET environment variable when omitted. `key`
    is always required -- unlike `download_to_dir` (which derives the local name
    from the key), the remote object key is never inferred from the filename.
    `_safe_join` is reused so an absolute path or ".." traversal in `filename`
    cannot read outside `directory`.
    """

    source = _safe_join(directory, filename)
    return upload_object(client, source, key, bucket=bucket, overwrite=overwrite)


def _object_exists(client: BaseClient, bucket: str, key: str) -> bool:
    """Return True if R2 has an object at `bucket`/`key`.

    Uses HEAD, so it never transfers the object body. A 404/NotFound is the
    "absent" answer; any other ClientError is re-raised so real failures
    (permissions, network, bucket missing) are not misread as "safe to upload".
    """

    try:
        client.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        code = error.response.get("Error", {}).get("Code", "")
        if code in ("404", "NotFound"):
            return False
        raise
    return True


def _safe_key(key: str) -> str:
    """Validate `key` as an R2 object key, refusing degenerate values.

    R2 keys are opaque strings and may contain "/" (the conventional prefix
    separator) and most other characters, but an empty key -- or "."/".." as
    the whole key -- is almost certainly a mistake. Raise ValueError rather
    than quietly creating an oddly-named object, matching `_safe_join`'s
    treatment of degenerate local names.
    """

    if key in ("", ".", ".."):
        raise ValueError(f"cannot use {key!r} as an R2 object key")
    return key
