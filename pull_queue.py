"""EC2-hosted worker that drains a Cloudflare queue and self-stops when idle.

The instance is powered on by a Cloudflare worker that first enqueues a message,
so the worker assumes that being up means there should be work. It polls the
queue, processes one message at a time, and stops the EC2 instance when the
queue stays empty past the idle limit, the max runtime is hit, or failures pile
up. See ensure_shutdown for the EC2-API-then-OS-poweroff shutdown strategy.

Runtime prerequisites:
  * Environment variables: CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_QUEUE_ID,
    CLOUDFLARE_API_TOKEN, AWS_REGION (loaded from the environment / a .env file).
  * Process supervisor: run under a supervisor (e.g. systemd). The worker exits
    on graceful signals and expects the supervisor NOT to relaunch it once the
    instance is stopping (it absorbs stop signals during the EC2 stop window).
  * IAM role permissions: ec2:StopInstances and ec2:DescribeInstanceAttribute.
  * Instance metadata: IMDSv2 must be reachable at IMDS_BASE_URL.
  * Shutdown behavior: the instance's InstanceInitiatedShutdownBehavior must be
    "stop" (verified at startup) so the OS-poweroff fallback cannot terminate it.
  * sudo: a passwordless rule allowing `sudo -n systemctl poweroff`, scoped to
    exactly that command.
"""

import json
import math
import os
import sys
import time
import signal
import random
import logging
import subprocess
import shutil
import stat
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal, Mapping, NoReturn, Optional, Protocol, TypedDict

import boto3
import requests
from botocore.exceptions import ClientError
from cloudflare import Cloudflare
from dotenv import load_dotenv

from config import ConfigError, require_env
from mesh_jobs import MeshGenerateJob, MeshRefineJob, parse_mesh_job_message
from r2 import (
    create_r2_client,
    default_bucket,
    download_object,
    temp_download_dir,
    upload_object,
)
from scanproject_merger import (
    RegistrationResult,
    export_merged_cloud_outputs,
    merge_scan_projects,
)
from scanproject_merger.registration import parameters as transform_parameters
from stitching import (
    STITCHED_MESH_FILENAME,
    STITCHED_MESH_METADATA_FILENAME,
    StitchingParams,
    stitch_point_cloud,
)


# =========================
# CONFIG
# =========================

load_dotenv()

# Seconds with no completed work before the instance shuts itself down. The
# idle clock starts at boot (see last_completed_work_time in main), so an
# instance that boots to an empty queue stops itself after this window. This is
# intentional: a Cloudflare worker enqueues a message *before* powering this
# instance on, so an instance being up implies work should be waiting. If none
# is found, the instance is unexpected and should shut down promptly.
IDLE_LIMIT_SECONDS = int(os.getenv("IDLE_LIMIT_SECONDS", "60"))
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "15"))
MAX_CONSECUTIVE_FAILURES = int(os.getenv("MAX_CONSECUTIVE_FAILURES", "10"))
MAX_BACKOFF_SECONDS = int(os.getenv("MAX_BACKOFF_SECONDS", "300"))
PROCESSING_RETRY_DELAY_SECONDS = int(
    os.getenv("PROCESSING_RETRY_DELAY_SECONDS", "60")
)
# 12h consumer contract. The default process runtime stays lower so a
# max-runtime job still has time to upload outputs and ack before the queue
# lease can expire. The Cloudflare worker fails jobs with no terminal status
# 24h after creation (`meshJobTimeoutMs` in
# `p2bp-cf-worker/src/routes/api/mesh.jobs.reconciliation.ts`); raising this
# past ~24h would make legitimately long runs get misreported as timed out.
MESH_JOB_CONSUMER_RUNTIME_CAP_SECONDS = 43_200
DEFAULT_MAX_RUNTIME_SECONDS = MESH_JOB_CONSUMER_RUNTIME_CAP_SECONDS - 10 * 60
MAX_RUNTIME_SECONDS = int(
    os.getenv("MAX_RUNTIME_SECONDS", str(DEFAULT_MAX_RUNTIME_SECONDS))
)
SHUTDOWN_RETRY_SECONDS = int(os.getenv("SHUTDOWN_RETRY_SECONDS", "30"))

# How long a pulled message stays invisible before redelivery. It must outlast
# the longest a single job can hold the lease before it is acked -- messages are
# acked only after the full merge completes (see handle_message), so a timeout
# shorter than the run plus output upload/ack buffer lets the lease expire
# mid-merge, the ack then targets an expired lease, and the message is
# redelivered and reprocessed (wasted compute, and a long job may never ack).
CLOUDFLARE_MAX_VISIBILITY_TIMEOUT_MS = (
    MESH_JOB_CONSUMER_RUNTIME_CAP_SECONDS * 1000
)
VISIBILITY_TIMEOUT_MS = int(
    os.getenv("VISIBILITY_TIMEOUT_MS", str(CLOUDFLARE_MAX_VISIBILITY_TIMEOUT_MS))
)

# Output voxel sizes. The full project cloud keeps scanproject_merger's
# production 2 cm grid by default; the preview writes a coarser 10 cm grid.
MERGED_POINT_CLOUD_DEDUPLICATE_VOXEL = float(
    os.getenv("MERGED_POINT_CLOUD_DEDUPLICATE_VOXEL", "0.02")
)
PREVIEW_POINT_CLOUD_DEDUPLICATE_VOXEL = float(
    os.getenv("PREVIEW_POINT_CLOUD_DEDUPLICATE_VOXEL", "0.10")
)
STITCHING_VOXEL_SIZE = float(os.getenv("STITCHING_VOXEL_SIZE", "0.05"))
STITCHING_POISSON_DEPTH = int(os.getenv("STITCHING_POISSON_DEPTH", "9"))
STITCHING_DENSITY_QUANTILE = float(
    os.getenv("STITCHING_DENSITY_QUANTILE", "0.02")
)
STITCHING_TARGET_TRIANGLES = int(
    os.getenv("STITCHING_TARGET_TRIANGLES", "1000000")
)
SCANPROJECT_ZIP_MAX_UNCOMPRESSED_BYTES = int(
    os.getenv("SCANPROJECT_ZIP_MAX_UNCOMPRESSED_BYTES", str(4 * 1024 * 1024 * 1024))
)

# EC2 Instance Metadata Service (IMDSv2). These are fixed infrastructure facts,
# not per-deploy tunables.
IMDS_BASE_URL = "http://169.254.169.254"
IMDS_TOKEN_TTL_SECONDS = "21600"  # 6h; only bounds a single metadata fetch
IMDS_REQUEST_TIMEOUT_SECONDS = 2

running = True


# =========================
# LOGGING
# =========================

logger = logging.getLogger("queue-worker")


# =========================
# SIGNAL HANDLING
# =========================

def handle_signal(signum: int, frame: Any) -> None:
    global running
    logger.warning(f"Received signal {signum}. Shutting down gracefully...")
    running = False


def configure_runtime() -> None:
    """Apply process-global side effects.

    Kept out of module import so the module can be imported (e.g. by tests)
    without reconfiguring logging or replacing the process signal handlers.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)


def validate_runtime_contract() -> None:
    """Fail fast when runtime knobs drift from the Worker contract."""

    if PROCESSING_RETRY_DELAY_SECONDS < 0:
        raise ConfigError("PROCESSING_RETRY_DELAY_SECONDS cannot be negative.")

    if MAX_RUNTIME_SECONDS > MESH_JOB_CONSUMER_RUNTIME_CAP_SECONDS:
        raise ConfigError(
            "MAX_RUNTIME_SECONDS cannot exceed "
            f"{MESH_JOB_CONSUMER_RUNTIME_CAP_SECONDS}; update the Worker "
            "meshJobConsumerRuntimeCapMs contract before raising it."
        )

    if VISIBILITY_TIMEOUT_MS > CLOUDFLARE_MAX_VISIBILITY_TIMEOUT_MS:
        raise ConfigError(
            "VISIBILITY_TIMEOUT_MS cannot exceed Cloudflare Queues' "
            f"{CLOUDFLARE_MAX_VISIBILITY_TIMEOUT_MS}ms maximum."
        )

    if VISIBILITY_TIMEOUT_MS < MAX_RUNTIME_SECONDS * 1000:
        raise ConfigError(
            "VISIBILITY_TIMEOUT_MS must be at least MAX_RUNTIME_SECONDS * 1000 "
            "so a job can ack before its queue lease expires."
        )

    if VISIBILITY_TIMEOUT_MS == MAX_RUNTIME_SECONDS * 1000:
        raise ConfigError(
            "VISIBILITY_TIMEOUT_MS must leave headroom above MAX_RUNTIME_SECONDS "
            "for output uploads and message ack."
        )


# =========================
# HELPERS
# =========================

def parse_body(body: Any) -> Any:
    if isinstance(body, str):
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return body
    return body


def _message_body_summary_for_log(message: Any) -> str:
    body = parse_body(getattr(message, "body", None))
    if not isinstance(body, dict):
        return f"body_type={type(body).__name__}"

    parts = ["body_type=dict"]
    if "type" in body:
        parts.append(f"type={body['type']!r}")
    if "version" in body:
        parts.append(f"version={body['version']!r}")

    zone_scan_keys = body.get("zoneScanObjectKeys")
    if isinstance(zone_scan_keys, list):
        parts.append(f"zone_scan_key_count={len(zone_scan_keys)}")

    return " ".join(parts)


def log_pulled_messages(messages: list[Any]) -> None:
    if not messages:
        return

    logger.info("Pulled %d message(s) from queue.", len(messages))
    for index, message in enumerate(messages, start=1):
        logger.info(
            "Pulled message %d/%d lease_id=%s %s",
            index,
            len(messages),
            getattr(message, "lease_id", None),
            _message_body_summary_for_log(message),
        )


def get_instance_id(max_attempts: int = 3) -> Optional[str]:
    for attempt in range(1, max_attempts + 1):
        try:
            token_response = requests.put(
                f"{IMDS_BASE_URL}/latest/api/token",
                headers={
                    "X-aws-ec2-metadata-token-ttl-seconds": IMDS_TOKEN_TTL_SECONDS
                },
                timeout=IMDS_REQUEST_TIMEOUT_SECONDS,
            )
            token_response.raise_for_status()
            token = token_response.text.strip()

            if not token:
                raise requests.RequestException("IMDS returned an empty token")

            response = requests.get(
                f"{IMDS_BASE_URL}/latest/meta-data/instance-id",
                headers={"X-aws-ec2-metadata-token": token},
                timeout=IMDS_REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()

            instance_id = response.text.strip()
            if not instance_id:
                raise requests.RequestException("IMDS returned an empty instance ID")

            return instance_id

        except requests.RequestException as e:
            logger.warning(
                f"IMDS request failed ({attempt}/{max_attempts}): {e}"
            )

            if attempt < max_attempts:
                time.sleep(2 ** (attempt - 1))

    return None


def power_off_os() -> None:
    # IMPORTANT: This is an OS-initiated shutdown. AWS applies the instance's
    # InstanceInitiatedShutdownBehavior attribute here, so this only *stops* the
    # instance when that attribute is "stop". If it were "terminate", this would
    # DESTROY the instance instead of stopping it. verify_shutdown_behavior()
    # is called at startup to guarantee the attribute is "stop" before the
    # worker ever reaches this fallback.
    subprocess.run(
        ["sudo", "-n", "systemctl", "poweroff"],
        check=True,
        timeout=10,
    )


def verify_shutdown_behavior(instance_id: Optional[str], region: str) -> None:
    """Refuse to run unless OS-initiated shutdown will *stop* the instance.

    The power_off_os() fallback triggers an OS-initiated shutdown, which AWS
    resolves using the instance's InstanceInitiatedShutdownBehavior attribute.
    If that attribute is "terminate", the fallback would destroy the instance
    instead of stopping it, leaving nothing for the launcher to start again.
    Fail fast at startup rather than risk that during a later shutdown.

    Requires the ec2:DescribeInstanceAttribute IAM permission. This is
    fail-closed: if the attribute cannot be positively confirmed to be "stop"
    (unknown instance ID, missing permission, transient API error, or any other
    value), the worker exits rather than risk destroying the instance.
    """
    if not instance_id:
        logger.critical(
            "Cannot verify InstanceInitiatedShutdownBehavior without an "
            "instance ID. Refusing to run."
        )
        sys.exit(1)

    try:
        ec2 = boto3.client("ec2", region_name=region)
        attribute = ec2.describe_instance_attribute(
            InstanceId=instance_id,
            Attribute="instanceInitiatedShutdownBehavior",
        )
        behavior = attribute["InstanceInitiatedShutdownBehavior"]["Value"]
    except Exception:
        logger.critical(
            "Could not verify InstanceInitiatedShutdownBehavior (check the "
            "ec2:DescribeInstanceAttribute permission). Refusing to run.",
            exc_info=True,
        )
        sys.exit(1)

    if behavior != "stop":
        logger.critical(
            "InstanceInitiatedShutdownBehavior is %r, not 'stop'. The OS "
            "poweroff fallback would destroy this instance. Refusing to run; "
            "set the shutdown behavior to 'stop' on the launch template.",
            behavior,
        )
        sys.exit(1)

    logger.info("Verified InstanceInitiatedShutdownBehavior=stop.")


def ensure_shutdown(
    reason: str,
    instance_id: Optional[str],
    region: str,
) -> NoReturn:
    logger.critical(f"Entering shutdown mode: {reason}")

    # Once we have committed to shutting down, ignore SIGTERM/SIGINT for the
    # entire shutdown window -- the EC2 API attempt(s), the OS-poweroff
    # fallback, and the retry sleeps. Absorbing a supervisor's stop signal
    # prevents it from observing an exit and launching a replacement worker on
    # an instance that is already going down. The OS still terminates us at the
    # end of its own shutdown sequence via SIGKILL, which cannot be ignored.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    shutdown_instance_id = instance_id

    while True:
        try:
            if not shutdown_instance_id:
                shutdown_instance_id = get_instance_id()

            if not shutdown_instance_id:
                raise RuntimeError("Unable to determine EC2 instance ID")

            logger.warning(
                f"Stopping EC2 instance {shutdown_instance_id} via EC2 API..."
            )

            ec2 = boto3.client("ec2", region_name=region)
            ec2.stop_instances(InstanceIds=[shutdown_instance_id])

            logger.warning(
                "EC2 stop request accepted. Waiting for the instance to stop."
            )

            # Stay alive (signals already ignored above) until the OS shutdown
            # sequence terminates us.
            while True:
                time.sleep(60)

        except Exception:
            logger.exception("EC2 API shutdown failed")

        try:
            logger.critical("Attempting operating-system poweroff...")
            power_off_os()
            sys.exit(0)

        except Exception:
            logger.exception("Operating-system poweroff failed")

        logger.critical(
            f"All shutdown methods failed. Retrying in "
            f"{SHUTDOWN_RETRY_SECONDS}s."
        )
        time.sleep(SHUTDOWN_RETRY_SECONDS)


# =========================
# CORE LOGIC
# =========================

def compute_backoff(consecutive_failures: int) -> float:
    """Exponential backoff (base POLL_INTERVAL_SECONDS) with +/-20% jitter, capped.

    Jitter spreads retries so repeated failures don't align into a thundering
    herd. The result is capped at MAX_BACKOFF_SECONDS.
    """
    return min(
        POLL_INTERVAL_SECONDS
        * (2 ** (consecutive_failures - 1))
        * random.uniform(0.8, 1.2),
        MAX_BACKOFF_SECONDS,
    )


class FailureTracker:
    """Counts consecutive failures per stage (e.g. "poll", "processing").

    Stages are counted independently: a success in one stage resets only that
    stage's counter. This keeps the "reset on success, increment on failure,
    give up at the limit" invariant in one place instead of spread across the
    main loop as separate counters.
    """

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._counts: dict[str, int] = {}

    def reset(self, stage: str) -> None:
        self._counts[stage] = 0

    def record(self, stage: str) -> int:
        """Increment the stage's counter and return the new count."""
        self._counts[stage] = self._counts.get(stage, 0) + 1
        return self._counts[stage]

    def limit_reached(self, stage: str) -> bool:
        return self._counts.get(stage, 0) >= self._limit


def _job_output_key(job: MeshGenerateJob, filename: str) -> str:
    """Versioned per-job output key.

    Mirrors `buildMeshJobObjectKey` in
    `p2bp-cf-worker/src/routes/api/mesh.jobs.keys.ts`; keep the two in sync.
    """
    return (
        f"organizations/{job.organizationId}/projects/{job.projectId}/"
        f"mesh-jobs/{job.jobId}/{filename}"
    )


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with a `Z` suffix.

    The worker's status schema (`meshJobStatusFileSchema` in
    `p2bp-cf-worker/src/lib/mesh/job-contract.ts`) accepts both `Z` and
    `+HH:MM` offsets; `Z` is kept as the canonical form written here.
    """
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# The exact states the Cloudflare worker's status schema accepts
# (`meshJobStatusFileSchema` in `p2bp-cf-worker/src/lib/mesh/job-contract.ts`).
# Anything else is silently treated as "no evidence" by the worker and would
# surface 24h later as a bogus timeout, so an unknown state must fail loudly
# here instead. `Literal` alone is not runtime-enforced, hence the guard in
# write_job_status.
MeshJobStatusState = Literal["running", "completed", "failed"]
MESH_JOB_STATUS_STATES: tuple[MeshJobStatusState, ...] = (
    "running",
    "completed",
    "failed",
)
MESH_JOB_STATUS_FILENAME = "status.json"

MESH_JOB_OUTPUT_FILENAMES: Mapping[str, str] = MappingProxyType(
    {
        "pointCloud": "merged-point-cloud.laz",
        "pointCloudBin": "merged-point-cloud.bin",
        "pointCloudPreview": "merged-point-cloud.preview.laz",
        "pointCloudPreviewBin": "merged-point-cloud.preview.bin",
    }
)


class MeshJobOutputKeys(TypedDict):
    pointCloud: str
    pointCloudBin: str
    pointCloudPreview: str
    pointCloudPreviewBin: str


class MeshJobR2Client(Protocol):
    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentType: str | None = None,
    ) -> dict[str, Any]: ...

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]: ...

    def download_file(self, bucket: str, key: str, dest: str) -> None: ...

    def upload_file(self, source: str, bucket: str, key: str) -> None: ...

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]: ...


def write_job_status(
    r2_client: MeshJobR2Client,
    job: MeshGenerateJob,
    state: MeshJobStatusState,
    started_at: str,
    completed_at: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Write the job's `status.json` to R2 (last write wins).

    The Cloudflare worker reconciles the `mesh_jobs` D1 row from this object
    on read; the schema is pinned in
    `p2bp-cf-worker/src/lib/mesh/job-contract.ts`. Unknown extra fields are
    ignored by the worker, so additions here are non-breaking.
    """
    if state not in MESH_JOB_STATUS_STATES:
        raise ValueError(f"invalid mesh job status state: {state!r}")

    status = {
        "state": state,
        "jobId": job.jobId,
        "startedAt": started_at,
        "completedAt": completed_at,
        "error": error,
    }
    r2_client.put_object(
        Bucket=default_bucket(),
        Key=_job_output_key(job, MESH_JOB_STATUS_FILENAME),
        Body=json.dumps(status).encode("utf-8"),
        ContentType="application/json",
    )


def _is_not_found_error(error: ClientError) -> bool:
    code = error.response.get("Error", {}).get("Code", "")
    return code in ("404", "NoSuchKey", "NotFound")


def mesh_job_status_is_completed(
    r2_client: MeshJobR2Client, job: MeshGenerateJob
) -> bool:
    """Return True when this job's status.json already says completed."""

    try:
        response = r2_client.get_object(
            Bucket=default_bucket(),
            Key=_job_output_key(job, MESH_JOB_STATUS_FILENAME),
        )
    except ClientError as error:
        if _is_not_found_error(error):
            return False
        raise

    try:
        status = json.loads(response["Body"].read().decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        return False

    if not isinstance(status, dict):
        return False
    if status.get("state") != "completed":
        return False

    status_job_id = status.get("jobId")
    return status_job_id in (None, job.jobId)


def _public_error_summary(error: BaseException) -> str:
    """Allowlisted public error for status.json: exception class name only.

    Raw exception text routinely embeds local filesystem paths, bucket
    endpoints, and R2 object keys, and the Cloudflare worker surfaces
    status.json's `error` to every project viewer. The full detail is already
    captured in this worker's own logs (the main loop logs the re-raised
    exception with logger.exception), so the public record carries only the
    exception class -- a category, never interpolated message text.
    """
    return (
        f"{type(error).__name__} while processing the mesh job; "
        f"details are in the consumer logs"
    )


def _write_job_status_failed(
    r2_client: MeshJobR2Client, job: MeshGenerateJob, started_at: str, error: str
) -> None:
    """Best-effort failed-status write: never mask the original exception."""
    try:
        write_job_status(
            r2_client,
            job,
            state="failed",
            started_at=started_at,
            completed_at=_utc_now_iso(),
            error=error,
        )
    except Exception:
        logger.exception("Could not write failed job status for %s", job.jobId)


def _safe_label(value: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in value)
    return safe[:80] or "scan"


def _scanproject_stem_from_key(key: str) -> str:
    name = os.path.basename(key.replace("\\", "/"))
    if not name:
        raise ValueError(f"cannot derive a scan name from object key {key!r}")
    stem = name[:-4] if name.lower().endswith(".zip") else name
    return _safe_label(stem)


def _zip_member_target(destination: Path, member_name: str) -> Path:
    normalized = member_name.replace("\\", "/")
    if normalized.startswith("/"):
        raise ValueError(f"zip member uses an absolute path: {member_name!r}")

    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"zip member escapes the scan package: {member_name!r}")

    target = destination.joinpath(*path.parts)
    if not target.resolve().is_relative_to(destination.resolve()):
        raise ValueError(f"zip member escapes the scan package: {member_name!r}")
    return target


def extract_scanproject_zip(archive: Path, destination: Path) -> Path:
    """Extract a zipped scanproject body into a ``.scanproject`` directory."""

    destination.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(archive) as zip_file:
            members = zip_file.infolist()
            if not members:
                raise ValueError(f"scanproject archive is empty: {archive}")
            total_uncompressed_size = sum(member.file_size for member in members)
            if total_uncompressed_size > SCANPROJECT_ZIP_MAX_UNCOMPRESSED_BYTES:
                raise ValueError(
                    f"scanproject archive expands to {total_uncompressed_size} "
                    "bytes, which exceeds the configured limit of "
                    f"{SCANPROJECT_ZIP_MAX_UNCOMPRESSED_BYTES} bytes"
                )

            for member in members:
                target = _zip_member_target(destination, member.filename)
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValueError(
                        f"zip member is a symbolic link: {member.filename!r}"
                    )
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zip_file.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise

    manifest = destination / "manifest.json"
    if not manifest.is_file():
        shutil.rmtree(destination, ignore_errors=True)
        raise ValueError(
            f"scanproject archive {archive} did not contain manifest.json "
            f"at its root"
        )

    return destination


def process_generate_job(job: MeshGenerateJob) -> None:
    r2_client = create_r2_client()
    output_keys: MeshJobOutputKeys = {
        "pointCloud": _job_output_key(job, MESH_JOB_OUTPUT_FILENAMES["pointCloud"]),
        "pointCloudBin": _job_output_key(
            job, MESH_JOB_OUTPUT_FILENAMES["pointCloudBin"]
        ),
        "pointCloudPreview": _job_output_key(
            job, MESH_JOB_OUTPUT_FILENAMES["pointCloudPreview"]
        ),
        "pointCloudPreviewBin": _job_output_key(
            job, MESH_JOB_OUTPUT_FILENAMES["pointCloudPreviewBin"]
        ),
    }

    if mesh_job_status_is_completed(r2_client, job):
        logger.info(
            "Skipping mesh.generate job %s because status.json is already completed.",
            job.jobId,
        )
        return

    started_at = _utc_now_iso()
    write_job_status(r2_client, job, state="running", started_at=started_at)

    try:
        _run_generate_job(r2_client, job, output_keys)
    except BaseException as error:
        # Record the failure for the worker's status reconciliation, then
        # re-raise so the message stays un-acked and gets redelivered. A later
        # successful redelivery overwrites this with a completed status.
        _write_job_status_failed(
            r2_client, job, started_at=started_at, error=_public_error_summary(error)
        )
        raise

    write_job_status(
        r2_client,
        job,
        state="completed",
        started_at=started_at,
        completed_at=_utc_now_iso(),
    )


def _log_registration_transforms(result: RegistrationResult) -> None:
    """Log how registration moved each scan, so alignments are auditable.

    The correction is what registration applied on top of each scan's own
    georeference (the anchor scan's correction is identity by construction);
    the full 4x4 matrices are in the workspace's .registration.json, which is
    discarded with the workspace, so this is the durable record.
    """
    logger.info(
        "Registration corrections for %d scan(s), anchor=%s:",
        len(result.scans),
        result.scans[0].project.identifier,
    )
    for index, (scan, correction) in enumerate(
        zip(result.scans, result.correction_transforms)
    ):
        yaw, tx, ty, tz = transform_parameters(correction)
        logger.info(
            "  scan %03d %s: yaw=%+.3f deg, translation=(%+.3f, %+.3f, %+.3f) m",
            index,
            scan.project.identifier,
            math.degrees(yaw),
            tx,
            ty,
            tz,
        )
    for edge in result.edges:
        logger.info(
            "  edge %03d->%03d: rmse=%.3f m, overlap=%.2f, correspondences=%d, init=%s",
            edge.moving,
            edge.fixed,
            edge.rmse,
            edge.overlap_ratio,
            edge.correspondence_count,
            edge.initialization,
        )
    for rejection in result.rejected_edges:
        logger.warning(
            "  edge %03d->%03d rejected (%s): rmse=%.3f m, overlap=%.2f",
            rejection.edge.moving,
            rejection.edge.fixed,
            rejection.reason,
            rejection.edge.rmse,
            rejection.edge.overlap_ratio,
        )


def _run_generate_job(
    r2_client: MeshJobR2Client,
    job: MeshGenerateJob,
    output_keys: MeshJobOutputKeys,
) -> None:
    with temp_download_dir(f"{job.organizationId}-{job.projectId}") as workspace:
        archives_dir = workspace / "archives"
        scanprojects_dir = workspace / "scanprojects"
        outputs_dir = workspace / "outputs"
        archives_dir.mkdir()
        scanprojects_dir.mkdir()
        outputs_dir.mkdir()

        scanproject_paths: list[Path] = []
        for index, object_key in enumerate(job.zoneScanObjectKeys):
            stem = _scanproject_stem_from_key(object_key)
            archive = archives_dir / f"{index:03d}-{stem}.zip"
            scanproject_dir = scanprojects_dir / f"{index:03d}-{stem}.scanproject"

            download_object(r2_client, object_key, archive)
            extract_scanproject_zip(archive, scanproject_dir)
            scanproject_paths.append(scanproject_dir)

        full_output = outputs_dir / MESH_JOB_OUTPUT_FILENAMES["pointCloud"]
        full_bin_output = outputs_dir / MESH_JOB_OUTPUT_FILENAMES["pointCloudBin"]
        preview_output = outputs_dir / MESH_JOB_OUTPUT_FILENAMES["pointCloudPreview"]
        preview_bin_output = (
            outputs_dir / MESH_JOB_OUTPUT_FILENAMES["pointCloudPreviewBin"]
        )
        stitched_mesh_output = outputs_dir / STITCHED_MESH_FILENAME
        stitched_mesh_metadata_output = (
            outputs_dir / STITCHED_MESH_METADATA_FILENAME
        )

        logger.info(
            "Merging %d scanproject archive(s) for organization=%s project=%s",
            len(scanproject_paths),
            job.organizationId,
            job.projectId,
        )
        outputs = merge_scan_projects(
            scanproject_paths,
            full_output,
            bin_output=full_bin_output,
            deduplicate_voxel=MERGED_POINT_CLOUD_DEDUPLICATE_VOXEL,
            export_minimum_confidence=0,
        )
        _log_registration_transforms(outputs.result)
        preview_points = export_merged_cloud_outputs(
            outputs.result,
            laz_output=preview_output,
            bin_output=preview_bin_output,
            minimum_confidence=0,
            deduplicate_voxel=PREVIEW_POINT_CLOUD_DEDUPLICATE_VOXEL,
        )
        full_point_count = outputs.point_count
        # Registration retains every source point. Release it before Open3D
        # loads the merged LAZ so stitching does not hold both representations.
        del outputs

        logger.info(
            "Stitching merged cloud with voxel=%.3f m, Poisson depth=%d, "
            "target triangles=%d",
            STITCHING_VOXEL_SIZE,
            STITCHING_POISSON_DEPTH,
            STITCHING_TARGET_TRIANGLES,
        )
        stitched = stitch_point_cloud(
            full_output,
            stitched_mesh_output,
            metadata_output=stitched_mesh_metadata_output,
            params=StitchingParams(
                voxel_size=STITCHING_VOXEL_SIZE,
                poisson_depth=STITCHING_POISSON_DEPTH,
                density_quantile=STITCHING_DENSITY_QUANTILE,
                target_triangles=STITCHING_TARGET_TRIANGLES,
            ),
        )

        logger.info(
            "Uploading merged cloud (%d points), preview (%d points), and "
            "stitched mesh (%d vertices, %d triangles)",
            full_point_count,
            preview_points,
            stitched.vertices,
            stitched.triangles,
        )
        upload_object(r2_client, full_output, output_keys["pointCloud"], overwrite=True)
        upload_object(
            r2_client,
            full_bin_output,
            output_keys["pointCloudBin"],
            overwrite=True,
        )
        upload_object(
            r2_client,
            preview_output,
            output_keys["pointCloudPreview"],
            overwrite=True,
        )
        upload_object(
            r2_client,
            preview_bin_output,
            output_keys["pointCloudPreviewBin"],
            overwrite=True,
        )
        upload_object(
            r2_client,
            stitched.mesh,
            _job_output_key(job, STITCHED_MESH_FILENAME),
            overwrite=True,
        )
        upload_object(
            r2_client,
            stitched.metadata,
            _job_output_key(job, STITCHED_MESH_METADATA_FILENAME),
            overwrite=True,
        )


def process_message(body: Any) -> None:
    """
    MUST raise Exception on failure.

    If this function does NOT raise, the message will be ACKed
    and considered permanently completed.
    """

    # Avoid logging the full payload because it may contain sensitive data.
    logger.info("Processing message with body type: %s", type(body).__name__)
    job = parse_mesh_job_message(body)

    if isinstance(job, MeshGenerateJob):
        process_generate_job(job)
        return

    if isinstance(job, MeshRefineJob):
        raise NotImplementedError("mesh.refine jobs are not supported yet")

    raise RuntimeError(f"unsupported message type: {type(job).__name__}")


def ack_message(
    client: Cloudflare, queue_id: str, account_id: str, lease_id: str
) -> None:
    client.queues.messages.ack(
        queue_id,
        account_id=account_id,
        acks=[{"lease_id": lease_id}],
        retries=[],
    )


def retry_message(
    client: Cloudflare,
    queue_id: str,
    account_id: str,
    lease_id: str,
    delay_seconds: int = PROCESSING_RETRY_DELAY_SECONDS,
) -> None:
    client.queues.messages.ack(
        queue_id,
        account_id=account_id,
        acks=[],
        retries=[{"lease_id": lease_id, "delay_seconds": delay_seconds}],
    )


def pull_one(client: Cloudflare, queue_id: str, account_id: str) -> list[Any]:
    """Pull a single message from the queue, returning a (possibly empty) list."""
    pull_response = client.queues.messages.pull(
        queue_id,
        account_id=account_id,
        batch_size=1,
        visibility_timeout_ms=VISIBILITY_TIMEOUT_MS,
    )
    # `or []` guards against the attribute being present but None, which the
    # SDK may return for an empty pull; a bare getattr default only covers a
    # missing attribute.
    messages = getattr(pull_response, "messages", None) or []
    log_pulled_messages(messages)
    return messages


def handle_message(
    client: Cloudflare, queue_id: str, account_id: str, message: Any
) -> None:
    """Process a single message, acking success and retrying failure promptly.

    Raises on any failure so the caller can record a processing failure and
    apply its local backoff/shutdown policy.
    """
    body = parse_body(message.body)
    lease_id = getattr(message, "lease_id", None)

    if not lease_id:
        raise RuntimeError("Pulled message is missing lease_id")

    try:
        process_message(body)
    except Exception:
        logger.warning(
            "Returning failed message to queue for retry in %ds...",
            PROCESSING_RETRY_DELAY_SECONDS,
        )
        retry_message(client, queue_id, account_id, lease_id)
        raise

    # ACK ONLY ON SUCCESS
    logger.info("Acknowledging message...")
    ack_message(client, queue_id, account_id, lease_id)
    logger.info("Message acknowledged.")


def handle_empty_poll(
    client: Cloudflare,
    queue_id: str,
    account_id: str,
    idle_for: float,
    instance_id: Optional[str],
    region: str,
) -> list[Any]:
    """Decide what to do when a poll returns no messages.

    Returns an empty list when the worker is not yet idle long enough and the
    caller should sleep and keep polling. Once the idle limit is reached, a
    final confirming poll is performed: if it is also empty the instance is shut
    down (this does not return); otherwise the newly arrived messages are
    returned for processing.
    """
    logger.info(f"No messages. Idle since last work: {idle_for:.0f}s")

    if idle_for < IDLE_LIMIT_SECONDS:
        return []

    # Final confirming poll before stopping. This NARROWS -- but cannot close --
    # the window between observing an empty queue and the asynchronous
    # stop_instances call: a producer can still enqueue after this poll returns
    # empty but before the instance stops, and because the instance still
    # appears "on" the launcher won't start a worker to pick the message up.
    # Fully closing this cross-system TOCTOU requires an idempotent launcher
    # that starts the instance on every enqueue. If the confirming poll is also
    # empty we stop; otherwise we process the new message.
    logger.warning("Idle limit reached. Performing a final confirming poll...")
    messages = pull_one(client, queue_id, account_id)

    if not messages:
        logger.warning("Queue confirmed empty. Shutting down EC2...")
        ensure_shutdown("idle limit reached", instance_id, region)

    logger.info(
        "Message arrived during idle confirmation; "
        "processing instead of shutting down."
    )
    return messages


# =========================
# MAIN LOOP
# =========================

def main() -> None:
    configure_runtime()

    try:
        validate_runtime_contract()
        account_id = require_env("CLOUDFLARE_ACCOUNT_ID")
        queue_id = require_env("CLOUDFLARE_QUEUE_ID")
        api_token = require_env("CLOUDFLARE_API_TOKEN")
        region = require_env("AWS_REGION")
    except ConfigError as e:
        logger.error(e)
        sys.exit(1)

    instance_id = get_instance_id()
    if instance_id:
        logger.info(f"Cached EC2 instance ID: {instance_id}")
    else:
        logger.warning(
            "Could not determine the EC2 instance ID during startup. "
            "Shutdown-behavior verification will refuse to run without it."
        )

    verify_shutdown_behavior(instance_id, region)

    client = Cloudflare(api_token=api_token)

    failures = FailureTracker(MAX_CONSECUTIVE_FAILURES)
    start_time = time.monotonic()

    last_completed_work_time = time.monotonic()

    logger.info("Queue worker started")
    logger.info(
        f"Configuration: idle_limit={IDLE_LIMIT_SECONDS}s, "
        f"max_runtime={MAX_RUNTIME_SECONDS}s, "
        f"poll_interval={POLL_INTERVAL_SECONDS}s"
    )

    while running:

        if time.monotonic() - start_time > MAX_RUNTIME_SECONDS:
            logger.warning("Max runtime reached. Shutting down.")
            ensure_shutdown("maximum runtime reached", instance_id, region)

        failure_stage = "poll"

        try:
            logger.debug("Polling queue...")

            messages = pull_one(client, queue_id, account_id)
            failures.reset("poll")

            if not messages:
                idle_for = time.monotonic() - last_completed_work_time
                messages = handle_empty_poll(
                    client, queue_id, account_id, idle_for, instance_id, region
                )
                if not messages:
                    time.sleep(POLL_INTERVAL_SECONDS)
                    continue

            failure_stage = "processing"
            handle_message(client, queue_id, account_id, messages[0])

            last_completed_work_time = time.monotonic()
            failures.reset("processing")

        except Exception as e:
            active_failures = failures.record(failure_stage)
            backoff = compute_backoff(active_failures)

            logger.exception(
                f"{failure_stage.capitalize()} error "
                f"({active_failures}/{MAX_CONSECUTIVE_FAILURES}): {e}"
            )

            if failures.limit_reached(failure_stage):
                logger.error(f"Too many consecutive {failure_stage} failures.")
                ensure_shutdown(
                    f"too many consecutive {failure_stage} failures",
                    instance_id,
                    region,
                )

            time.sleep(backoff)

    logger.info("Worker exited cleanly.")


if __name__ == "__main__":
    main()
