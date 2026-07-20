"""Tests for the queue worker entry point.

The worker is an EC2-hosted long-runner with many external dependencies
(IMDS, boto3, Cloudflare); these tests cover the thin config surface that
is practical to exercise without standing up the surrounding infrastructure.
"""

import json
import logging
from pathlib import Path
from types import SimpleNamespace
import zipfile

import config
import numpy as np
import pull_queue
import pytest


def fake_registration_result(identifier="zone-a"):
    """Just enough RegistrationResult shape for the worker's transform logging."""
    return SimpleNamespace(
        scans=[SimpleNamespace(project=SimpleNamespace(identifier=identifier))],
        correction_transforms=[np.eye(4)],
        edges=[],
        rejected_edges=[],
    )

REQUIRED_WORKER_ENV = (
    "CLOUDFLARE_ACCOUNT_ID",
    "CLOUDFLARE_QUEUE_ID",
    "CLOUDFLARE_API_TOKEN",
    "AWS_REGION",
)


def set_valid_runtime_contract(monkeypatch):
    monkeypatch.setattr(pull_queue, "MAX_RUNTIME_SECONDS", 60)
    monkeypatch.setattr(pull_queue, "VISIBILITY_TIMEOUT_MS", 61_000)
    monkeypatch.setattr(pull_queue, "PROCESSING_RETRY_DELAY_SECONDS", 60)


def test_pull_queue_uses_shared_config_helper():
    """The worker must reuse the shared config helpers, not a local duplicate."""
    assert pull_queue.require_env is config.require_env
    assert pull_queue.ConfigError is config.ConfigError


def test_main_exits_when_required_config_missing(monkeypatch):
    """main() must report and exit(1) when a required env var is missing.

    The shared helper raises ConfigError; the entry point turns that into a
    logged error plus a non-zero exit (see config.py's module docstring).
    configure_runtime is stubbed so the test never installs real signal
    handlers or reconfigures logging.
    """
    monkeypatch.setattr(pull_queue, "configure_runtime", lambda: None)
    monkeypatch.setattr(pull_queue, "validate_runtime_contract", lambda: None)
    for var in REQUIRED_WORKER_ENV:
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(SystemExit) as exc:
        pull_queue.main()

    assert exc.value.code == 1


def test_main_exits_when_runtime_contract_invalid(monkeypatch):
    monkeypatch.setattr(pull_queue, "configure_runtime", lambda: None)
    monkeypatch.setattr(
        pull_queue,
        "validate_runtime_contract",
        lambda: (_ for _ in ()).throw(config.ConfigError("bad runtime config")),
    )

    with pytest.raises(SystemExit) as exc:
        pull_queue.main()

    assert exc.value.code == 1


def make_zip(path: Path, files: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return path


def zip_bytes(tmp_path: Path, files: dict[str, bytes]) -> bytes:
    path = make_zip(tmp_path / "scan.zip", files)
    return path.read_bytes()


def test_pull_one_logs_pulled_messages(caplog):
    message = SimpleNamespace(
        lease_id="lease_123",
        body=json.dumps(
            {
                "type": "mesh.generate",
                "version": 2,
                "organizationId": "org_123",
                "projectId": "proj_456",
                "zoneScanObjectKeys": ["organizations/org_123/private.zip"],
            }
        ),
    )
    response = SimpleNamespace(messages=[message])
    client = SimpleNamespace(
        queues=SimpleNamespace(
            messages=SimpleNamespace(pull=lambda *args, **kwargs: response)
        )
    )

    with caplog.at_level(logging.INFO, logger="queue-worker"):
        assert pull_queue.pull_one(client, "queue_123", "account_123") == [message]

    assert "Pulled 1 message(s) from queue." in caplog.text
    assert "lease_id=lease_123" in caplog.text
    assert "type='mesh.generate'" in caplog.text
    assert "version=2" in caplog.text
    assert "zone_scan_key_count=1" in caplog.text
    assert "proj_456" not in caplog.text
    assert "organizations/org_123/private.zip" not in caplog.text


def test_pull_one_leases_long_enough_to_outlast_a_job(monkeypatch):
    monkeypatch.setattr(pull_queue, "MAX_RUNTIME_SECONDS", 60)
    monkeypatch.setattr(pull_queue, "VISIBILITY_TIMEOUT_MS", 61_000)

    captured: dict[str, object] = {}

    def fake_pull(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(messages=[])

    client = SimpleNamespace(
        queues=SimpleNamespace(messages=SimpleNamespace(pull=fake_pull))
    )

    pull_queue.pull_one(client, "queue_123", "account_123")

    # A message is acked only after the full merge completes, so the lease must
    # outlast the longest a single job can run -- otherwise the ack races an
    # expired lease and the message is redelivered and reprocessed.
    assert (
        captured["visibility_timeout_ms"] >= pull_queue.MAX_RUNTIME_SECONDS * 1000
    )
    # ...and stay within Cloudflare Queues' 12h maximum.
    assert (
        captured["visibility_timeout_ms"]
        <= pull_queue.CLOUDFLARE_MAX_VISIBILITY_TIMEOUT_MS
    )


def test_runtime_contract_rejects_runtime_above_worker_cap(monkeypatch):
    set_valid_runtime_contract(monkeypatch)
    monkeypatch.setattr(
        pull_queue,
        "MAX_RUNTIME_SECONDS",
        pull_queue.MESH_JOB_CONSUMER_RUNTIME_CAP_SECONDS + 1,
    )

    with pytest.raises(config.ConfigError, match="MAX_RUNTIME_SECONDS"):
        pull_queue.validate_runtime_contract()


def test_runtime_contract_rejects_short_visibility_timeout(monkeypatch):
    set_valid_runtime_contract(monkeypatch)
    monkeypatch.setattr(pull_queue, "VISIBILITY_TIMEOUT_MS", 59_999)

    with pytest.raises(config.ConfigError, match="VISIBILITY_TIMEOUT_MS"):
        pull_queue.validate_runtime_contract()


def test_runtime_contract_rejects_visibility_timeout_without_headroom(
    monkeypatch,
):
    set_valid_runtime_contract(monkeypatch)
    monkeypatch.setattr(pull_queue, "VISIBILITY_TIMEOUT_MS", 60_000)

    with pytest.raises(config.ConfigError, match="headroom"):
        pull_queue.validate_runtime_contract()


def test_runtime_contract_rejects_visibility_timeout_above_cloudflare_cap(
    monkeypatch,
):
    set_valid_runtime_contract(monkeypatch)
    monkeypatch.setattr(
        pull_queue,
        "VISIBILITY_TIMEOUT_MS",
        pull_queue.CLOUDFLARE_MAX_VISIBILITY_TIMEOUT_MS + 1,
    )

    with pytest.raises(config.ConfigError, match="VISIBILITY_TIMEOUT_MS"):
        pull_queue.validate_runtime_contract()


def test_runtime_contract_rejects_negative_processing_retry_delay(monkeypatch):
    set_valid_runtime_contract(monkeypatch)
    monkeypatch.setattr(pull_queue, "PROCESSING_RETRY_DELAY_SECONDS", -1)

    with pytest.raises(config.ConfigError, match="PROCESSING_RETRY_DELAY_SECONDS"):
        pull_queue.validate_runtime_contract()


def test_handle_message_retries_processing_failures(monkeypatch):
    captured: dict[str, object] = {}

    def fake_ack(queue_id, *, account_id, acks, retries):
        captured.update(
            {
                "account_id": account_id,
                "acks": acks,
                "queue_id": queue_id,
                "retries": retries,
            }
        )

    client = SimpleNamespace(
        queues=SimpleNamespace(messages=SimpleNamespace(ack=fake_ack))
    )
    message = SimpleNamespace(lease_id="lease_123", body={"type": "bad"})

    def broken_process_message(body):
        assert body == {"type": "bad"}
        raise RuntimeError("merge failed")

    monkeypatch.setattr(pull_queue, "process_message", broken_process_message)
    monkeypatch.setattr(pull_queue, "PROCESSING_RETRY_DELAY_SECONDS", 60)

    with pytest.raises(RuntimeError, match="merge failed"):
        pull_queue.handle_message(client, "queue_123", "account_123", message)

    assert captured == {
        "account_id": "account_123",
        "acks": [],
        "queue_id": "queue_123",
        "retries": [{"lease_id": "lease_123", "delay_seconds": 60}],
    }


def test_process_generate_job_downloads_merges_and_uploads_outputs(
    fake_client,
    monkeypatch,
    tmp_path,
):
    fake_client.payload = zip_bytes(tmp_path, {"manifest.json": b"{}"})
    monkeypatch.setenv("R2_BUCKET", "env-bucket")
    monkeypatch.setattr(pull_queue, "create_r2_client", lambda: fake_client)

    captured: dict[str, object] = {}

    def fake_merge_scan_projects(inputs, output, **kwargs):
        captured["inputs"] = list(inputs)
        captured["output"] = output
        captured["bin_output"] = kwargs.pop("bin_output")
        captured["merge_kwargs"] = kwargs
        captured["manifest_exists"] = (Path(inputs[0]) / "manifest.json").is_file()
        output.write_bytes(b"full")
        captured["bin_output"].write_bytes(b"full-bin")
        registration = fake_registration_result()
        captured["registration"] = registration
        return SimpleNamespace(point_count=10, result=registration)

    def fake_export_merged_cloud_outputs(result, **kwargs):
        captured["preview_result"] = result
        laz_output = kwargs.pop("laz_output")
        bin_output = kwargs.pop("bin_output")
        captured["preview_output"] = laz_output
        captured["preview_bin_output"] = bin_output
        captured["preview_kwargs"] = kwargs
        laz_output.write_bytes(b"preview")
        bin_output.write_bytes(b"preview-bin")
        return 3

    def fake_stitch_point_cloud(input_cloud, output_mesh, **kwargs):
        captured["stitch_input"] = input_cloud
        captured["stitch_output"] = output_mesh
        captured["stitch_metadata_output"] = kwargs.pop("metadata_output")
        captured["stitch_params"] = kwargs.pop("params")
        assert kwargs == {}
        output_mesh.write_bytes(b"glTF-mesh")
        captured["stitch_metadata_output"].write_bytes(b"{}")
        return SimpleNamespace(
            mesh=output_mesh,
            metadata=captured["stitch_metadata_output"],
            vertices=20,
            triangles=30,
        )

    monkeypatch.setattr(pull_queue, "merge_scan_projects", fake_merge_scan_projects)
    monkeypatch.setattr(
        pull_queue,
        "export_merged_cloud_outputs",
        fake_export_merged_cloud_outputs,
    )
    monkeypatch.setattr(
        pull_queue,
        "stitch_point_cloud",
        fake_stitch_point_cloud,
    )

    pull_queue.process_message(
        {
            "type": "mesh.generate",
            "version": 2,
            "jobId": "job_789",
            "organizationId": "org_123",
            "projectId": "proj_456",
            "zoneScanObjectKeys": ["uploads/zone-a.zip"],
        }
    )

    assert fake_client.calls == [
        (
            "env-bucket",
            "uploads/zone-a.zip",
            str(
                Path(captured["inputs"][0]).parent.parent
                / "archives"
                / "000-zone-a.zip"
            ),
        )
    ]
    assert Path(captured["inputs"][0]).name == "000-zone-a.scanproject"
    assert captured["manifest_exists"] is True
    assert captured["output"].name == "merged-point-cloud.laz"
    assert captured["merge_kwargs"] == {
        "deduplicate_voxel": pull_queue.MERGED_POINT_CLOUD_DEDUPLICATE_VOXEL,
        "export_minimum_confidence": 0,
    }
    assert captured["preview_result"] is captured["registration"]
    assert captured["bin_output"].name == "merged-point-cloud.bin"
    assert captured["preview_output"].name == "merged-point-cloud.preview.laz"
    assert captured["preview_bin_output"].name == "merged-point-cloud.preview.bin"
    assert captured["preview_kwargs"] == {
        "minimum_confidence": 0,
        "deduplicate_voxel": pull_queue.PREVIEW_POINT_CLOUD_DEDUPLICATE_VOXEL,
    }
    assert captured["stitch_input"] == captured["output"]
    assert Path(captured["stitch_output"]).name == "stitched-mesh.glb"
    assert (
        Path(captured["stitch_metadata_output"]).name
        == "stitched-mesh.metadata.json"
    )
    assert captured["stitch_params"] == pull_queue.StitchingParams(
        voxel_size=pull_queue.STITCHING_VOXEL_SIZE,
        poisson_depth=pull_queue.STITCHING_POISSON_DEPTH,
        density_quantile=pull_queue.STITCHING_DENSITY_QUANTILE,
        target_triangles=pull_queue.STITCHING_TARGET_TRIANGLES,
        read_chunk_points=pull_queue.STITCHING_READ_CHUNK_POINTS,
        maximum_stitching_points=pull_queue.STITCHING_MAXIMUM_POINTS,
    )
    job_prefix = "organizations/org_123/projects/proj_456/mesh-jobs/job_789"
    assert fake_client.upload_calls == [
        (
            str(Path(captured["output"])),
            "env-bucket",
            f"{job_prefix}/merged-point-cloud.laz",
        ),
        (
            str(Path(captured["output"]).with_name("merged-point-cloud.bin")),
            "env-bucket",
            f"{job_prefix}/merged-point-cloud.bin",
        ),
        (
            str(Path(captured["output"]).with_name("merged-point-cloud.preview.laz")),
            "env-bucket",
            f"{job_prefix}/merged-point-cloud.preview.laz",
        ),
        (
            str(Path(captured["output"]).with_name("merged-point-cloud.preview.bin")),
            "env-bucket",
            f"{job_prefix}/merged-point-cloud.preview.bin",
        ),
        (
            str(Path(captured["output"]).with_name("stitched-mesh.glb")),
            "env-bucket",
            f"{job_prefix}/stitched-mesh.glb",
        ),
        (
            str(
                Path(captured["output"]).with_name(
                    "stitched-mesh.metadata.json"
                )
            ),
            "env-bucket",
            f"{job_prefix}/stitched-mesh.metadata.json",
        ),
    ]

    # The worker reconciles job status from these writes: running before any
    # work, completed after every output has been uploaded.
    assert [(bucket, key) for bucket, key, _, _ in fake_client.put_calls] == [
        ("env-bucket", f"{job_prefix}/status.json"),
        ("env-bucket", f"{job_prefix}/status.json"),
    ]
    running_status = json.loads(fake_client.put_calls[0][2])
    completed_status = json.loads(fake_client.put_calls[1][2])
    assert running_status["state"] == "running"
    assert running_status["jobId"] == "job_789"
    assert running_status["startedAt"].endswith("Z")
    assert completed_status["state"] == "completed"
    assert completed_status["startedAt"] == running_status["startedAt"]
    assert completed_status["completedAt"].endswith("Z")
    assert completed_status["error"] is None
    assert fake_client.put_calls[0][3] == "application/json"


def test_process_generate_job_writes_failed_status_and_reraises(
    fake_client,
    monkeypatch,
    tmp_path,
):
    fake_client.payload = zip_bytes(tmp_path, {"manifest.json": b"{}"})
    monkeypatch.setenv("R2_BUCKET", "env-bucket")
    monkeypatch.setattr(pull_queue, "create_r2_client", lambda: fake_client)

    def broken_merge(*args, **kwargs):
        raise RuntimeError("registration diverged")

    monkeypatch.setattr(pull_queue, "merge_scan_projects", broken_merge)

    with pytest.raises(RuntimeError, match="registration diverged"):
        pull_queue.process_message(
            {
                "type": "mesh.generate",
                "version": 2,
                "jobId": "job_789",
                "organizationId": "org_123",
                "projectId": "proj_456",
                "zoneScanObjectKeys": ["uploads/zone-a.zip"],
            }
        )

    job_prefix = "organizations/org_123/projects/proj_456/mesh-jobs/job_789"
    states = [json.loads(body)["state"] for _, key, body, _ in fake_client.put_calls]
    assert [key for _, key, _, _ in fake_client.put_calls] == [
        f"{job_prefix}/status.json",
        f"{job_prefix}/status.json",
    ]
    assert states == ["running", "failed"]
    failed_status = json.loads(fake_client.put_calls[1][2])
    # The public error is an allowlisted category (exception class only);
    # raw exception text may embed paths/endpoints and must never leak into
    # status.json, which the worker surfaces to every project viewer.
    assert failed_status["error"] == (
        "RuntimeError while processing the mesh job; "
        "details are in the consumer logs"
    )
    assert "registration diverged" not in failed_status["error"]
    assert failed_status["completedAt"].endswith("Z")


def test_process_generate_job_skips_completed_redelivery(
    fake_client,
    monkeypatch,
):
    monkeypatch.setenv("R2_BUCKET", "env-bucket")
    monkeypatch.setattr(pull_queue, "create_r2_client", lambda: fake_client)
    job_prefix = "organizations/org_123/projects/proj_456/mesh-jobs/job_789"
    fake_client.put_object(
        Bucket="env-bucket",
        Key=f"{job_prefix}/status.json",
        Body=json.dumps(
            {
                "state": "completed",
                "jobId": "job_789",
                "startedAt": "2026-07-04T00:00:00Z",
                "completedAt": "2026-07-04T00:05:00Z",
                "error": None,
            }
        ).encode("utf-8"),
        ContentType="application/json",
    )
    fake_client.put_calls.clear()

    pull_queue.process_message(
        {
            "type": "mesh.generate",
            "version": 2,
            "jobId": "job_789",
            "organizationId": "org_123",
            "projectId": "proj_456",
            "zoneScanObjectKeys": ["uploads/zone-a.zip"],
        }
    )

    assert fake_client.calls == []
    assert fake_client.upload_calls == []
    assert fake_client.put_calls == []


def test_mesh_job_status_is_completed_rejects_mismatched_job_id(
    fake_client,
    monkeypatch,
):
    monkeypatch.setenv("R2_BUCKET", "env-bucket")
    job = pull_queue.parse_mesh_job_message(
        {
            "type": "mesh.generate",
            "version": 2,
            "jobId": "job_789",
            "organizationId": "org_123",
            "projectId": "proj_456",
            "zoneScanObjectKeys": ["uploads/zone-a.zip"],
        }
    )
    fake_client.put_object(
        Bucket="env-bucket",
        Key="organizations/org_123/projects/proj_456/mesh-jobs/job_789/status.json",
        Body=json.dumps({"state": "completed", "jobId": "other-job"}).encode("utf-8"),
        ContentType="application/json",
    )

    assert pull_queue.mesh_job_status_is_completed(fake_client, job) is False


def test_mesh_job_status_is_completed_ignores_malformed_status(
    fake_client,
    monkeypatch,
):
    monkeypatch.setenv("R2_BUCKET", "env-bucket")
    job = pull_queue.parse_mesh_job_message(
        {
            "type": "mesh.generate",
            "version": 2,
            "jobId": "job_789",
            "organizationId": "org_123",
            "projectId": "proj_456",
            "zoneScanObjectKeys": ["uploads/zone-a.zip"],
        }
    )
    fake_client.put_object(
        Bucket="env-bucket",
        Key="organizations/org_123/projects/proj_456/mesh-jobs/job_789/status.json",
        Body=b"not json",
        ContentType="application/json",
    )

    assert pull_queue.mesh_job_status_is_completed(fake_client, job) is False


def test_extract_scanproject_zip_rejects_path_traversal(tmp_path):
    archive = make_zip(tmp_path / "bad.zip", {"../escape.txt": b"nope"})

    with pytest.raises(ValueError, match="escapes"):
        pull_queue.extract_scanproject_zip(archive, tmp_path / "out.scanproject")

    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path / "out.scanproject").exists()


def test_extract_scanproject_zip_requires_root_manifest(tmp_path):
    archive = make_zip(tmp_path / "wrapped.zip", {"folder/manifest.json": b"{}"})

    with pytest.raises(ValueError, match="manifest.json"):
        pull_queue.extract_scanproject_zip(archive, tmp_path / "out.scanproject")

    assert not (tmp_path / "out.scanproject").exists()


def test_extract_scanproject_zip_rejects_oversized_expansion(monkeypatch, tmp_path):
    archive = make_zip(
        tmp_path / "huge.zip",
        {
            "manifest.json": b"{}",
            "point-cloud.bin": b"x" * 11,
        },
    )
    monkeypatch.setattr(pull_queue, "SCANPROJECT_ZIP_MAX_UNCOMPRESSED_BYTES", 10)

    with pytest.raises(ValueError, match="exceeds the configured limit"):
        pull_queue.extract_scanproject_zip(archive, tmp_path / "out.scanproject")

    assert not (tmp_path / "out.scanproject").exists()


def test_process_message_rejects_refine_jobs():
    with pytest.raises(NotImplementedError):
        pull_queue.process_message(
            {
                "type": "mesh.refine",
                "version": 1,
                "organizationId": "org",
                "projectId": "proj",
            }
        )


def test_write_job_status_rejects_unknown_state(fake_client, monkeypatch):
    monkeypatch.setenv("R2_BUCKET", "env-bucket")
    job = pull_queue.parse_mesh_job_message(
        {
            "type": "mesh.generate",
            "version": 2,
            "jobId": "job_789",
            "organizationId": "org_123",
            "projectId": "proj_456",
            "zoneScanObjectKeys": ["uploads/zone-a.zip"],
        }
    )

    with pytest.raises(ValueError, match="invalid mesh job status state"):
        pull_queue.write_job_status(
            fake_client,
            job,
            state="complete",  # type: ignore[arg-type]
            started_at="2026-07-04T00:00:00Z",
        )

    assert fake_client.put_calls == []


def test_status_states_match_worker_contract():
    """Pin against `meshJobStatusFileSchema` in
    `p2bp-cf-worker/src/lib/mesh/job-contract.ts`; update both together."""
    assert pull_queue.MESH_JOB_STATUS_STATES == ("running", "completed", "failed")
    assert pull_queue.MESH_JOB_STATUS_FILENAME == "status.json"


def test_worker_stitching_defaults_match_library_defaults():
    defaults = pull_queue.DEFAULT_STITCHING_PARAMS

    assert pull_queue.STITCHING_VOXEL_SIZE == defaults.voxel_size
    assert pull_queue.STITCHING_POISSON_DEPTH == defaults.poisson_depth
    assert pull_queue.STITCHING_DENSITY_QUANTILE == defaults.density_quantile
    assert pull_queue.STITCHING_TARGET_TRIANGLES == defaults.target_triangles
    assert pull_queue.STITCHING_READ_CHUNK_POINTS == defaults.read_chunk_points
    assert (
        pull_queue.STITCHING_MAXIMUM_POINTS
        == defaults.maximum_stitching_points
    )


def test_output_filename_contract_is_read_only():
    with pytest.raises(TypeError):
        pull_queue.MESH_JOB_OUTPUT_FILENAMES["pointCloud"] = "changed.laz"
