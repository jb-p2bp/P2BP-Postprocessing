"""Tests for the queue worker entry point.

The worker is an EC2-hosted long-runner with many external dependencies
(IMDS, boto3, Cloudflare); these tests cover the thin config surface that
is practical to exercise without standing up the surrounding infrastructure.
"""

from pathlib import Path
from types import SimpleNamespace
import zipfile

import config
import pull_queue
import pytest

REQUIRED_WORKER_ENV = (
    "CLOUDFLARE_ACCOUNT_ID",
    "CLOUDFLARE_QUEUE_ID",
    "CLOUDFLARE_API_TOKEN",
    "AWS_REGION",
)


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
    for var in REQUIRED_WORKER_ENV:
        monkeypatch.delenv(var, raising=False)

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
        return SimpleNamespace(point_count=10, result="REGISTRATION")

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

    monkeypatch.setattr(pull_queue, "merge_scan_projects", fake_merge_scan_projects)
    monkeypatch.setattr(pull_queue, "export_merged_cloud_outputs", fake_export_merged_cloud_outputs)

    pull_queue.process_message(
        {
            "type": "mesh.generate",
            "version": 1,
            "organizationId": "org_123",
            "projectId": "proj_456",
            "zoneScanObjectKeys": ["uploads/zone-a.zip"],
        }
    )

    assert fake_client.calls == [
        (
            "env-bucket",
            "uploads/zone-a.zip",
            str(Path(captured["inputs"][0]).parent.parent / "archives" / "000-zone-a.zip"),
        )
    ]
    assert Path(captured["inputs"][0]).name == "000-zone-a.scanproject"
    assert captured["manifest_exists"] is True
    assert captured["output"].name == "merged-point-cloud.laz"
    assert captured["merge_kwargs"] == {
        "deduplicate_voxel": pull_queue.MERGED_POINT_CLOUD_DEDUPLICATE_VOXEL,
        "export_minimum_confidence": 0,
    }
    assert captured["preview_result"] == "REGISTRATION"
    assert captured["bin_output"].name == "merged-point-cloud.bin"
    assert captured["preview_output"].name == "merged-point-cloud.preview.laz"
    assert captured["preview_bin_output"].name == "merged-point-cloud.preview.bin"
    assert captured["preview_kwargs"] == {
        "minimum_confidence": 0,
        "deduplicate_voxel": pull_queue.PREVIEW_POINT_CLOUD_DEDUPLICATE_VOXEL,
    }
    assert fake_client.upload_calls == [
        (
            str(Path(captured["output"])),
            "env-bucket",
            "organizations/org_123/projects/proj_456/merged-point-cloud.laz",
        ),
        (
            str(Path(captured["output"]).with_name("merged-point-cloud.bin")),
            "env-bucket",
            "organizations/org_123/projects/proj_456/merged-point-cloud.bin",
        ),
        (
            str(Path(captured["output"]).with_name("merged-point-cloud.preview.laz")),
            "env-bucket",
            "organizations/org_123/projects/proj_456/merged-point-cloud.preview.laz",
        ),
        (
            str(Path(captured["output"]).with_name("merged-point-cloud.preview.bin")),
            "env-bucket",
            "organizations/org_123/projects/proj_456/merged-point-cloud.preview.bin",
        ),
    ]


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
