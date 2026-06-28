import os
import stat as stat_mod

import config
import pytest
import r2
from botocore.exceptions import ClientError

VALID_ACCOUNT_ID = "abcdef0123456789abcdef0123456789"

posix_only = pytest.mark.skipif(
    not hasattr(os, "getuid"), reason="POSIX-only ownership/symlink checks"
)


# --- _r2_endpoint_url ---------------------------------------------------------


def test_endpoint_url_valid():
    assert (
        r2._r2_endpoint_url(VALID_ACCOUNT_ID)
        == f"https://{VALID_ACCOUNT_ID}.r2.cloudflarestorage.com"
    )


@pytest.mark.parametrize(
    "bad",
    [
        "short",
        VALID_ACCOUNT_ID + "x",  # 33 chars
        VALID_ACCOUNT_ID.upper(),  # uppercase
        "acct.evil.com/",
        "evil@host",
        "",
    ],
)
def test_endpoint_url_rejects_malformed(bad):
    with pytest.raises(config.ConfigError):
        r2._r2_endpoint_url(bad)


# --- _sanitize ---------------------------------------------------------------


def test_sanitize_replaces_illegal_chars():
    assert r2._sanitize("a!b c/d") == "a_b_c_d"


def test_sanitize_truncates_to_64():
    assert len(r2._sanitize("x" * 200)) == 64


def test_sanitize_empty_falls_back():
    assert r2._sanitize("") == "job"


# --- _safe_join --------------------------------------------------------------


def test_safe_join_uses_last_segment(tmp_path):
    assert r2._safe_join(tmp_path, "scans/2024/a.las") == tmp_path / "a.las"


def test_safe_join_strips_absolute_path(tmp_path):
    dest = r2._safe_join(tmp_path, "/etc/passwd")
    assert dest.parent == tmp_path
    assert dest.name == "passwd"


def test_safe_join_strips_traversal(tmp_path):
    dest = r2._safe_join(tmp_path, "../../../etc/cron.d/evil")
    assert dest.parent == tmp_path
    assert dest.name == "evil"


@pytest.mark.parametrize("bad", ["", ".", "..", "a/b/"])
def test_safe_join_rejects_degenerate(tmp_path, bad):
    with pytest.raises(ValueError):
        r2._safe_join(tmp_path, bad)


# --- _tmp_base_dir ------------------------------------------------------------


def test__tmp_base_dir_rejects_relative(monkeypatch):
    monkeypatch.setenv("P2BP_TMP_DIR", "relative/dir")
    with pytest.raises(config.ConfigError):
        r2._tmp_base_dir()


def test__tmp_base_dir_accepts_absolute(monkeypatch, tmp_path):
    monkeypatch.setenv("P2BP_TMP_DIR", str(tmp_path))
    assert r2._tmp_base_dir() == tmp_path


def test__tmp_base_dir_default(monkeypatch):
    monkeypatch.delenv("P2BP_TMP_DIR", raising=False)
    base = r2._tmp_base_dir()
    assert base.is_absolute()
    assert base.name == "p2bp-tmp"


# --- temp directories --------------------------------------------------------


def test_new_download_dir_unique_and_nested():
    a = r2.new_download_dir()
    b = r2.new_download_dir()
    assert a != b
    assert a.exists() and b.exists()
    assert a.parent == b.parent == r2._r2_downloads_dir()


def test_new_download_dir_uses_label_prefix():
    d = r2.new_download_dir("proj-42")
    assert d.name.startswith("proj-42-")


def test_temp_download_dir_cleans_up():
    with r2.temp_download_dir() as d:
        assert d.exists()
        saved = d
    assert not saved.exists()


# --- _ensure_private_dir -----------------------------------------------------


def test_ensure_private_dir_creates(tmp_path):
    target = tmp_path / "ws"
    assert r2._ensure_private_dir(target) == target
    assert target.is_dir()


def test_ensure_private_dir_missing_parent_raises(tmp_path):
    with pytest.raises(config.ConfigError):
        r2._ensure_private_dir(tmp_path / "missing" / "ws")


@posix_only
def test_ensure_private_dir_rejects_symlink(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    os.symlink(real, link)
    with pytest.raises(r2.InsecureTempDirError):
        r2._ensure_private_dir(link)


@posix_only
def test_ensure_private_dir_rejects_foreign_owner(tmp_path, monkeypatch):
    target = tmp_path / "ws"
    target.mkdir(mode=0o700)
    monkeypatch.setattr(os, "getuid", lambda: os.stat(target).st_uid + 1)
    with pytest.raises(r2.InsecureTempDirError):
        r2._ensure_private_dir(target)


@posix_only
def test_ensure_private_dir_tightens_mode(tmp_path):
    target = tmp_path / "ws"
    target.mkdir()
    os.chmod(target, 0o755)
    r2._ensure_private_dir(target)
    assert stat_mod.S_IMODE(os.stat(target).st_mode) == 0o700


# --- downloads ---------------------------------------------------------------


def test_download_object_writes_and_calls_client(fake_client, tmp_path):
    dest = tmp_path / "a.bin"
    result = r2.download_object(fake_client, "k", dest, bucket="b")
    assert result == dest
    assert dest.read_bytes() == b"data"
    assert fake_client.calls == [("b", "k", str(dest))]


def test_download_object_refuses_overwrite(fake_client, tmp_path):
    dest = tmp_path / "a.bin"
    dest.write_text("existing")
    with pytest.raises(FileExistsError):
        r2.download_object(fake_client, "k", dest, bucket="b")


def test_download_object_overwrite_allowed(fake_client, tmp_path):
    dest = tmp_path / "a.bin"
    dest.write_text("existing")
    r2.download_object(fake_client, "k", dest, bucket="b", overwrite=True)
    assert dest.read_bytes() == b"data"


def test_download_object_bucket_defaults_to_env(fake_client, tmp_path, monkeypatch):
    monkeypatch.setenv("R2_BUCKET", "env-bucket")
    r2.download_object(fake_client, "k", tmp_path / "a.bin")
    assert fake_client.calls[-1][0] == "env-bucket"


def test_download_object_missing_bucket_raises(fake_client, tmp_path, monkeypatch):
    monkeypatch.delenv("R2_BUCKET", raising=False)
    with pytest.raises(config.ConfigError):
        r2.download_object(fake_client, "k", tmp_path / "a.bin")


def test_download_to_dir_derives_name_from_key(fake_client, tmp_path):
    result = r2.download_to_dir(fake_client, "scans/2024/a.las", tmp_path, bucket="b")
    assert result == tmp_path / "a.las"
    assert result.exists()


def test_download_to_dir_collision_raises(fake_client, tmp_path):
    r2.download_to_dir(fake_client, "2024/scan.las", tmp_path, bucket="b")
    with pytest.raises(FileExistsError):
        r2.download_to_dir(fake_client, "2025/scan.las", tmp_path, bucket="b")


def test_download_to_temp_fresh_dir(fake_client):
    result = r2.download_to_temp(fake_client, "scans/x.las", bucket="b")
    assert result.name == "x.las"
    assert result.exists()
    # Its contract: a fresh unique dir directly under the downloads root.
    assert result.parent.parent == r2._r2_downloads_dir()


def test_download_object_creates_parent_dirs(fake_client, tmp_path):
    dest = tmp_path / "sub" / "nested" / "a.bin"
    r2.download_object(fake_client, "k", dest, bucket="b")
    assert dest.exists()


def test_download_to_dir_filename_override(fake_client, tmp_path):
    result = r2.download_to_dir(
        fake_client, "scans/a.las", tmp_path, bucket="b", filename="renamed.las"
    )
    assert result == tmp_path / "renamed.las"
    assert result.exists()


def test_download_to_dir_overwrite_passthrough(fake_client, tmp_path):
    r2.download_to_dir(fake_client, "2024/scan.las", tmp_path, bucket="b")
    # Same basename would collide; overwrite must thread through to download_object.
    r2.download_to_dir(fake_client, "2025/scan.las", tmp_path, bucket="b", overwrite=True)
    assert (tmp_path / "scan.las").exists()


# --- client / config ---------------------------------------------------------


def test_create_r2_client_missing_account_id_raises(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    with pytest.raises(config.ConfigError):
        r2.create_r2_client()


def test_create_r2_client_builds_endpoint(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", VALID_ACCOUNT_ID)
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "ak")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "sk")
    captured = {}

    def fake_boto_client(service, **kwargs):
        captured["service"] = service
        captured.update(kwargs)
        return "CLIENT"

    monkeypatch.setattr(r2.boto3, "client", fake_boto_client)

    assert r2.create_r2_client() == "CLIENT"
    assert captured["service"] == "s3"
    assert (
        captured["endpoint_url"]
        == f"https://{VALID_ACCOUNT_ID}.r2.cloudflarestorage.com"
    )
    assert captured["region_name"] == "auto"
    assert captured["aws_access_key_id"] == "ak"
    assert captured["aws_secret_access_key"] == "sk"


def test_default_bucket_from_env(monkeypatch):
    monkeypatch.setenv("R2_BUCKET", "my-bucket")
    assert r2.default_bucket() == "my-bucket"


def test_default_bucket_missing_raises(monkeypatch):
    monkeypatch.delenv("R2_BUCKET", raising=False)
    with pytest.raises(config.ConfigError):
        r2.default_bucket()


# --- misc helpers ------------------------------------------------------------


def test_safe_join_normalizes_backslashes(tmp_path):
    assert r2._safe_join(tmp_path, "a\\b\\c.las") == tmp_path / "c.las"


def test_insecure_temp_dir_error_is_runtime_error():
    assert issubclass(r2.InsecureTempDirError, RuntimeError)


def test_ensure_private_dir_rejects_regular_file(tmp_path):
    target = tmp_path / "afile"
    target.write_text("not a dir")
    with pytest.raises(FileExistsError):
        r2._ensure_private_dir(target)


def test_temp_download_dir_cleans_up_on_exception():
    saved = None
    with pytest.raises(RuntimeError):
        with r2.temp_download_dir() as path:
            saved = path
            assert path.exists()
            raise RuntimeError("boom")
    assert saved is not None
    assert not saved.exists()


# --- uploads -----------------------------------------------------------------


def test_upload_object_calls_client_and_returns_key(fake_client, tmp_path):
    source = tmp_path / "a.bin"
    source.write_bytes(b"payload")
    result = r2.upload_object(fake_client, source, "k", bucket="b")
    assert result == "k"
    assert fake_client.upload_calls == [(str(source), "b", "k")]


def test_upload_object_refuses_overwrite(fake_client, tmp_path):
    source = tmp_path / "a.bin"
    source.write_bytes(b"payload")
    fake_client.existing.add("k")
    with pytest.raises(FileExistsError):
        r2.upload_object(fake_client, source, "k", bucket="b")
    # The transfer must not happen when the clobber check fails.
    assert fake_client.upload_calls == []


def test_upload_object_overwrite_allowed(fake_client, tmp_path):
    source = tmp_path / "a.bin"
    source.write_bytes(b"payload")
    fake_client.existing.add("k")
    r2.upload_object(fake_client, source, "k", bucket="b", overwrite=True)
    assert fake_client.upload_calls == [(str(source), "b", "k")]


def test_upload_object_bucket_defaults_to_env(fake_client, tmp_path, monkeypatch):
    monkeypatch.setenv("R2_BUCKET", "env-bucket")
    source = tmp_path / "a.bin"
    source.write_bytes(b"payload")
    r2.upload_object(fake_client, source, "k")
    assert fake_client.upload_calls[-1][1] == "env-bucket"


def test_upload_object_missing_bucket_raises(fake_client, tmp_path, monkeypatch):
    monkeypatch.delenv("R2_BUCKET", raising=False)
    source = tmp_path / "a.bin"
    source.write_bytes(b"payload")
    with pytest.raises(config.ConfigError):
        r2.upload_object(fake_client, source, "k")


@pytest.mark.parametrize("bad", ["", ".", ".."])
def test_upload_object_rejects_degenerate_key(fake_client, tmp_path, bad):
    source = tmp_path / "a.bin"
    source.write_bytes(b"payload")
    with pytest.raises(ValueError):
        r2.upload_object(fake_client, source, bad, bucket="b")


def test_upload_object_accepts_slashed_key(fake_client, tmp_path):
    source = tmp_path / "a.las"
    source.write_bytes(b"payload")
    key = "scans/2024/a.las"
    assert r2.upload_object(fake_client, source, key, bucket="b") == key
    assert fake_client.upload_calls == [(str(source), "b", key)]


def test_upload_from_dir_uses_explicit_key(fake_client, tmp_path):
    source = tmp_path / "scan.las"
    source.write_bytes(b"payload")
    result = r2.upload_from_dir(
        fake_client, tmp_path, "scan.las", "scans/2024/scan.las", bucket="b"
    )
    assert result == "scans/2024/scan.las"
    assert fake_client.upload_calls == [(str(source), "b", "scans/2024/scan.las")]


def test_upload_from_dir_does_not_infer_key_from_filename(fake_client, tmp_path):
    source = tmp_path / "scan.las"
    source.write_bytes(b"payload")
    # The key is unrelated to the filename; passing a distinct one is honored.
    result = r2.upload_from_dir(fake_client, tmp_path, "scan.las", "renamed.las", bucket="b")
    assert result == "renamed.las"
    assert fake_client.upload_calls == [(str(source), "b", "renamed.las")]


def test_upload_from_dir_strips_traversal(fake_client, tmp_path):
    # _safe_join collapses to basename, so the source is tmp_path/evil; the
    # "../etc" cannot read outside the directory. The key is independent.
    source = tmp_path / "evil"
    source.write_bytes(b"payload")
    result = r2.upload_from_dir(fake_client, tmp_path, "../../../etc/evil", "out", bucket="b")
    assert result == "out"
    assert fake_client.upload_calls == [(str(source), "b", "out")]


def test_upload_from_dir_collision_raises(fake_client, tmp_path):
    source = tmp_path / "scan.las"
    source.write_bytes(b"payload")
    fake_client.existing.add("scan.las")
    with pytest.raises(FileExistsError):
        r2.upload_from_dir(fake_client, tmp_path, "scan.las", "scan.las", bucket="b")
    assert fake_client.upload_calls == []


def test_upload_from_dir_rejects_degenerate_filename(fake_client, tmp_path):
    with pytest.raises(ValueError):
        r2.upload_from_dir(fake_client, tmp_path, "", "somekey", bucket="b")


def test_upload_from_dir_rejects_degenerate_key(fake_client, tmp_path):
    source = tmp_path / "scan.las"
    source.write_bytes(b"payload")
    with pytest.raises(ValueError):
        r2.upload_from_dir(fake_client, tmp_path, "scan.las", "", bucket="b")


# --- _object_exists / _safe_key ----------------------------------------------


def test_object_exists_false_for_missing(fake_client):
    assert r2._object_exists(fake_client, "b", "absent") is False


def test_object_exists_true_when_present(fake_client):
    fake_client.existing.add("k")
    assert r2._object_exists(fake_client, "b", "k") is True


def test_object_exists_reraises_non_404():
    class ForbiddenClient:
        def head_object(self, Bucket, Key):
            raise ClientError(
                {"Error": {"Code": "403", "Message": "Forbidden"}}, "HeadObject"
            )

    with pytest.raises(ClientError):
        r2._object_exists(ForbiddenClient(), "b", "k")


@pytest.mark.parametrize("bad", ["", ".", ".."])
def test_safe_key_rejects_degenerate(bad):
    with pytest.raises(ValueError):
        r2._safe_key(bad)


@pytest.mark.parametrize("ok", ["a.las", "scans/2024/a.las", "with space"])
def test_safe_key_accepts_normal_keys(ok):
    assert r2._safe_key(ok) == ok
