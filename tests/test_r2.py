import os
import stat as stat_mod

import config
import pytest
import r2

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
