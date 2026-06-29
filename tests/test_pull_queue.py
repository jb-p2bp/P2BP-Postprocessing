"""Tests for the queue worker entry point.

The worker is an EC2-hosted long-runner with many external dependencies
(IMDS, boto3, Cloudflare); these tests cover the thin config surface that
is practical to exercise without standing up the surrounding infrastructure.
"""

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
