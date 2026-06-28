import config
import pytest


def test_require_env_returns_value(monkeypatch):
    monkeypatch.setenv("SOME_VAR", "value")
    assert config.require_env("SOME_VAR") == "value"


def test_require_env_missing_raises(monkeypatch):
    monkeypatch.delenv("SOME_VAR", raising=False)
    with pytest.raises(config.ConfigError):
        config.require_env("SOME_VAR")


def test_require_env_empty_string_is_missing(monkeypatch):
    monkeypatch.setenv("SOME_VAR", "")
    with pytest.raises(config.ConfigError):
        config.require_env("SOME_VAR")


def test_config_error_is_runtime_error():
    assert issubclass(config.ConfigError, RuntimeError)
