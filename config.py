"""Shared configuration helpers.

Small, dependency-free utilities for reading required configuration from the
environment. Library modules should let `ConfigError` propagate; entry-point
scripts catch it and decide how to report/exit.
"""

import os


class ConfigError(RuntimeError):
    """Raised when required configuration (e.g. an env var) is missing."""


def require_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise ConfigError(f"Missing required environment variable: {name}")

    return value
