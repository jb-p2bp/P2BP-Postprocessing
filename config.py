"""Shared configuration helpers.

Small, dependency-free utilities for reading required configuration from the
environment. Library modules should let `ConfigError` propagate; entry-point
scripts catch it and decide how to report/exit.
"""

import os

__all__ = ["ConfigError", "require_env"]


class ConfigError(RuntimeError):
    """Raised when required configuration (e.g. an env var) is missing."""


def require_env(name: str) -> str:
    """Return the value of env var `name`, or raise ConfigError if unset.

    An empty string is treated as missing.
    """

    value = os.getenv(name)

    if not value:
        raise ConfigError(f"Missing required environment variable: {name}")

    return value
