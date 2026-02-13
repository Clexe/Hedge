"""
Centralised configuration loader.

Reads config/default.yaml, merges optional config/secrets.yaml, and exposes
a plain dict (or nested attribute-access object) to every other module.
Environment variables prefixed with HEDGE_ override YAML values.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CFG = _PROJECT_ROOT / "config" / "default.yaml"
_SECRETS_CFG = _PROJECT_ROOT / "config" / "secrets.yaml"


class _AttrDict(dict):
    """Dict subclass that allows attribute-style access (read-only helper)."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'") from None
        if isinstance(val, dict) and not isinstance(val, _AttrDict):
            val = _AttrDict(val)
            self[key] = val
        return val


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (mutates *base*)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _apply_env_overrides(cfg: dict, prefix: str = "HEDGE") -> dict:
    """
    Scan environment for variables like HEDGE__DATA__PROVIDER=polygon
    and override the corresponding nested key (data.provider).
    Double-underscore is the nesting separator.
    """
    for key, value in os.environ.items():
        if not key.startswith(prefix + "__"):
            continue
        parts = key[len(prefix) + 2 :].lower().split("__")
        d = cfg
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        # Basic type coercion.
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"  # type: ignore[assignment]
        else:
            try:
                value = int(value)  # type: ignore[assignment]
            except ValueError:
                try:
                    value = float(value)  # type: ignore[assignment]
                except ValueError:
                    pass
        d[parts[-1]] = value
    return cfg


def load_config(
    default_path: Path | str = _DEFAULT_CFG,
    secrets_path: Path | str = _SECRETS_CFG,
) -> _AttrDict:
    """Return the merged, env-overridden configuration as an AttrDict."""
    with open(default_path, "r") as fh:
        cfg: dict = yaml.safe_load(fh) or {}

    secrets_path = Path(secrets_path)
    if secrets_path.exists():
        with open(secrets_path, "r") as fh:
            secrets = yaml.safe_load(fh) or {}
        _deep_merge(cfg, secrets)

    cfg = _apply_env_overrides(cfg)
    return _AttrDict(cfg)


# Module-level singleton — import this everywhere.
settings: _AttrDict | None = None


def get_settings() -> _AttrDict:
    global settings
    if settings is None:
        settings = load_config()
    return settings
