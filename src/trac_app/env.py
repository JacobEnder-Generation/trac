from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"
ENV_PATH = ASSETS_DIR / ".env"


def _parse_env_lines(lines: Iterable[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            values[key] = value
    return values


@lru_cache(maxsize=1)
def load_env_values() -> dict[str, str]:
    if not ENV_PATH.exists():
        return {}
    try:
        content = ENV_PATH.read_text(encoding="utf-8")
    except OSError:
        return {}
    return _parse_env_lines(content.splitlines())


def get_env_value(key: str, default: str | None = None) -> str | None:
    env_val = os.environ.get(key)
    if env_val:
        return env_val
    return load_env_values().get(key, default)
