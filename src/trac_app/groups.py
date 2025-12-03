from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"
GROUPS_FILE = ASSETS_DIR / "ticker_groups.json"


@dataclass(frozen=True)
class TickerGroup:
    name: str
    tickers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "tickers": list(dict.fromkeys(self.tickers))}


def _normalise_name(name: str) -> str:
    return name.strip().lower()


def load_groups() -> list[TickerGroup]:
    if not GROUPS_FILE.exists():
        return []
    try:
        raw = json.loads(GROUPS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    groups: list[TickerGroup] = []
    payload = raw if isinstance(raw, list) else raw.get("groups", [])
    for entry in payload if isinstance(payload, list) else []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        tickers = entry.get("tickers")
        if not isinstance(name, str) or not isinstance(tickers, list):
            continue
        cleaned_tickers = [
            str(t).strip() for t in tickers if isinstance(t, str) and t.strip()
        ]
        if not name.strip() or not cleaned_tickers:
            continue
        groups.append(TickerGroup(name=name.strip(), tickers=tuple(dict.fromkeys(cleaned_tickers))))
    return sorted(groups, key=lambda g: g.name.lower())


def _write_groups(groups: Iterable[TickerGroup]) -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    serialisable = [group.to_dict() for group in groups]
    GROUPS_FILE.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")


def upsert_group(name: str, tickers: Iterable[str]) -> None:
    trimmed_name = name.strip()
    if not trimmed_name:
        raise ValueError("Group name cannot be empty.")
    tickers_list = [t.strip() for t in tickers if t and t.strip()]
    if not tickers_list:
        raise ValueError("Ticker group must contain at least one ticker")

    existing = load_groups()
    normalised_target = _normalise_name(trimmed_name)

    filtered = [g for g in existing if _normalise_name(g.name) != normalised_target]
    filtered.append(TickerGroup(name=trimmed_name, tickers=tuple(dict.fromkeys(tickers_list))))
    _write_groups(filtered)


def delete_group(name: str) -> bool:
    trimmed_name = name.strip()
    if not trimmed_name:
        return False

    existing = load_groups()
    normalised_target = _normalise_name(trimmed_name)
    remaining = [g for g in existing if _normalise_name(g.name) != normalised_target]
    if len(remaining) == len(existing):
        return False
    _write_groups(remaining)
    return True
