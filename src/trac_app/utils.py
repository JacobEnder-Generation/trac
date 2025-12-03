from __future__ import annotations

from typing import Any

import pandas as pd

from .config import TIME_RANGES


def extract_region(ticker: str) -> str:
    if "-" in ticker:
        suffix = ticker.split("-")[-1]
        return suffix[:2].upper()
    return "NA"


def format_date(ts: pd.Timestamp) -> str:
    return ts.strftime("%B %d, %Y").replace(" 0", " ")


def apply_time_range(data: dict[str, Any], label: str) -> dict[str, Any]:
    offset = TIME_RANGES.get(label)
    if offset is None:
        return data

    dt = data["dt"]
    price = data["price"]
    bv = data["bv"]

    price_valid = price.dropna()
    if price_valid.empty:
        return data

    end = price_valid.index.max()
    start = end - offset

    mask = (dt >= start) & (dt <= end)
    if not mask.any():
        return data

    dt_filtered = dt[mask]
    price_filtered = price.reindex(dt_filtered)
    bv_filtered = bv.reindex(dt_filtered)
    return {"dt": dt_filtered, "price": price_filtered, "bv": bv_filtered}

