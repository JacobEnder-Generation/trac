from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .db import DataFetcher, DatabaseConnection
from .processing import DataProcessor


def _prepare_fundamental_bvps(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Return fundamentals with computed BVPS (adjusted for invalid rows)."""
    if fundamentals.empty:
        return pd.DataFrame(columns=["DATE", "BVPS"])  # keep expected columns

    df = fundamentals.copy()
    mask = (
        df["ff_com_shs_out"].notna()
        & df["ff_com_shs_out"].ne(0)
        & df["ff_com_eq"].notna()
    )
    df = df.loc[mask].copy()
    if df.empty:
        return pd.DataFrame(columns=["DATE", "BVPS"])

    df["BVPS"] = df["ff_com_eq"].astype(float) / df["ff_com_shs_out"].astype(float)
    df = df[~df["BVPS"].isna()].copy()
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.normalize()
    df = df.sort_values("DATE")
    df = df.drop_duplicates("DATE", keep="last")
    return df[["DATE", "BVPS"]]


def _assign_quarterly_points(
    base: pd.DataFrame,
    values: pd.DataFrame,
    date_col: str,
    value_col: str,
    target_date_col: str,
    target_value_col: str,
) -> None:
    if base.empty or values.empty:
        return

    dates = base["Date"].values
    for _, row in values.iterrows():
        if pd.isna(row[date_col]):
            continue
        release_date = pd.Timestamp(row[date_col]).normalize()
        loc_candidates = np.where(dates >= release_date.to_numpy())[0]
        idx = loc_candidates[0] if len(loc_candidates) else None
        if idx is None:
            continue
        base.at[base.index[idx], target_date_col] = release_date
        base.at[base.index[idx], target_value_col] = float(row[value_col])


def load_backtest_record(ticker: str) -> Optional[pd.DataFrame]:
    """Build the backtester-friendly DataFrame for a single ticker."""
    with DatabaseConnection() as conn:
        fundamentals = DataFetcher.get_fundamentals_data(conn, ticker)
        daily = DataFetcher.get_daily_data(conn, ticker)
        forecast = DataFetcher.get_forecast_data(conn, ticker)
        splits = DataFetcher.get_split_history(conn, ticker)

    if daily.empty:
        return None

    fundamentals_adj, daily_adj = DataProcessor.adjust_for_splits(
        fundamentals, daily, splits
    )

    daily_adj = daily_adj.sort_values("p_date").dropna(subset=["p_date", "close"])
    fundamentals_adj = fundamentals_adj.sort_values("DATE")
    forecast = forecast.sort_values("date")

    record = pd.DataFrame(
        {
            "Date": pd.to_datetime(daily_adj["p_date"]).dt.normalize(),
            "Close": daily_adj["close"].astype(float).values,
            "BV_Date": pd.NaT,
            "BVPS": np.nan,
            "EstBV_Date": pd.NaT,
            "EstBVPS": np.nan,
            "FlatBVPS": np.nan,
        }
    )

    actual_bvps = _prepare_fundamental_bvps(fundamentals_adj)
    _assign_quarterly_points(
        record,
        actual_bvps,
        date_col="DATE",
        value_col="BVPS",
        target_date_col="BV_Date",
        target_value_col="BVPS",
    )

    if not forecast.empty:
        forecast_clean = forecast.copy()
        forecast_clean["date"] = pd.to_datetime(forecast_clean["date"]).dt.normalize()
        forecast_clean = forecast_clean.dropna(subset=["date", "bps_forecast"])
        forecast_clean = forecast_clean.sort_values("date")
        _assign_quarterly_points(
            record,
            forecast_clean,
            date_col="date",
            value_col="bps_forecast",
            target_date_col="EstBV_Date",
            target_value_col="EstBVPS",
        )

        if record["FlatBVPS"].isna().all():
            latest_est = forecast_clean.iloc[-1]["bps_forecast"]
            record["FlatBVPS"] = float(latest_est)

    if record["FlatBVPS"].isna().all() and not actual_bvps.empty:
        record["FlatBVPS"] = float(actual_bvps.iloc[-1]["BVPS"])

    record = record.sort_values("Date").drop_duplicates("Date", keep="last")
    record.reset_index(drop=True, inplace=True)
    return record


def load_backtest_records(tickers: Iterable[str]) -> Dict[str, pd.DataFrame]:
    results: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        rec = load_backtest_record(ticker)
        if rec is not None and not rec.empty:
            results[ticker] = rec
    return results
