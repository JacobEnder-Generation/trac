from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from .config import ETF_ADJUSTMENT_FACTORS
from .db import DataFetcher, DatabaseConnection


class DataProcessor:
    @staticmethod
    def interpolate_bvps(
        fundamentals: pd.DataFrame,
        forecast: pd.DataFrame,
        dates: pd.DatetimeIndex,
    ) -> pd.Series:
        # Ensure required columns exist to avoid KeyError when DataFrames are empty
        all_points = pd.DataFrame(columns=["date", "bvps"])

        if not fundamentals.empty:
            hist = fundamentals.copy()
            mask = (
                hist["ff_com_shs_out"].notna()
                & hist["ff_com_shs_out"].ne(0)
                & hist["ff_com_eq"].notna()
            )
            hist = hist.loc[mask]
            hist["bvps"] = hist["ff_com_eq"] / hist["ff_com_shs_out"]
            hist = hist[["DATE", "bvps"]].rename(columns={"DATE": "date"})
            all_points = pd.concat([all_points, hist])

        if not forecast.empty:
            fut = forecast[["date", "bps_forecast"]].rename(
                columns={"bps_forecast": "bvps"}
            )
            all_points = pd.concat([all_points, fut])

        all_points = (
            all_points.dropna(subset=["date"])
            .drop_duplicates("date")
            .sort_values("date")
        )

        if all_points.empty:
            return pd.Series(index=dates, dtype="float64")

        bvps = pd.Series(index=dates, dtype="float64")
        all_points["date"] = all_points["date"].dt.normalize()

        for _, row in all_points.iterrows():
            if row["date"] in bvps.index:
                bvps.loc[row["date"]] = row["bvps"]

        if bvps.notna().sum() >= 2:
            bvps = bvps.interpolate(method="linear")
        bvps = bvps.ffill().bfill()
        return bvps

    @staticmethod
    def adjust_for_splits(
        fundamentals: pd.DataFrame,
        daily: pd.DataFrame,
        splits: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if splits.empty:
            return fundamentals, daily

        fundamentals = fundamentals.copy()
        daily = daily.copy()
        splits = splits.sort_values("split_date")

        for _, split in splits.iterrows():
            factor = 1 / split["split_factor"]
            price_mask = daily["p_date"] < split["split_date"]
            daily.loc[price_mask, "close"] /= factor

            share_mask = fundamentals["DATE"] < split["split_date"]
            fundamentals.loc[share_mask, "ff_com_shs_out"] *= factor

        return fundamentals, daily


@st.cache_data(show_spinner=False)
def get_ticker_list() -> list[str]:
    with DatabaseConnection() as conn:
        return DataFetcher.get_available_tickers(conn)


@st.cache_data(show_spinner=False)
def get_etf_list() -> list[str]:
    with DatabaseConnection() as conn:
        return DataFetcher.get_etf_tickers(conn)


@st.cache_data(show_spinner=False)
def load_ticker_block(ticker: str) -> Optional[dict[str, pd.Series]]:
    with DatabaseConnection() as conn:
        fundamentals = DataFetcher.get_fundamentals_data(conn, ticker)
        daily = DataFetcher.get_daily_data(conn, ticker)
        forecast = DataFetcher.get_forecast_data(conn, ticker)
        splits = DataFetcher.get_split_history(conn, ticker)

    if daily.empty:
        return None

    fundamentals, daily = DataProcessor.adjust_for_splits(
        fundamentals, daily, splits
    )

    daily = daily.sort_values("p_date")
    fundamentals = fundamentals.sort_values("DATE")
    forecast = forecast.sort_values("date")

    price_series = (
        daily.set_index("p_date")["close"]
        .astype(float)
        .sort_index()
        .dropna()
    )

    if price_series.empty:
        return None

    start_candidates = [price_series.index.min()]
    if not fundamentals.empty:
        start_candidates.append(fundamentals["DATE"].min())
    if not forecast.empty:
        start_candidates.append(forecast["date"].min())
    start = min(start_candidates)

    end_candidates = [price_series.index.max()]
    if not fundamentals.empty:
        end_candidates.append(fundamentals["DATE"].max())
    forecast_max: Optional[pd.Timestamp] = None
    if not forecast.empty:
        forecast_dates = pd.to_datetime(forecast["date"]).dropna()
        if not forecast_dates.empty:
            forecast_max = forecast_dates.max().normalize()
            end_candidates.append(forecast_max)
    end = max(end_candidates)

    today = pd.Timestamp.today().normalize()
    default_cap = today + pd.DateOffset(months=6)
    horizon_cap = today + pd.DateOffset(months=36)
    forecast_cap = (
        forecast_max + pd.DateOffset(days=30) if forecast_max is not None else None
    )
    max_future = default_cap
    if forecast_cap is not None:
        max_future = max(max_future, forecast_cap)
    max_future = min(max_future, horizon_cap)
    if end > max_future:
        end = max_future

    daily_index = pd.date_range(start=start, end=end, freq="D")

    price_interp = (
        price_series.reindex(daily_index)
        .interpolate(method="time", limit_direction="both")
        .astype(float)
    )
    price_daily = pd.Series(np.nan, index=daily_index, dtype=float)
    price_first, price_last = price_series.index.min(), price_series.index.max()
    price_mask = (daily_index >= price_first) & (daily_index <= price_last)
    price_daily.loc[price_mask] = price_interp.loc[price_mask]

    bv_series = DataProcessor.interpolate_bvps(fundamentals, forecast, daily_index)
    bv_series = bv_series.astype(float)
    bv_series[bv_series <= 0] = np.nan

    if bv_series.dropna().empty:
        return None

    return {"dt": daily_index, "price": price_daily, "bv": bv_series}


@st.cache_data(show_spinner=False)
def load_etf_block(ticker: str) -> Optional[dict[str, pd.Series]]:
    with DatabaseConnection() as conn:
        daily = DataFetcher.get_etf_daily_data(conn, ticker)
        bvps = DataFetcher.get_etf_bvps_data(conn, ticker)
        splits = DataFetcher.get_split_history(conn, ticker)

    if daily.empty:
        return None

    daily = daily.sort_values("p_date")
    fundamentals = pd.DataFrame(
        {
            "DATE": pd.Series(dtype="datetime64[ns]"),
            "ff_com_eq": pd.Series(dtype="float64"),
            "ff_com_shs_out": pd.Series(dtype="float64"),
        }
    )

    if not bvps.empty:
        bvps = bvps.sort_values("DATE")
        adj = ETF_ADJUSTMENT_FACTORS.get(ticker, 0.0)
        if adj:
            bvps = bvps.assign(bvps=bvps["bvps"] * (1 - adj))
        forecast = pd.DataFrame(
            {
                "date": bvps["DATE"].to_numpy(),
                "bps_forecast": bvps["bvps"].to_numpy(),
            }
        )
    else:
        forecast = pd.DataFrame(columns=["date", "bps_forecast"])

    fundamentals, daily = DataProcessor.adjust_for_splits(fundamentals, daily, splits)

    price_series = (
        daily.set_index("p_date")["close"]
        .astype(float)
        .sort_index()
        .dropna()
    )
    if price_series.empty:
        return None

    start_candidates = [price_series.index.min()]
    if not forecast.empty:
        start_candidates.append(forecast["date"].min())
    start = min(start_candidates)

    end_candidates = [price_series.index.max()]
    forecast_max: Optional[pd.Timestamp] = None
    if not forecast.empty:
        forecast_dates = pd.to_datetime(forecast["date"]).dropna()
        if not forecast_dates.empty:
            forecast_max = forecast_dates.max().normalize()
            end_candidates.append(forecast_max)
    end = max(end_candidates)

    today = pd.Timestamp.today().normalize()
    default_cap = today + pd.DateOffset(months=6)
    horizon_cap = today + pd.DateOffset(months=36)
    forecast_cap = (
        forecast_max + pd.DateOffset(days=30) if forecast_max is not None else None
    )
    max_future = default_cap
    if forecast_cap is not None:
        max_future = max(max_future, forecast_cap)
    max_future = min(max_future, horizon_cap)
    if end > max_future:
        end = max_future

    daily_index = pd.date_range(start=start, end=end, freq="D")

    price_interp = (
        price_series.reindex(daily_index)
        .interpolate(method="time", limit_direction="both")
        .astype(float)
    )
    price_daily = pd.Series(np.nan, index=daily_index, dtype=float)
    price_first, price_last = price_series.index.min(), price_series.index.max()
    price_mask = (daily_index >= price_first) & (daily_index <= price_last)
    price_daily.loc[price_mask] = price_interp.loc[price_mask]

    bv_series = DataProcessor.interpolate_bvps(fundamentals, forecast, daily_index)
    bv_series = bv_series.astype(float)
    bv_series[bv_series <= 0] = np.nan

    if bv_series.dropna().empty:
        return None

    return {"dt": daily_index, "price": price_daily, "bv": bv_series}
