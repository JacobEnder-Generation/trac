from __future__ import annotations

import io
from datetime import timedelta
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def _compute_chart_context(
    price: pd.Series,
    bv: pd.Series,
    band_base: float,
    corridor: float,
    num_bands: int,
    y_mode: str,
) -> tuple[float, float, int, int]:
    p = price.dropna()
    p = p[p > 0]
    if p.empty:
        raise ValueError("No positive price data available.")

    if y_mode == "Price min/max ±20%":
        pmin, pmax = float(p.min()), float(p.max())
        ymin, ymax = max(pmin * 0.80, 1e-6), pmax * 1.20
    else:
        last_price = float(p.iloc[-1])
        ymin, ymax = max(last_price * 0.80, 1e-6), last_price * 1.20
    if ymax <= ymin:
        ymax = ymin * 2

    k_lo, k_hi = 1 - corridor, 1 + corridor
    bv_positive = bv[(bv > 0) & np.isfinite(bv)]
    bv_med = float(np.nanmedian(bv_positive)) if not bv_positive.empty else np.nan
    if np.isfinite(bv_med) and bv_med > 0:
        lo_mid = ymin / (1 + corridor)
        hi_mid = ymax / (1 - corridor)
        denom = np.log(band_base) if band_base > 0 else np.nan
        if not np.isfinite(denom) or denom == 0:
            n_lo, n_hi = -num_bands, num_bands
        else:
            n_lo = int(np.floor(np.log(lo_mid / bv_med) / denom)) - 2
            n_hi = int(np.ceil(np.log(hi_mid / bv_med) / denom)) + 2
            n_lo = max(n_lo, -num_bands)
            n_hi = min(n_hi, num_bands)
    else:
        n_lo, n_hi = -num_bands, num_bands

    return ymin, ymax, n_lo, n_hi


def make_chart(
    data: dict[str, pd.Series],
    band_base: float,
    corridor: float,
    num_bands: int,
    y_mode: str,
) -> go.Figure:
    dt = data["dt"]
    price = data["price"]
    bv = data["bv"]

    try:
        ymin, ymax, n_lo, n_hi = _compute_chart_context(
            price, bv, band_base, corridor, num_bands, y_mode
        )
    except ValueError:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dt,
            y=price,
            mode="lines",
            name="Price",
            line=dict(color="black", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dt,
            y=bv,
            mode="lines",
            name="BVPS",
            line=dict(color="navy", width=2, dash="dot"),
        )
    )

    k_lo, k_hi = 1 - corridor, 1 + corridor
    for n in range(n_lo, n_hi + 1):
        factor = band_base**n
        mid = bv * factor
        lo, hi = mid * k_lo, mid * k_hi

        fig.add_trace(
            go.Scatter(
                x=dt,
                y=lo,
                mode="lines",
                line=dict(color="rgba(95,160,255,0.7)", dash="dash", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dt,
                y=hi,
                mode="lines",
                line=dict(color="rgba(95,160,255,0.7)", dash="dash", width=1),
                fill="tonexty",
                fillcolor="rgba(169,200,255,0.25)",
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dt,
                y=mid,
                mode="lines",
                line=dict(color="navy", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_yaxes(
        type="log", range=[np.log10(ymin), np.log10(ymax)], tickformat="$~s"
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        template="simple_white",
    )
    return fig


def create_matplotlib_chart(
    ticker: str,
    data: dict[str, pd.Series],
    band_base: float,
    corridor: float,
    num_bands: int,
    y_mode: str,
) -> Optional[plt.Figure]:
    price = data["price"].copy().sort_index()
    bv = data["bv"].copy().sort_index()
    price[price <= 0] = np.nan
    bv[bv <= 0] = np.nan
    bv = bv.reindex(price.index)

    try:
        ymin, ymax, n_lo, n_hi = _compute_chart_context(
            price, bv, band_base, corridor, num_bands, y_mode
        )
    except ValueError:
        return None

    dt_index = price.index
    if isinstance(dt_index, pd.DatetimeIndex):
        dt_index = dt_index[~dt_index.isna()]
    if len(dt_index) == 0:
        return None

    dt_index = pd.to_datetime(pd.Index(dt_index))
    start_label = dt_index.min().strftime("%B %Y")
    end_label = dt_index.max().strftime("%B %Y")
    dt = dt_index.to_pydatetime()

    fig, ax = plt.subplots(figsize=(6, 8), dpi=150)
    ax.set_facecolor("white")

    band_color = "#5F8DFF"
    corridor_color = "#A9C8FF"

    k_lo, k_hi = 1 - corridor, 1 + corridor
    band_mid_values: list[float] = []
    for n in range(n_lo, n_hi + 1):
        factor = band_base**n
        mid = bv * factor
        lo = mid * k_lo
        hi = mid * k_hi
        ax.plot(
            dt,
            lo,
            color=band_color,
            linewidth=0.7,
            linestyle="--",
            alpha=0.7,
        )
        ax.plot(
            dt,
            hi,
            color=band_color,
            linewidth=0.7,
            linestyle="--",
            alpha=0.7,
        )
        ax.fill_between(
            dt,
            lo,
            hi,
            color=corridor_color,
            alpha=0.05,
            linewidth=0,
        )
        ax.plot(
            dt,
            mid,
            color=band_color,
            linewidth=0.7,
            linestyle=(0, (1, 2)),
            alpha=0.9,
        )
        mid_non_na = mid.dropna()
        if not mid_non_na.empty:
            band_mid_values.append(float(mid_non_na.iloc[-1]))

    ax.plot(
        dt,
        price,
        color="black",
        linewidth=1.6,
        solid_capstyle="round",
        label="Price",
    )
    ax.plot(
        dt,
        bv,
        color="#1F3A93",
        linewidth=1.1,
        linestyle=":",
        alpha=0.9,
        label="BVPS",
    )

    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    if len(dt) == 1:
        single = dt[0]
        ax.set_xlim(single - timedelta(days=5), single + timedelta(days=5))
    else:
        ax.set_xlim(dt[0], dt[-1])

    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    def _currency_formatter(val, _pos):
        if val >= 1_000_000_000:
            return f"${val / 1_000_000_000:.1f}B"
        if val >= 1_000_000:
            return f"${val / 1_000_000:.1f}M"
        if val >= 1_000:
            return f"${val:,.0f}"
        return f"${val:.2f}"

    currency_formatter = mticker.FuncFormatter(_currency_formatter)
    if band_mid_values:
        tick_positions = sorted({val for val in band_mid_values if val > 0})
        ax.yaxis.set_major_locator(mticker.FixedLocator(tick_positions))
    else:
        major_locator = mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=12)
        ax.yaxis.set_major_locator(major_locator)
    ax.yaxis.set_major_formatter(currency_formatter)
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    ax.grid(axis="y", color="#D8E3FF", linewidth=0.6, alpha=0.6)
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.5, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5F5")
    ax.spines["bottom"].set_color("#CBD5F5")
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    ax.legend(loc="upper left", frameon=False, fontsize=8)
    if start_label == end_label:
        title_suffix = start_label
    else:
        title_suffix = f"{start_label} - {end_label}"
    ax.set_title(
        f"{ticker.upper()} ({title_suffix})",
        fontsize=15,
        fontweight="bold",
        pad=12,
        color="#038CAD",
    )

    fig.tight_layout()
    return fig


def figure_to_png_bytes(fig: plt.Figure, dpi: int = 160) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()
