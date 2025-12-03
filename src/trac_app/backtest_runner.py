from __future__ import annotations

import io
import tempfile
from dataclasses import dataclass
from datetime import date as _dt_date
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from PyPDF2 import PdfMerger

from trac_app.trac_chartbook_backtest import (
    BacktestParameters,
    BacktestResult,
    compute_backtest,
    render_pdf,
    _portfolio_equity_and_weights,
    _render_aggregated_statistics,
    _render_daily_report,
    _render_portfolio_summary,
)

from .backtest_data import load_backtest_records


@dataclass
class SingleBacktestArtifacts:
    result: BacktestResult
    pdf_bytes: bytes
    pdf_name: str
    png_bytes: bytes
    cycles_csv: bytes
    benchmark_csv: bytes
    strategy_csv: bytes


@dataclass
class PortfolioArtifacts:
    idx: pd.DatetimeIndex
    equity: pd.Series
    weights: pd.DataFrame
    equity_pdf: Optional[bytes]
    daily_report_pdf: Optional[bytes]
    stats_pdf: Optional[bytes]
    daily_holdings_csv: Optional[bytes]
    equity_csv: Optional[bytes]
    trades_csv: Optional[bytes]
    summary_csv: Optional[bytes]


@dataclass
class BacktestRunOutput:
    per_ticker: Dict[str, SingleBacktestArtifacts]
    portfolio: Optional[PortfolioArtifacts]
    combined_pdf: Optional[bytes]
    combined_name: Optional[str]
    failed: List[str]


def _build_csv_bytes(lines: List[List[str]]) -> bytes:
    buffer = io.StringIO()
    for row in lines:
        buffer.write(",".join(row))
        buffer.write("\n")
    return buffer.getvalue().encode("utf-8")


def _build_benchmark_csv(result: BacktestResult) -> bytes:
    bench = result.bench
    lines = [
        ["BuyHold_Start", "BuyHold_End", "Total", "Annualized", "Days"],
        [
            bench["StartDate"].date().isoformat(),
            bench["EndDate"].date().isoformat(),
            f"{bench['TotalReturn']:.6f}",
            f"{bench['Annualized']:.6f}",
            str(bench["Days"]),
        ],
    ]
    return _build_csv_bytes(lines)


def _build_strategy_csv(result: BacktestResult) -> bytes:
    strat = result.strat_rf
    header = ["HasData", "Start", "End", "Total", "Annualized", "Days", "NumCycles", "RF", "Span"]
    if strat.get("HasData", True):
        row = [
            "True",
            strat["StartDate"].date().isoformat(),
            strat["EndDate"].date().isoformat(),
            f"{strat['TotalReturn']:.6f}",
            f"{strat['Annualized']:.6f}",
            str(strat["Days"]),
            str(strat["NumCycles"]),
            "0.03",
            "cycles",
        ]
    else:
        row = ["False", "", "", "", "", "", "0", "0.03", "cycles"]
    return _build_csv_bytes([header, row])


def run_backtests(
    tickers: Iterable[str],
    *,
    rule: str = "revised_exit",
    weighting: str = "equal",
    generate_portfolio: bool = True,
    label: Optional[str] = None,
    params: Optional[BacktestParameters] = None,
) -> BacktestRunOutput:
    tickers = list(dict.fromkeys(tickers))  # deduplicate preserving order
    if not tickers:
        return BacktestRunOutput(per_ticker={}, portfolio=None, combined_pdf=None, combined_name=None, failed=[])

    records = load_backtest_records(tickers)
    failed = [t for t in tickers if t not in records]
    if not records:
        return BacktestRunOutput(per_ticker={}, portfolio=None, combined_pdf=None, combined_name=None, failed=failed)

    per_ticker: Dict[str, SingleBacktestArtifacts] = {}
    pdf_sources: List[bytes] = []
    today_str = _dt_date.today().strftime("%Y%m%d")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        for ticker, record in records.items():
            result = compute_backtest(ticker, record, rule, params=params)

            safe_ticker = re.sub(r"[^A-Za-z0-9._-]", "_", ticker)
            pdf_filename = f"{safe_ticker}_TRAC_Backtest_Chartbook_{today_str}.pdf"
            pdf_path = tmp_path / pdf_filename
            png_path = tmp_path / f"{safe_ticker}_{rule}.png"
            render_pdf(
                f"{result.ticker}  •  {'Core_BOB_Strategy' if rule == 'revised_exit' else rule}",
                result.dates,
                result.close,
                result.bvps,
                result.buys,
                result.cycles,
                pdf_path,
                png_path,
                result.bench,
                result.strat_rf,
                result.params,
            )

            pdf_bytes = pdf_path.read_bytes()
            pdf_sources.append(pdf_bytes)
            png_bytes = png_path.read_bytes() if png_path.exists() else b""

            cycles_csv = result.cycles.to_csv(index=False).encode("utf-8")
            bench_csv = _build_benchmark_csv(result)
            strat_csv = _build_strategy_csv(result)

            per_ticker[ticker] = SingleBacktestArtifacts(
                result=result,
                pdf_bytes=pdf_bytes,
                pdf_name=pdf_filename,
                png_bytes=png_bytes,
                cycles_csv=cycles_csv,
                benchmark_csv=bench_csv,
                strategy_csv=strat_csv,
            )

        portfolio_artifacts: Optional[PortfolioArtifacts] = None

        if generate_portfolio and len(per_ticker) > 0:
            prices_dict = {t: art.result.close for t, art in per_ticker.items()}
            cycles_dict = {t: art.result.cycles for t, art in per_ticker.items()}
            buys_dict = {t: art.result.buys for t, art in per_ticker.items()}
            fmv_dict = {t: art.result.fmv for t, art in per_ticker.items()}

            idxP, eqP, WP = _portfolio_equity_and_weights(
                prices_dict,
                cycles_dict,
                buys_dict,
                fmv_dict=fmv_dict,
                rf=0.03,
                weighting=weighting,
            )

            aligned_prices = {
                t: art.result.close.reindex(idxP).astype(float).ffill().bfill()
                for t, art in per_ticker.items()
            }

            base_name = "Portfolio"

            portfolio_pdf_path = tmp_path / f"{base_name}_PORTFOLIO.pdf"
            _render_portfolio_summary(
                base_name,
                idxP,
                eqP,
                WP,
                portfolio_pdf_path,
                bm_idx=None,
                bm_curve=None,
                bm_ann=None,
                bm_end=None,
            )

            stats_pdf_path = tmp_path / f"{base_name}_STATS.pdf"
            _render_aggregated_statistics(base_name, cycles_dict, stats_pdf_path)

            daily_pdf_path = tmp_path / f"{base_name}_DAILY.pdf"
            _render_daily_report(
                base_name,
                idxP,
                eqP,
                WP,
                cycles_dict,
                prices_dict,
                daily_pdf_path,
                bm_idx=None,
                bm_curve=None,
                bm_ann=None,
                bm_end=None,
            )

            if portfolio_pdf_path.exists():
                pdf_sources.append(portfolio_pdf_path.read_bytes())
            if stats_pdf_path.exists():
                pdf_sources.append(stats_pdf_path.read_bytes())
            if daily_pdf_path.exists():
                pdf_sources.append(daily_pdf_path.read_bytes())

            # Generate CSV-style artifacts mirroring CLI exports
            daily_holdings_rows: List[Dict[str, str]] = []
            for date, weights in WP.iterrows():
                date_str = date.strftime("%Y-%m-%d")
                for ticker, weight in weights.items():
                    if weight <= 0.001:
                        continue
                    price_series = aligned_prices.get(ticker)
                    current_price = float(price_series.loc[date]) if price_series is not None else float("nan")

                    cycles_df = cycles_dict.get(ticker)
                    entry_date = ""
                    entry_price = ""
                    unrealized = ""
                    if cycles_df is not None and not cycles_df.empty:
                        open_cycles = cycles_df[cycles_df["SellDate"].isna()]
                        if not open_cycles.empty:
                            latest = open_cycles.iloc[-1]
                            entry_price = f"{float(latest['BuyPx']):.2f}"
                            entry_date = pd.to_datetime(latest["BuyDate"]).strftime("%Y-%m-%d")
                            unrealized_return = (current_price / float(latest["BuyPx"]) - 1.0) * 100.0
                            unrealized = f"{unrealized_return:+.2f}"

                    daily_holdings_rows.append(
                        {
                            "Date": date_str,
                            "Ticker": ticker,
                            "Weight_%": f"{weight * 100:.2f}",
                            "Entry_Date": entry_date,
                            "Entry_Price": entry_price,
                            "Current_Price": f"{current_price:.2f}",
                            "Unrealized_Return_%": unrealized,
                        }
                    )

                cash_weight = 1.0 - weights.sum()
                if cash_weight > 0.001:
                    daily_holdings_rows.append(
                        {
                            "Date": date_str,
                            "Ticker": "CASH",
                            "Weight_%": f"{cash_weight * 100:.2f}",
                            "Entry_Date": "",
                            "Entry_Price": "",
                            "Current_Price": "",
                            "Unrealized_Return_%": "",
                        }
                    )

            daily_holdings_csv: Optional[bytes] = None
            if daily_holdings_rows:
                daily_holdings_df = pd.DataFrame(daily_holdings_rows)
                daily_holdings_csv = daily_holdings_df.to_csv(index=False).encode("utf-8")

            equity_rows: List[Dict[str, str]] = []
            for i, date in enumerate(idxP):
                value = float(eqP.iloc[i])
                if i == 0:
                    daily_return = 0.0
                else:
                    prev = float(eqP.iloc[i - 1])
                    daily_return = (value / prev - 1.0) * 100.0 if prev != 0 else 0.0
                cumulative = (value / 100.0 - 1.0) * 100.0
                equity_rows.append(
                    {
                        "Date": date.strftime("%Y-%m-%d"),
                        "Portfolio_Value": f"{value:.2f}",
                        "Daily_Return_%": f"{daily_return:+.4f}",
                        "Cumulative_Return_%": f"{cumulative:+.2f}",
                    }
                )

            equity_csv = pd.DataFrame(equity_rows).to_csv(index=False).encode("utf-8")

            all_trades: List[pd.DataFrame] = []
            for ticker, cycles_df in cycles_dict.items():
                if cycles_df is None or cycles_df.empty:
                    continue
                trades = cycles_df.copy()
                trades.insert(0, "Ticker", ticker)
                all_trades.append(trades)

            trades_csv = None
            summary_csv = None
            if all_trades:
                master_df = pd.concat(all_trades, ignore_index=True)
                master_df = master_df.sort_values("BuyDate").reset_index(drop=True)
                master_df["HoldDays"] = (
                    pd.to_datetime(master_df["SellDate"]) - pd.to_datetime(master_df["BuyDate"])
                ).dt.days
                trades_csv = master_df.to_csv(index=False).encode("utf-8")

                summary_rows = []
                for ticker in master_df["Ticker"].unique():
                    completed = master_df[master_df["Ticker"] == ticker].dropna(subset=["SellDate"])
                    if completed.empty:
                        continue
                    wins = completed[completed["ReturnPct"] > 0]
                    summary_rows.append(
                        {
                            "Ticker": ticker,
                            "Total_Trades": len(completed),
                            "Wins": len(wins),
                            "Losses": len(completed) - len(wins),
                            "Win_Rate_%": len(wins) / len(completed) * 100.0,
                            "Avg_Return_%": completed["ReturnPct"].mean(),
                            "Best_Trade_%": completed["ReturnPct"].max(),
                            "Worst_Trade_%": completed["ReturnPct"].min(),
                            "Avg_Hold_Days": completed["HoldDays"].mean(),
                        }
                    )

                if summary_rows:
                    summary_df = pd.DataFrame(summary_rows).sort_values("Avg_Return_%", ascending=False)
                    summary_csv = summary_df.to_csv(index=False).encode("utf-8")

            portfolio_artifacts = PortfolioArtifacts(
                idx=idxP,
                equity=eqP,
                weights=WP,
                equity_pdf=portfolio_pdf_path.read_bytes() if portfolio_pdf_path.exists() else None,
                daily_report_pdf=daily_pdf_path.read_bytes() if daily_pdf_path.exists() else None,
                stats_pdf=stats_pdf_path.read_bytes() if stats_pdf_path.exists() else None,
                daily_holdings_csv=daily_holdings_csv,
                equity_csv=equity_csv,
                trades_csv=trades_csv,
                summary_csv=summary_csv,
            )

    combined_pdf: Optional[bytes] = None
    combined_name: Optional[str] = None
    if pdf_sources:
        merger = PdfMerger()
        for blob in pdf_sources:
            merger.append(io.BytesIO(blob))
        output = io.BytesIO()
        merger.write(output)
        merger.close()
        combined_pdf = output.getvalue()
        if label:
            safe_label = re.sub(r"[^A-Za-z0-9._-]", "_", label.strip()) or "Backtest"
        else:
            safe_label = "Backtest"
        combined_name = f"{safe_label}_TRAC_Backtest_Chartbook_{today_str}.pdf"

    return BacktestRunOutput(
        per_ticker=per_ticker,
        portfolio=portfolio_artifacts,
        combined_pdf=combined_pdf,
        combined_name=combined_name,
        failed=failed,
    )
 
