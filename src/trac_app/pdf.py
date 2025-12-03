from __future__ import annotations

import io
import textwrap
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .charts import create_matplotlib_chart
from .processing import load_etf_block, load_ticker_block
from .utils import apply_time_range

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"
TITLE_PAGE_LOGO = ASSETS_DIR / "full_logo.png"


@lru_cache(maxsize=None)
def _load_logo_image(path: Path) -> np.ndarray:
    return plt.imread(str(path))


def _add_logo_to_figure(
    fig: plt.Figure,
    logo_path: Path,
    *,
    max_width_inches: float = 1.5,
    margin_inches: float = 0.35,
) -> None:
    if not logo_path.exists():
        return

    image = _load_logo_image(logo_path)
    if image.size == 0:
        return

    fig_width, fig_height = fig.get_size_inches()
    if fig_width <= 0 or fig_height <= 0:
        return

    logo_width_inches = min(max_width_inches, max(fig_width - 2 * margin_inches, 0))
    if logo_width_inches <= 0:
        return

    image_height, image_width = image.shape[0], image.shape[1]
    aspect_ratio = image_height / image_width if image_width else 1

    width_fraction = logo_width_inches / fig_width
    height_inches = logo_width_inches * aspect_ratio
    height_fraction = height_inches / fig_height

    margin_x_fraction = margin_inches / fig_width
    margin_y_fraction = margin_inches / fig_height
    bottom = 1.0 - margin_y_fraction - height_fraction

    if bottom < 0:
        bottom = 0

    ax = fig.add_axes(
        [margin_x_fraction, bottom, width_fraction, height_fraction], anchor="NW"
    )
    ax.imshow(image)
    ax.axis("off")
    ax.set_zorder(100)
    ax.patch.set_alpha(0)


FONT_COLOR = "#038CAD"
ANALYSIS_COLOR = "#000000"
PAGE_WIDTH_INCHES = 8.5
PAGE_HEIGHT_INCHES = 11.0
ANALYSIS_MARGIN_INCHES = 0.5
BULLET_INDENT_INCHES = 0.3
LEFT_MARGIN_X = ANALYSIS_MARGIN_INCHES / PAGE_WIDTH_INCHES
INDENT_MARGIN_X = (ANALYSIS_MARGIN_INCHES + BULLET_INDENT_INCHES) / PAGE_WIDTH_INCHES
BOTTOM_MARGIN_Y = ANALYSIS_MARGIN_INCHES / PAGE_HEIGHT_INCHES
TITLE_Y = 1 - (ANALYSIS_MARGIN_INCHES / PAGE_HEIGHT_INCHES)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.sans-serif": ["DejaVu Sans", "Nimbus Sans", "Liberation Sans"],
    }
)


def create_title_page_fig() -> plt.Figure:
    fig = plt.figure(figsize=(PAGE_WIDTH_INCHES, PAGE_HEIGHT_INCHES))
    ax = fig.add_subplot(111)
    ax.axis("off")
    fig.text(
        0.5,
        0.7,
        "TRAC Chart Book",
        fontsize=33,
        fontweight="light",
        ha="center",
        color=ANALYSIS_COLOR,
    )
    fig.text(
        0.5,
        0.62,
        "Generation PMCA",
        fontsize=17,
        fontweight="light",
        ha="center",
        color=ANALYSIS_COLOR,
    )
    _add_logo_to_figure(fig, TITLE_PAGE_LOGO, max_width_inches=1.44, margin_inches=0.5)
    return fig


def _prepare_analysis_segments(text: str, width: int = 86) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            segments.append({"blank": True})
            continue

        header = False
        if line.startswith("###"):
            header = True
            line = line[3:].strip()

        bullet = False
        if line.startswith("- "):
            bullet = True
            line = line[2:].strip()
        elif line.startswith("• "):
            bullet = True
            line = line[2:].strip()
        elif line.startswith("* "):
            bullet = True
            line = line[2:].strip()

        bold = False
        if line.startswith("**") and line.endswith("**") and len(line) > 4:
            bold = True
            line = line[2:-2].strip()
        line = line.replace("**", "")

        if (header or bold) and line.endswith(":"):
            line = line[:-1].rstrip()

        if header:
            segments.append(
                {"text": line, "header": True, "bold": True, "bullet": False, "indent": False}
            )
            continue

        wrapped = textwrap.wrap(line, width=width) or [line]
        for idx, fragment in enumerate(wrapped):
            segments.append(
                {
                    "text": fragment,
                    "header": False,
                    "bold": bold,
                    "bullet": bullet and idx == 0,
                    "indent": bullet and idx > 0,
                }
            )
    return segments


def _create_analysis_figures(analyses: list[tuple[str, str]]) -> list[plt.Figure]:
    if not analyses:
        return []

    figures: list[plt.Figure] = []

    cover = plt.figure(figsize=(PAGE_WIDTH_INCHES, PAGE_HEIGHT_INCHES))
    ax_cover = cover.add_subplot(111)
    ax_cover.axis("off")
    cover.text(
        0.5,
        0.5,
        "Analyses",
        fontsize=33,
        fontweight="light",
        ha="center",
        va="center",
        color=ANALYSIS_COLOR,
    )
    figures.append(cover)

    def _new_analysis_page(ticker: str, index: int) -> tuple[plt.Figure, float]:
        fig_page = plt.figure(figsize=(PAGE_WIDTH_INCHES, PAGE_HEIGHT_INCHES))
        ax_page = fig_page.add_subplot(111)
        ax_page.axis("off")
        suffix = " (cont.)" if index > 0 else ""
        fig_page.text(
            0.5,
            TITLE_Y,
            f"{ticker}{suffix} Synopsis",
            fontsize=18,
            fontweight="bold",
            ha="center",
            va="top",
            color=ANALYSIS_COLOR,
        )
        return fig_page, TITLE_Y - 0.06

    for ticker, text in analyses:
        segments = _prepare_analysis_segments(text)
        first_bullet_idx: Optional[int] = None
        for idx, segment in enumerate(segments):
            if segment.get("bullet"):
                first_bullet_idx = idx
                break

        if first_bullet_idx is not None:
            prior_idx = first_bullet_idx - 1
            while prior_idx >= 0 and segments[prior_idx].get("blank"):
                prior_idx -= 1

            if prior_idx >= 0:
                prior_segment = segments[prior_idx]
                if prior_segment.get("header"):
                    prior_segment["text"] = "Overview"
                    prior_segment["bold"] = True
                    prior_segment["bullet"] = False
                    prior_segment["indent"] = False
                elif prior_segment.get("text", "").strip().lower() == "overview":
                    prior_segment["header"] = True
                    prior_segment["bold"] = True
                    prior_segment["bullet"] = False
                    prior_segment["indent"] = False
                else:
                    segments.insert(
                        first_bullet_idx,
                        {
                            "text": "Overview",
                            "header": True,
                            "bold": True,
                            "bullet": False,
                            "indent": False,
                        },
                    )
            else:
                segments.insert(
                    first_bullet_idx,
                    {
                        "text": "Overview",
                        "header": True,
                        "bold": True,
                        "bullet": False,
                        "indent": False,
                    },
                )
        page_index = 0
        fig_page, cursor_y = _new_analysis_page(ticker, page_index)

        for segment in segments:
            if segment.get("blank"):
                next_cursor = cursor_y - 0.024
                if next_cursor < BOTTOM_MARGIN_Y:
                    figures.append(fig_page)
                    page_index += 1
                    fig_page, cursor_y = _new_analysis_page(ticker, page_index)
                    continue
                cursor_y = next_cursor
                continue

            is_header = bool(segment.get("header"))
            line_height = 0.042 if is_header else 0.028
            if cursor_y - line_height < BOTTOM_MARGIN_Y:
                figures.append(fig_page)
                page_index += 1
                fig_page, cursor_y = _new_analysis_page(ticker, page_index)

            if is_header:
                fig_page.text(
                    LEFT_MARGIN_X,
                    cursor_y,
                    segment["text"],
                    fontsize=13,
                    fontweight="bold",
                    ha="left",
                    va="top",
                    color=ANALYSIS_COLOR,
                )
                cursor_y -= line_height
                continue

            text_value = str(segment.get("text", ""))
            x_position = INDENT_MARGIN_X if segment.get("indent") else LEFT_MARGIN_X
            if segment.get("bullet"):
                text_value = f"• {text_value}"
            font_weight = "bold" if segment.get("bold") else "normal"

            fig_page.text(
                x_position,
                cursor_y,
                text_value,
                fontsize=10,
                fontweight=font_weight,
                ha="left",
                va="top",
                color=ANALYSIS_COLOR,
            )
            cursor_y -= line_height

        figures.append(fig_page)

    return figures


def build_pdf_report(
    stock_tickers: list[str],
    etf_tickers: list[str],
    time_range: str,
    band_base: float,
    corridor: float,
    num_bands: int,
    y_mode: str,
    analysis_fetcher: Optional[Callable[[str], str]] = None,
) -> tuple[Optional[bytes], list[str]]:
    notes: list[str] = []
    stock_figs: list[tuple[str, plt.Figure]] = []
    etf_figs: list[tuple[str, plt.Figure]] = []
    for ticker in stock_tickers:
        data = load_ticker_block(ticker)
        if data is None:
            notes.append(f"{ticker}: no stock data available.")
            continue

        filtered = apply_time_range(data, time_range)
        price = filtered["price"].dropna()
        if price.empty:
            notes.append(f"{ticker}: insufficient stock price data for selected range.")
            continue

        fig = create_matplotlib_chart(
            ticker, filtered, band_base, corridor, num_bands, y_mode
        )
        if fig is None:
            notes.append(f"{ticker}: failed to render stock chart.")
            continue

        stock_figs.append((ticker, fig))

    for ticker in etf_tickers:
        data = load_etf_block(ticker)
        if data is None:
            notes.append(f"{ticker}: no ETF data available.")
            continue

        filtered = apply_time_range(data, time_range)
        price = filtered["price"].dropna()
        if price.empty:
            notes.append(f"{ticker}: insufficient ETF price data for selected range.")
            continue

        fig = create_matplotlib_chart(
            ticker, filtered, band_base, corridor, num_bands, y_mode
        )
        if fig is None:
            notes.append(f"{ticker}: failed to render ETF chart.")
            continue

        etf_figs.append((ticker, fig))

    if not stock_figs and not etf_figs:
        return None, notes

    analysis_results: list[tuple[str, str]] = []
    if analysis_fetcher:
        seen: set[str] = set()
        for ticker in [*stock_tickers, *etf_tickers]:
            ticker_key = ticker.upper()
            if ticker_key in seen:
                continue
            seen.add(ticker_key)
            try:
                response_text = analysis_fetcher(ticker_key)
            except Exception as exc:
                notes.append(f"{ticker_key}: AI analysis unavailable ({exc}).")
                continue
            if response_text:
                analysis_results.append((ticker_key, response_text))
            else:
                notes.append(f"{ticker_key}: AI analysis returned no content.")

    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        title_fig = create_title_page_fig()
        pdf.savefig(title_fig, bbox_inches="tight")
        plt.close(title_fig)

        for ticker, fig in stock_figs:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        for ticker, fig in etf_figs:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        for analysis_fig in _create_analysis_figures(analysis_results):
            pdf.savefig(analysis_fig, bbox_inches="tight")
            plt.close(analysis_fig)

    buffer.seek(0)
    return buffer.getvalue(), notes
