from __future__ import annotations

import csv
from datetime import date as _dt_date
from io import StringIO
from pathlib import Path
from typing import Optional

import streamlit as st

from trac_app.charts import create_matplotlib_chart, figure_to_png_bytes, make_chart
from trac_app.config import DEF_BAND_BASE, DEF_CORRIDOR, DEF_NUM_BANDS, TIME_RANGES, THEME_CSS
from trac_app.groups import delete_group, load_groups, upsert_group
from trac_app.pdf import build_pdf_report
from trac_app.processing import (
    get_etf_list,
    get_ticker_list,
    load_etf_block,
    load_ticker_block,
)
from trac_app.backtest_runner import BacktestParameters, run_backtests
from trac_app.utils import extract_region
from trac_app.oai import OpenAIIntegrationError, fetch_analysis


def _get_group_by_name(groups: list, name: str):
    return next((g for g in groups if g.name == name), None)


def _close_groups_modal() -> None:
    st.session_state["groups_modal_open"] = False
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if callable(rerun):
        rerun()


def _render_groups_modal_body(all_available_tickers: list[str], *, inline: bool = False) -> None:
    stored_groups = load_groups()
    group_names = [group.name for group in stored_groups]
    group_choices = ["Create new"] + group_names

    upload_state_key = "group_upload_info_inline" if inline else "group_upload_info"
    st.session_state.setdefault(upload_state_key, None)

    select_key = "group_editor_select_inline" if inline else "group_editor_select"
    name_key = "group_manager_name_inline" if inline else "group_manager_name"
    members_key = "group_manager_members_inline" if inline else "group_manager_members"
    close_key = "group_close_inline" if inline else "group_close"
    delete_key = "group_delete_inline" if inline else "group_delete"
    save_key = "group_save_inline" if inline else "group_save"

    st.session_state.setdefault(select_key, "Create new")
    pending_select = st.session_state.pop("group_editor_select_pending", None)
    if pending_select:
        st.session_state[select_key] = pending_select
    if st.session_state[select_key] not in group_choices:
        st.session_state[select_key] = "Create new"

    st.session_state.setdefault(name_key, "")
    pending_name = st.session_state.pop("group_manager_name_pending", None)
    if pending_name is not None:
        st.session_state[name_key] = pending_name

    st.session_state.setdefault(members_key, [])
    pending_members = st.session_state.pop("group_manager_members_pending", None)
    if pending_members is not None:
        st.session_state[members_key] = pending_members

    filtered_members = [
        t for t in st.session_state.get(members_key, []) if t in all_available_tickers
    ]
    if filtered_members != st.session_state.get(members_key, []):
        st.session_state[members_key] = filtered_members

    st.markdown("### Ticker groups")
    selected_group_option = st.selectbox(
        "Edit or create a saved group",
        group_choices,
        key=select_key,
    )

    previous_key = "group_editor_selected_last_inline" if inline else "group_editor_selected_last"
    previous_option = st.session_state.get(previous_key)
    missing_for_group: list[str] = []
    current_target = _get_group_by_name(stored_groups, selected_group_option)
    if selected_group_option != previous_option:
        if current_target:
            available_set = set(all_available_tickers)
            st.session_state[name_key] = current_target.name
            st.session_state[members_key] = [
                t for t in current_target.tickers if t in available_set
            ]
            missing_for_group = [
                t for t in current_target.tickers if t not in available_set
            ]
        else:
            st.session_state[name_key] = ""
            st.session_state[members_key] = []
    else:
        if current_target:
            missing_for_group = [
                t for t in current_target.tickers if t not in all_available_tickers
            ]
    st.session_state[previous_key] = selected_group_option

    group_name = st.text_input(
        "Group name",
        key=name_key,
        placeholder="Enter a group name",
    )
    group_tickers = st.multiselect(
        "Tickers in this group",
        all_available_tickers,
        key=members_key,
        placeholder="Select tickers to include",
    )

    if missing_for_group:
        st.warning(
            f"The saved group includes tickers that are unavailable: {', '.join(sorted(missing_for_group))}"
        )

    save_disabled = not group_name.strip() or not group_tickers
    save_col, delete_col, close_col = st.columns([1, 1, 1])
    with save_col:
        if st.button(
            "Save group",
            use_container_width=True,
            disabled=save_disabled,
            key=save_key,
        ):
            try:
                upsert_group(group_name, group_tickers)
            except ValueError as exc:
                st.warning(str(exc))
            else:
                trimmed = group_name.strip()
                st.session_state["group_editor_select_pending"] = trimmed
                st.session_state["group_manager_name_pending"] = trimmed
                st.session_state["group_manager_members_pending"] = list(dict.fromkeys(group_tickers))
                st.success(f"Saved group '{trimmed}'.")
    with delete_col:
        delete_disabled = selected_group_option == "Create new"
        if st.button(
            "Delete group",
            use_container_width=True,
            disabled=delete_disabled,
            key=delete_key,
        ):
            if not delete_disabled:
                removed = delete_group(selected_group_option)
                if removed:
                    st.session_state["group_editor_select_pending"] = "Create new"
                    st.session_state["group_manager_name_pending"] = ""
                    st.session_state["group_manager_members_pending"] = []
                    st.success(f"Deleted group '{selected_group_option}'.")
                else:
                    st.warning("Unable to remove group (it may have been deleted already).")
    with close_col:
        if st.button("Close", use_container_width=True, key=close_key):
            _close_groups_modal()
    if st.session_state.get("groups_modal_open", False) and inline:
        st.caption("Close to return to the dashboard.")

    st.markdown("### Upload group from CSV")
    upload_key = f"group_csv_upload_{'inline' if inline else 'modal'}"
    uploaded_file = st.file_uploader(
        "One ticker per row (no header)",
        type="csv",
        key=upload_key,
        help="Upload a CSV to create or overwrite a saved group.",
    )

    if uploaded_file is not None:
        try:
            info = _parse_group_csv(uploaded_file, all_available_tickers)
        except ValueError as exc:
            st.warning(str(exc))
        else:
            st.session_state[upload_state_key] = info

    upload_info = st.session_state.get(upload_state_key)
    if upload_info:
        st.caption(f"Proposed group name: **{upload_info['name']}**")
        st.caption(f"Tickers in file: {len(upload_info['raw'])}")
        st.caption(f"Matched tickers: {len(upload_info['tickers'])}")
        if upload_info["missing"]:
            preview = ", ".join(upload_info["missing"][:10])
            suffix = " …" if len(upload_info["missing"]) > 10 else ""
            st.warning(f"Excluded tickers not found in database: {preview}{suffix}")

        existing = _get_group_by_name(stored_groups, upload_info["name"])
        confirm_label = "Overwrite group" if existing else "Create group"
        confirm_disabled = len(upload_info["tickers"]) == 0
        if confirm_disabled:
            st.info("No valid tickers matched this CSV; adjust the file to proceed.")
        elif existing:
            st.info(
                f"Group '{upload_info['name']}' already exists and will be replaced if you proceed."
            )

        confirm_key = f"confirm_group_upload_{upload_state_key}"
        cancel_key = f"cancel_group_upload_{upload_state_key}"

        if st.button(confirm_label, key=confirm_key, disabled=confirm_disabled):
            try:
                upsert_group(upload_info["name"], upload_info["tickers"])
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.success(
                    f"Saved group '{upload_info['name']}' with {len(upload_info['tickers'])} tickers."
                )
                st.session_state[upload_state_key] = None
                st.session_state["group_editor_select_pending"] = upload_info["name"]
                st.session_state["group_manager_name_pending"] = upload_info["name"]
                st.session_state["group_manager_members_pending"] = upload_info["tickers"]
                rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
                if callable(rerun):
                    rerun()

        if st.button("Cancel upload", key=cancel_key):
            st.session_state[upload_state_key] = None
            rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
            if callable(rerun):
                rerun()



def _render_groups_modal(all_available_tickers: list[str]) -> None:
    if hasattr(st, "dialog"):
        @st.dialog("Manage ticker groups")
        def _show_dialog() -> None:
            _render_groups_modal_body(all_available_tickers)
        _show_dialog()
        return

    if hasattr(st, "experimental_dialog"):
        @st.experimental_dialog("Manage ticker groups")  # type: ignore[attr-defined]
        def _show_dialog_exp() -> None:
            _render_groups_modal_body(all_available_tickers)
        _show_dialog_exp()
        return

    # Fallback inline container
    fallback_container = st.container()
    with fallback_container:
        st.markdown("<div class='group-modal-fallback'>", unsafe_allow_html=True)
        st.info("Streamlit version does not support modals; showing inline editor instead.")
        _render_groups_modal_body(all_available_tickers, inline=True)
        st.markdown("</div>", unsafe_allow_html=True)


def _parse_group_csv(uploaded_file, available_tickers: list[str]):
    try:
        raw_bytes = uploaded_file.getvalue()
    except Exception as exc:  # pragma: no cover - depends on Streamlit internals
        raise ValueError(f"Unable to read uploaded file: {exc}") from exc

    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - rare encoding fallback
        text = raw_bytes.decode(errors="ignore")

    reader = csv.reader(StringIO(text))
    raw_symbols: list[str] = []
    for row in reader:
        if not row:
            continue
        symbol = str(row[0]).strip()
        if symbol:
            raw_symbols.append(symbol)

    if not raw_symbols:
        raise ValueError("No ticker symbols found in the uploaded CSV.")

    unique_symbols = list(dict.fromkeys(raw_symbols))
    available_set = set(available_tickers)
    matched = [sym for sym in unique_symbols if sym in available_set]
    missing = [sym for sym in unique_symbols if sym not in available_set]

    name = Path(uploaded_file.name).stem.strip() or "Uploaded Group"
    token = (uploaded_file.name, len(raw_bytes))
    return {
        "name": name,
        "tickers": matched,
        "missing": missing,
        "raw": unique_symbols,
        "token": token,
    }


GROUP_MODAL_CSS = """
<style>
[data-testid="stDialog"] {
    background-color: rgba(3, 12, 25, 0.75) !important;
}
[data-testid="stDialog"] > div:nth-child(2) {
    background-color: transparent !important;
    box-shadow: none !important;
}
[data-testid="stDialog"] > div:nth-child(2) > div {
    background-color: #061225 !important;
    border-radius: 14px !important;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45) !important;
    padding: 1.5rem 1.75rem !important;
    color: #f0f4ff !important;
}
[data-testid="stDialog"] h1,
[data-testid="stDialog"] h2,
[data-testid="stDialog"] h3,
[data-testid="stDialog"] label,
[data-testid="stDialog"] span,
[data-testid="stDialog"] p,
[data-testid="stDialog"] [data-testid="stMarkdownContainer"] p,
[data-testid="stDialog"] label[data-testid="stWidgetLabel"] div[data-testid="stMarkdownContainer"] p {
    color: #0b1f33 !important;
}
[data-testid="stDialog"] div[data-baseweb="select"] > div,
[data-testid="stDialog"] input,
[data-testid="stDialog"] textarea {
    background-color: rgba(8, 24, 40, 0.85) !important;
    border: 1px solid rgba(95, 160, 255, 0.5) !important;
    color: #f0f4ff !important;
}
[data-testid="stDialog"] div[data-baseweb="select"] span,
[data-testid="stDialog"] div[data-baseweb="select"] p {
    color: #f0f4ff !important;
}
[data-testid="stDialog"] div[data-baseweb="select"] div {
    color: #f0f4ff !important;
}
[data-testid="stDialog"] div[data-baseweb="select"] svg,
[data-testid="stDialog"] svg {
    fill: #5FA0FF !important;
    color: #5FA0FF !important;
}
[data-testid="stDialog"] .stButton button,
[data-testid="stDialog"] .stDownloadButton button {
    background-color: var(--trac-gold) !important;
    color: #ffffff !important;
    border-radius: 6px !important;
    border: none !important;
}
[data-testid="stDialog"] .stButton button:hover,
[data-testid="stDialog"] .stDownloadButton button:hover {
    background-color: rgba(199, 171, 102, 0.85) !important;
}
[data-testid="stDialog"] div[data-testid="stFileUploader"] > div:first-child {
    background-color: rgba(8, 24, 40, 0.85) !important;
    border: 1px solid rgba(95, 160, 255, 0.5) !important;
    border-radius: 8px !important;
    padding: 0.75rem !important;
}
[data-testid="stDialog"] section[data-testid="stFileUploadDropzone"] {
    background-color: rgba(8, 24, 40, 0.85) !important;
    border: 1px dashed #5FA0FF !important;
    border-radius: 6px !important;
    color: #f0f4ff !important;
    opacity: 1 !important;
}
[data-testid="stDialog"] section[data-testid="stFileUploadDropzone"] * {
    color: #f0f4ff !important;
    opacity: 1 !important;
}
[data-testid="stDialog"] section[data-testid="stFileUploadDropzone"] div[role="button"] {
    background: transparent !important;
    border: 1px solid #5FA0FF55 !important;
    color: #f0f4ff !important;
}
[data-testid="stDialog"] section[data-testid="stFileUploadDropzone"] span,
[data-testid="stDialog"] section[data-testid="stFileUploadDropzone"] p {
    color: #dbe7ff !important;
}
[data-testid="stDialog"] input::placeholder,
[data-testid="stDialog"] textarea::placeholder {
    color: rgba(240, 244, 255, 0.7) !important;
}
.group-modal-fallback {
    background-color: #061225;
    padding: 1.25rem;
    border-radius: 12px;
}
</style>
"""


st.set_page_config(page_title="TRAC Suite", layout="wide")


st.markdown(THEME_CSS, unsafe_allow_html=True)
st.markdown(GROUP_MODAL_CSS, unsafe_allow_html=True)
st.title("TRAC Suite")

try:
    tickers = get_ticker_list()
except ConnectionError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # pragma: no cover - surfaced to user
    st.error(f"Unexpected error while loading tickers: {exc}")
    st.stop()

if not tickers:
    st.error("No tickers available from the database.")
    st.stop()

try:
    etf_tickers = get_etf_list()
except ConnectionError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # pragma: no cover - surfaced to user
    st.error(f"Unexpected error while loading ETFs: {exc}")
    st.stop()

if not etf_tickers:
    st.warning("No ETFs available from the database.")
    etf_tickers = []

if "report_pdf" not in st.session_state:
    st.session_state["report_pdf"] = None
    st.session_state["report_notes"] = []
    st.session_state["report_params"] = None
if "ai_analysis_cache" not in st.session_state:
    st.session_state["ai_analysis_cache"] = {}

regions = sorted({extract_region(t) for t in tickers})
if not regions:
    st.error("No regions could be derived from ticker symbols.")
    st.stop()
default_region = "US" if "US" in regions else regions[0]

with st.sidebar:
    security_type = st.selectbox("Security type", ["Stocks", "ETFs"], index=0)

    if security_type == "Stocks":
        region_index = regions.index(default_region) if default_region in regions else 0
        selected_region = st.selectbox("Region", regions, index=region_index)
        filtered_tickers = [
            t for t in tickers if extract_region(t) == selected_region
        ]
        if not filtered_tickers:
            st.warning(f"No tickers found for region {selected_region}.")
            st.stop()
        ticker_label = "Ticker"
    else:
        selected_region = None
        filtered_tickers = sorted(etf_tickers)
        if not filtered_tickers:
            st.warning("No ETFs available to display.")
            st.stop()
        ticker_label = "ETF"

    sel = st.selectbox(ticker_label, filtered_tickers, index=0)

    band_base = DEF_BAND_BASE
    corridor = DEF_CORRIDOR
    st.caption(f"Band base fixed at {band_base:.6f}; corridor ±{corridor:.3f}.")
    num_bands = st.slider("Number of band levels each side", 1, 20, DEF_NUM_BANDS)
    y_mode = st.radio(
        "Y-range mode", ["Price min/max ±20%", "Last price ±20%"], index=0
    )

    all_available_tickers = sorted(set(tickers) | set(etf_tickers))

    st.session_state.setdefault("groups_modal_open", False)
    if st.button("Groups", use_container_width=True):
        st.session_state["groups_modal_open"] = True

if st.session_state.get("groups_modal_open"):
    _render_groups_modal(all_available_tickers)

tab_chart, tab_report, tab_backtest = st.tabs(["Interactive Chart", "Create a Report", "Backtesting"])

with tab_chart:
    with st.spinner("Loading data..."):
        try:
            if security_type == "Stocks":
                block = load_ticker_block(sel)
            else:
                block = load_etf_block(sel)
        except ConnectionError as exc:
            st.error(str(exc))
            block = None
        except Exception as exc:  # pragma: no cover - surfaced to user
            st.error(f"Unexpected error while loading data: {exc}")
            block = None

    if block is None:
        st.warning("No valid data returned for that selection.")
    else:
        st.subheader(f"{sel} ({security_type[:-1]})")
        fig = make_chart(block, band_base, corridor, num_bands, y_mode)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

        png_bytes = None
        try:
            mpl_fig = create_matplotlib_chart(
                sel, block, band_base, corridor, num_bands, y_mode
            )
            if mpl_fig:
                png_bytes = figure_to_png_bytes(mpl_fig)
        except Exception:
            png_bytes = None

        col_left, _, col_right = st.columns([1, 0.15, 1])
        with col_left:
            if png_bytes:
                st.download_button(
                    "Download Chart",
                    data=png_bytes,
                    file_name=f"{sel}_TRAC.png",
                    mime="image/png",
                )
            else:
                st.caption(
                    "PNG export unavailable for this selection (requires sufficient positive data)."
                )

        analysis_cache = st.session_state.setdefault("ai_analysis_cache", {})
        with col_right:
            trigger_analysis = st.button("Generate AI Analysis")
        if trigger_analysis:
            with st.spinner("Fetching..."):
                try:
                    result_text = fetch_analysis(sel)
                except OpenAIIntegrationError as exc:
                    st.error(str(exc))
                except Exception as exc:  # pragma: no cover - surfaced to user
                    st.error(f"Unexpected error while requesting AI analysis: {exc}")
                else:
                    analysis_cache[sel] = result_text
                    st.session_state["ai_analysis_cache"] = analysis_cache

        latest_analysis = st.session_state.get("ai_analysis_cache", {}).get(sel)
        if latest_analysis:
            st.markdown("**AI Analysis**")
            st.markdown(latest_analysis)

with tab_report:
    st.write(
        "Select one or more securities, choose a time range, and generate a PDF "
        "containing TRAC charts that mirror the interactive view."
    )

    label_list = list(TIME_RANGES.keys())
    default_range_index = label_list.index("all time") if "all time" in label_list else len(label_list) - 1
    selected_range = st.radio(
        "Time range",
        label_list,
        index=default_range_index,
        horizontal=True,
    )
    stock_options_for_report = sorted(
        [t for t in tickers if selected_region and extract_region(t) == selected_region]
    ) if selected_region else sorted(tickers)
    etf_options_for_report = sorted(etf_tickers)

    stock_default = [sel] if (security_type == "Stocks" and sel in stock_options_for_report) else []
    etf_default = [sel] if (security_type == "ETFs" and sel in etf_options_for_report) else []

    if "stock_report_selection" not in st.session_state:
        st.session_state["stock_report_selection"] = stock_default
    if "etf_report_selection" not in st.session_state:
        st.session_state["etf_report_selection"] = etf_default

    current_stock_selection = st.session_state.get("stock_report_selection", [])
    filtered_stock_selection = [
        t for t in current_stock_selection if t in stock_options_for_report
    ]
    if filtered_stock_selection != current_stock_selection:
        st.session_state["stock_report_selection"] = filtered_stock_selection

    current_etf_selection = st.session_state.get("etf_report_selection", [])
    filtered_etf_selection = [
        t for t in current_etf_selection if t in etf_options_for_report
    ]
    if filtered_etf_selection != current_etf_selection:
        st.session_state["etf_report_selection"] = filtered_etf_selection

    saved_groups_for_report = load_groups()
    group_names_for_report = [group.name for group in saved_groups_for_report]
    if group_names_for_report:
        st.markdown("**Saved groups**")
        group_choices = ["None"] + group_names_for_report
        pending_group_selection = st.session_state.pop("report_group_selection_pending", None)
        st.session_state.setdefault("report_group_selection", "None")
        if pending_group_selection is not None:
            if pending_group_selection in group_choices:
                st.session_state["report_group_selection"] = pending_group_selection
            else:
                st.session_state["report_group_selection"] = "None"
        if st.session_state["report_group_selection"] not in group_choices:
            st.session_state["report_group_selection"] = "None"
        selected_report_group = st.selectbox(
            "Load tickers from a saved group",
            group_choices,
            key="report_group_selection",
        )
        group_action_cols = st.columns([1, 1, 2])
        load_group_button = group_action_cols[0].button(
            "Load group",
            use_container_width=True,
            disabled=selected_report_group == "None",
        )
        clear_group_button = group_action_cols[1].button(
            "Clear selection",
            use_container_width=True,
        )
        if load_group_button and selected_report_group != "None":
            target_group = next(
                (g for g in saved_groups_for_report if g.name == selected_report_group),
                None,
            )
            if target_group:
                group_stock_members = [
                    t for t in target_group.tickers if t in stock_options_for_report
                ]
                missing_members = [
                    t
                    for t in target_group.tickers
                    if t not in stock_options_for_report and t not in etf_options_for_report
                ]
                st.session_state["stock_report_selection"] = group_stock_members
                st.session_state["report_group_selection_pending"] = selected_report_group
                if missing_members:
                    st.warning(
                        f"Excluded unavailable tickers: {', '.join(sorted(missing_members))}."
                    )
                st.success(f"Loaded group '{selected_report_group}' into stock selection.")
                if any(t in etf_options_for_report for t in target_group.tickers):
                    st.info("ETFs from this group are not loaded automatically; adjust the ETF list manually if needed.")
        if clear_group_button:
            st.session_state["stock_report_selection"] = []
            st.session_state["etf_report_selection"] = []
            st.session_state["report_group_selection_pending"] = "None"

    stock_report_tickers = st.multiselect(
        "Add stock tickers",
        stock_options_for_report,
        key="stock_report_selection",
        help="Start typing to search and add more tickers.",
    )
    etf_report_tickers = st.multiselect(
        "Add ETFs",
        etf_options_for_report,
        key="etf_report_selection",
        help="Start typing to search and add more ETFs.",
    )
    include_ai_analyses = st.checkbox(
        "Include AI analyses in report",
        value=False,
        help="Uses AI to generate key info and moves on each security.",
    )

    current_params = (
        selected_region,
        tuple(stock_report_tickers),
        tuple(etf_report_tickers),
        selected_range,
        float(band_base),
        float(corridor),
        int(num_bands),
        y_mode,
        include_ai_analyses,
    )

    if st.session_state.get("report_params") != current_params:
        st.session_state["report_pdf"] = None
        st.session_state["report_notes"] = []
        st.session_state["report_params"] = current_params

    generate = st.button(
        "Generate PDF",
        type="primary",
        use_container_width=False,
        disabled=not stock_report_tickers and not etf_report_tickers,
    )

    if generate and (stock_report_tickers or etf_report_tickers):
        with st.spinner("Building PDF report..."):
            pdf_bytes, notes = build_pdf_report(
                list(stock_report_tickers),
                list(etf_report_tickers),
                selected_range,
                float(band_base),
                float(corridor),
                int(num_bands),
                y_mode,
                analysis_fetcher=fetch_analysis if include_ai_analyses else None,
            )
        st.session_state["report_notes"] = notes
        if pdf_bytes:
            st.session_state["report_pdf"] = pdf_bytes
            st.success("Report ready. Use the download button below.")
        else:
            st.session_state["report_pdf"] = None
            st.error("No charts could be generated. Review the notes below.")

    if st.session_state.get("report_pdf"):
        st.download_button(
            "Download PDF report",
            data=st.session_state["report_pdf"],
            file_name="TRAC_report.pdf",
            mime="application/pdf",
        )

    if st.session_state.get("report_notes"):
        notes_list = st.session_state["report_notes"]
        if notes_list:
            st.markdown("**Notes**")
            for note in notes_list:
                st.markdown(f"- {note}")

with tab_backtest:
    st.write(
        "Run the TRAC backtesting engine on one or more tickers that you select manually "
        "or from a saved group, then download the resulting PDFs and CSVs."
    )

    stock_universe = sorted(tickers)
    if not stock_universe:
        st.warning("No stock tickers are available from the database.")
    else:
        stored_groups = load_groups()

        if st.session_state.get("backtest_selection_mode") == "Single ticker":
            st.session_state["backtest_selection_mode"] = "Manual selection"
        st.session_state.setdefault("backtest_selection_mode", "Manual selection")
        selection_mode = st.radio(
            "Selection mode",
            ["Manual selection", "Saved group"],
            key="backtest_selection_mode",
            help="Backtests currently support stock tickers pulled from the database.",
        )

        selected_tickers: list[str] = []
        missing_group_members: list[str] = []
        run_label: Optional[str] = None

        if selection_mode == "Manual selection":
            manual_key = "backtest_manual_tickers"
            if manual_key not in st.session_state:
                st.session_state[manual_key] = [sel] if sel in stock_universe else []

            manual_selection = st.session_state.get(manual_key, [])
            filtered_manual = [t for t in manual_selection if t in stock_universe]
            if filtered_manual != manual_selection:
                st.session_state[manual_key] = filtered_manual

            if not st.session_state[manual_key] and sel in stock_universe:
                st.session_state[manual_key] = [sel]

            chosen_tickers = st.multiselect(
                "Tickers",
                stock_universe,
                key=manual_key,
                help="Start typing to search and add tickers for this backtest run.",
            )
            if chosen_tickers:
                selected_tickers = list(chosen_tickers)
                run_label = chosen_tickers[0] if len(chosen_tickers) == 1 else None
        elif selection_mode == "Saved group":
            group_names = [g.name for g in stored_groups]
            if not group_names:
                st.info("No saved groups found. Use the Groups button to create one.")
            else:
                st.session_state.setdefault("backtest_group_name", group_names[0])
                selected_group_name = st.selectbox(
                    "Saved group",
                    group_names,
                    key="backtest_group_name",
                )
                group_obj = _get_group_by_name(stored_groups, selected_group_name)
                if group_obj:
                    selected_tickers = [t for t in group_obj.tickers if t in stock_universe]
                    missing_group_members = [t for t in group_obj.tickers if t not in stock_universe]
                    run_label = selected_group_name
                    if missing_group_members:
                        st.warning(
                            "Excluded unavailable tickers from this group: "
                            + ", ".join(sorted(missing_group_members))
                        )
        rule_options = {
            "revised_exit": "Core BOB Strategy (revised exits)",
            "rule1": "Rule 1 (targets only)",
            "rule1_stop": "Rule 1 with stop",
        }
        rule_choice = st.selectbox(
            "Backtest rule",
            options=list(rule_options.keys()),
            format_func=lambda key: rule_options.get(key, key),
            key="backtest_rule",
        )

        weighting_labels = {"equal": "Equal weight", "undervalued": "Discount-to-value"}
        weighting_choice = st.selectbox(
            "Weighting scheme",
            options=list(weighting_labels.keys()),
            format_func=lambda key: weighting_labels.get(key, key),
            help="Equal weight treats all holdings the same; discount-to-value tilts toward larger FMV discounts.",
            key="backtest_weighting",
        )

        st.markdown("#### Strategy parameters")
        param_col_left, param_col_right = st.columns(2)
        with param_col_left:
            band_width_pct = st.slider(
                "Band width (±%)",
                min_value=2.0,
                max_value=15.0,
                value=6.4,
                step=0.1,
                key="backtest_band_width",
                help="Controls the thickness of each band corridor around book value.",
            )
            length_labels = {
                None: "Full history",
                5: "Last 5 years",
                10: "Last 10 years",
                20: "Last 20 years",
            }
            backtest_length_years = st.selectbox(
                "Backtest length",
                options=list(length_labels.keys()),
                format_func=lambda key: length_labels[key],
                key="backtest_length",
                help="Limit the analysis window to a recent period if desired.",
            )
        with param_col_right:
            buy_day_count = st.slider(
                "Buy signal confirmation days",
                min_value=1,
                max_value=10,
                value=2,
                key="backtest_buy_days",
                help="Number of consecutive closes above the breakout band required before buying.",
            )
            ceiling_day_count = st.slider(
                "Ceiling sell confirmation days",
                min_value=1,
                max_value=15,
                value=5,
                key="backtest_ceiling_days",
                help="Consecutive closes below the active ceiling before triggering a ceiling exit.",
            )
            stop_day_count = st.slider(
                "Stop sell confirmation days",
                min_value=1,
                max_value=15,
                value=6,
                key="backtest_stop_days",
                help="Consecutive closes below the entry band lower edge before a stop exit.",
            )

        params = BacktestParameters(
            band_tolerance=band_width_pct / 100.0,
            buy_signal_days=buy_day_count,
            ceiling_sell_days=ceiling_day_count,
            stop_sell_days=stop_day_count,
            backtest_years=backtest_length_years,
        ).normalized()

        include_portfolio = st.checkbox(
            "Generate portfolio summary outputs",
            value=True,
            help="Creates combined PDFs/CSVs when more than one ticker is included.",
            key="backtest_include_portfolio",
        )

        run_disabled = len(selected_tickers) == 0
        run_button = st.button(
            "Run backtest",
            type="primary",
            disabled=run_disabled,
            help=None if selected_tickers else "Select at least one ticker to enable the backtest.",
        )

        if run_button and selected_tickers:
            try:
                with st.spinner("Running backtests..."):
                    output = run_backtests(
                        selected_tickers,
                        rule=rule_choice,
                        weighting=weighting_choice,
                        generate_portfolio=include_portfolio and len(selected_tickers) > 1,
                        label=run_label,
                        params=params,
                    )
            except Exception as exc:  # pragma: no cover - surfaced to user
                st.error(f"Backtest failed: {exc}")
            else:
                st.session_state["backtest_result"] = {
                    "tickers": selected_tickers,
                    "rule": rule_choice,
                    "weighting": weighting_choice,
                    "output": output,
                    "label": run_label,
                    "params": params,
                }
                if output.per_ticker:
                    st.success("Backtest completed.")
                else:
                    st.warning("No valid results were generated for the selected tickers.")

        stored_run = st.session_state.get("backtest_result")
        if stored_run:
            output = stored_run.get("output")
            tickers_run = stored_run.get("tickers", [])
            rule_run = stored_run.get("rule")
            weighting_run = stored_run.get("weighting")
            label_run = stored_run.get("label")
            params_run = stored_run.get("params")

            if output:
                tickers_label = ", ".join(tickers_run) if tickers_run else "(none)"
                st.markdown(
                    f"**Last run** — Tickers: {tickers_label} | Rule: {rule_options.get(rule_run, rule_run)} | "
                    f"Weighting: {weighting_run}"
                )
                if isinstance(params_run, BacktestParameters):
                    length_desc = (
                        "Full history"
                        if not params_run.backtest_years
                        else f"Last {params_run.backtest_years} years"
                    )
                    st.caption(
                        f"Band width ±{params_run.band_tolerance*100:.1f}% | "
                        f"Buy {params_run.buy_signal_days}d | "
                        f"Ceiling {params_run.ceiling_sell_days}d | "
                        f"Stop {params_run.stop_sell_days}d | "
                        f"{length_desc}"
                    )

                if output.failed:
                    st.warning(
                        "No database data was found for: " + ", ".join(sorted(output.failed))
                    )

                if output.combined_pdf:
                    combined_name = output.combined_name
                    if not combined_name:
                        base_label = label_run or "Backtest"
                        safe_base = "".join(
                            ch if ch.isalnum() or ch in "._-" else "_" for ch in base_label
                        )
                        combined_name = f"{safe_base}_TRAC_Backtest_Chartbook_{_dt_date.today().strftime('%Y%m%d')}.pdf"
                    st.download_button(
                        "Download combined PDF",
                        data=output.combined_pdf,
                        file_name=combined_name,
                        mime="application/pdf",
                        key="backtest_combined_pdf",
                    )

                portfolio = output.portfolio
                if portfolio:
                    st.markdown("### Portfolio Summary")
                    final_value = float(portfolio.equity.iloc[-1]) if len(portfolio.equity) else 100.0
                    st.caption(f"Equity curve ending value: ${final_value:,.2f}")

                    col_port_a, col_port_b, col_port_c = st.columns(3)
                    if portfolio.equity_pdf:
                        col_port_a.download_button(
                            "Portfolio PDF",
                            data=portfolio.equity_pdf,
                            file_name="portfolio_equity.pdf",
                            mime="application/pdf",
                            key="backtest_portfolio_pdf",
                        )
                    if portfolio.stats_pdf:
                        col_port_b.download_button(
                            "Stats PDF",
                            data=portfolio.stats_pdf,
                            file_name="portfolio_stats.pdf",
                            mime="application/pdf",
                            key="backtest_stats_pdf",
                        )
                    if portfolio.daily_report_pdf:
                        col_port_c.download_button(
                            "Daily Report PDF",
                            data=portfolio.daily_report_pdf,
                            file_name="portfolio_daily.pdf",
                            mime="application/pdf",
                            key="backtest_daily_pdf",
                        )

                    col_port_csv1, col_port_csv2, col_port_csv3 = st.columns(3)
                    if portfolio.daily_holdings_csv:
                        col_port_csv1.download_button(
                            "Daily Holdings CSV",
                            data=portfolio.daily_holdings_csv,
                            file_name="portfolio_daily_holdings.csv",
                            mime="text/csv",
                            key="backtest_holdings_csv",
                        )
                    if portfolio.equity_csv:
                        col_port_csv2.download_button(
                            "Equity Curve CSV",
                            data=portfolio.equity_csv,
                            file_name="portfolio_equity.csv",
                            mime="text/csv",
                            key="backtest_equity_csv",
                        )
                    if portfolio.trades_csv:
                        col_port_csv3.download_button(
                            "Trades CSV",
                            data=portfolio.trades_csv,
                            file_name="portfolio_trades.csv",
                            mime="text/csv",
                            key="backtest_trades_csv",
                        )

                    if portfolio.summary_csv:
                        st.download_button(
                            "Per-ticker summary CSV",
                            data=portfolio.summary_csv,
                            file_name="portfolio_summary.csv",
                            mime="text/csv",
                            key="backtest_summary_csv",
                        )

                for ticker in tickers_run:
                    artifacts = output.per_ticker.get(ticker)
                    if not artifacts:
                        continue

                    safe_ticker = ticker.replace(":", "_").replace("/", "_")
                    result = artifacts.result

                    st.markdown(f"### {ticker}")

                    strat = result.strat_rf
                    if strat.get("HasData", True):
                        st.caption(
                            f"BOB strategy: {strat['TotalReturn']*100:+.1f}% total | "
                            f"{strat['Annualized']*100:+.1f}% annualized over {strat['Days']} days."
                        )
                    else:
                        st.caption("BOB strategy: No completed trades.")

                    for warning_text in result.warnings:
                        st.warning(warning_text)

                    downloads_col1, downloads_col2, downloads_col3 = st.columns(3)
                    with downloads_col1:
                        st.download_button(
                            "PDF",
                            data=artifacts.pdf_bytes,
                            file_name=artifacts.pdf_name,
                            mime="application/pdf",
                            key=f"backtest_pdf_{safe_ticker}",
                        )
                        if artifacts.png_bytes:
                            st.download_button(
                                "Chart PNG",
                                data=artifacts.png_bytes,
                                file_name=f"{safe_ticker}_{rule_run}.png",
                                mime="image/png",
                                key=f"backtest_png_{safe_ticker}",
                            )
                    with downloads_col2:
                        st.download_button(
                            "Cycles CSV",
                            data=artifacts.cycles_csv,
                            file_name=f"{safe_ticker}_{rule_run}_cycles.csv",
                            mime="text/csv",
                            key=f"backtest_cycles_{safe_ticker}",
                        )
                        st.download_button(
                            "Benchmark CSV",
                            data=artifacts.benchmark_csv,
                            file_name=f"{safe_ticker}_{rule_run}_benchmark.csv",
                            mime="text/csv",
                            key=f"backtest_bench_{safe_ticker}",
                        )
                    with downloads_col3:
                        st.download_button(
                            "Strategy CSV",
                            data=artifacts.strategy_csv,
                            file_name=f"{safe_ticker}_{rule_run}_strategy.csv",
                            mime="text/csv",
                            key=f"backtest_strategy_{safe_ticker}",
                        )

                    st.divider()
