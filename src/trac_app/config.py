from __future__ import annotations

import pandas as pd

# ===== TRAC defaults =====
DEF_BAND_BASE = 1.2844025
DEF_CORRIDOR = 0.064
DEF_NUM_BANDS = 30

TIME_RANGES = {
    "5 days": pd.DateOffset(days=5),
    "1 mo": pd.DateOffset(months=1),
    "3 mo": pd.DateOffset(months=3),
    "6 mo": pd.DateOffset(months=6),
    "1 yr": pd.DateOffset(years=1),
    "3 yr": pd.DateOffset(years=3),
    "5 yr": pd.DateOffset(years=5),
    "all time": None,
}

TIME_RANGE_DESCRIPTIONS = {
    "5 days": "five days",
    "1 mo": "one month",
    "3 mo": "three months",
    "6 mo": "six months",
    "1 yr": "one year",
    "3 yr": "three years",
    "5 yr": "five years",
    "all time": "the full available history",
}

ETF_ADJUSTMENT_FACTORS = {
    "SPY-US": 0.02,
    "QQQ-US": 0.005,
    "XLF-US": 0.04,
}

THEME_CSS = """
<style>
:root {
    --trac-bg: rgba(24, 61, 80, 0.8);
    --trac-text: #FFFFFF;
    --trac-gold: #C7AB66;
    --trac-white: #FFFFFF;
    --trac-accent: #038CAD;
    --trac-filter-bg: #D5D8DB;
}
.stApp {
    background: transparent;
    color: var(--trac-text);
}
.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: var(--trac-bg);
    z-index: -1;
}
.stMarkdown, .stText, .stCaption, p, label, span, h1, h2, h3, h4, h5, h6 {
    color: var(--trac-text) !important;
}
.stMarkdown a, .stText a, a {
    color: var(--trac-accent) !important;
}
.stButton > button, .stDownloadButton > button {
    background-color: var(--trac-gold);
    color: var(--trac-white) !important;
    border: 1px solid var(--trac-gold);
}
.stButton > button:hover, .stButton > button:focus,
.stDownloadButton > button:hover, .stDownloadButton > button:focus {
    background-color: var(--trac-gold);
    color: var(--trac-accent) !important;
    opacity: 0.88;
}
section[data-testid="stSidebar"] {
    background: rgba(24, 61, 80, 0.85);
}
section[data-testid="stSidebar"] * {
    color: var(--trac-text) !important;
}
.stSelectbox label, .stSlider label, .stRadio label, .stCheckbox label {
    color: var(--trac-text) !important;
}
.stMultiSelect div[data-baseweb="select"] > div,
.stSelectbox div[data-baseweb="select"] > div,
.stSelectbox div[data-baseweb="select"] > div:focus {
    background-color: var(--trac-filter-bg) !important;
    color: var(--trac-accent) !important;
    border: none !important;
}
.stMultiSelect div[data-baseweb="select"] span,
.stSelectbox div[data-baseweb="select"] span,
.stSelectbox div[data-baseweb="select"] div,
.stSelectbox div[data-baseweb="select"] p {
    color: var(--trac-accent) !important;
}
.stTextInput input {
    background-color: var(--trac-filter-bg) !important;
    color: var(--trac-accent) !important;
    border: none !important;
}
.stTextInput input::placeholder {
    color: rgba(3, 140, 173, 0.7) !important;
}
.stSelectbox div[data-baseweb="select"] input,
.stMultiSelect div[data-baseweb="select"] input {
    color: var(--trac-accent) !important;
}
.stMultiSelect div[data-baseweb="tag"] {
    background-color: var(--trac-filter-bg) !important;
    color: var(--trac-accent) !important;
    border: none !important;
}
.stMultiSelect div[data-baseweb="select"] svg {
    fill: var(--trac-accent) !important;
}
.stMultiSelect div[data-baseweb="tag"] svg {
    fill: var(--trac-accent) !important;
}
button[title="Expand sidebar"],
button[aria-label="Expand sidebar"],
button[title="Collapse sidebar"],
button[aria-label="Collapse sidebar"] {
    background-color: rgba(199, 171, 102, 0.18) !important;
    border: 1px solid var(--trac-accent) !important;
    color: var(--trac-accent) !important;
    box-shadow: none !important;
}
button[title="Expand sidebar"] svg,
button[aria-label="Expand sidebar"] svg,
button[title="Collapse sidebar"] svg,
button[aria-label="Collapse sidebar"] svg {
    fill: var(--trac-accent) !important;
}
.stRadio div[role="radio"] label span {
    color: var(--trac-text) !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--trac-text) !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--trac-accent) !important;
}
div[data-testid="stDialog"] h1,
div[data-testid="stDialog"] h2,
div[data-testid="stDialog"] h3,
div[data-testid="stDialog"] label {
    color: var(--trac-text) !important;
}
div[data-testid="stTooltip"],
div[data-baseweb="tooltip"] {
    color: #000000 !important;
    background-color: #FFFFFF !important;
    border: 1px solid rgba(0, 0, 0, 0.25) !important;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.18) !important;
}
div[data-testid="stTooltip"] *,
div[data-baseweb="tooltip"] * {
    color: #000000 !important;
}

</style>
"""

OAI_PROMPT = [
    f"You're a reporter at the Wall Street Journal that covers portfolio moves by notable investors. Give me a synopsis for ",  
    f"If you don't see anything, leave this section blank.",
    f"I also want to know what leading sell side analysts think about the stock. Give me a synopsis. Return all results in bullet form.",
    f"If you don't see anything, leave this section blank.",
    f"I also want to know if there is any relevant news that could move the price, for instance, a semiconductor company upgrading a factory,",
    f"or Apple releasing a new iPhone. If you don't see anything, leave this section blank.",
    f"Report your answer in three sections: one called Overview, one called Analyst Sentiment, and one called Relevant News.",
    f"Don't include any extra text, like offering to follow up. Just present the facts, and nothing else.",
    f"Bold any specific important words by using **s (Markdown style), so that the report can be read at a glance.",
    f"Don't reference any specific price or market cap data, as this is already included in the reports.",
    f"Don't include random facts about the company like its name or its sector."
]
