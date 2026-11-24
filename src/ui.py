"""
Design System & Reusable Native UI Helpers for the Streamlit Application.

Color Palette (Section 9):
  - Forest Green : #153E2D
  - Primary Green: #2F855A
  - Accent Green : #A7C957
  - Background   : #F3F7F3
  - White        : #FFFFFF
  - Text         : #263238
"""
from __future__ import annotations

import html
from contextlib import contextmanager

import streamlit as st

COLORS = {
    "forest": "#153E2D",
    "primary": "#2F855A",
    "accent": "#A7C957",
    "background": "#F3F7F3",
    "white": "#FFFFFF",
    "text": "#263238",
    "muted": "#5C6F68",
    "border": "#D8E4DC",
    "warning": "#D69E2E",
    "danger": "#C0392B",
}
C = COLORS

PLOTLY_COLORWAY = ["#2F855A", "#153E2D", "#A7C957", "#D69E2E", "#3B7A57", "#607D8B", "#8FBC8F"]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {{
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    color: {C['text']};
}}
.stApp {{ background-color: {C['background']}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{
    padding-top: 1.75rem !important;
    padding-left: clamp(1rem, 2.5vw, 2.5rem) !important;
    padding-right: clamp(1rem, 2.5vw, 2.5rem) !important;
    padding-bottom: 3rem !important;
    max-width: 100% !important;
}}

/* Sidebar Styling */
section[data-testid="stSidebar"] {{
    background-color: {C['forest']};
    border-right: 1px solid rgba(255,255,255,0.1);
}}
section[data-testid="stSidebar"] * {{ color: #ECF4EE; }}
section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.15); }}

/* Categorized Native Sidebar Navigation */
section[data-testid="stSidebar"] [data-testid="stSidebarNav"]::before {{
    content: "🛒 Online Shoppers ML";
    display: block;
    font-weight: 700;
    font-size: 1.12rem;
    color: #FFFFFF;
    padding: 1rem 1rem 0.75rem;
    margin-bottom: 0.4rem;
    border-bottom: 1px solid rgba(255,255,255,0.14);
}}
section[data-testid="stSidebar"] [data-testid="stSidebarNavSeparator"],
section[data-testid="stSidebar"] [data-testid="stSidebarNavItems"] > div > span {{
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-size: 0.70rem !important;
    font-weight: 700 !important;
    color: {C['accent']} !important;
    opacity: 1 !important;
}}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] {{
    border-radius: 10px;
    margin: 2px 8px;
    padding: 0.48rem 0.75rem;
    transition: background 0.15s ease;
}}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"]:hover {{
    background: rgba(255,255,255,0.09);
}}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][aria-current="page"] {{
    background: {C['primary']} !important;
    font-weight: 600;
    box-shadow: inset 3px 0 0 {C['accent']};
}}

.side-panel {{
    font-size: 0.80rem;
    line-height: 1.55;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 0.8rem 0.9rem;
    margin-top: 0.5rem;
}}

/* Consistent Page Header Card */
.page-header {{
    background: {C['white']};
    border: 1px solid {C['border']};
    border-radius: 16px;
    padding: 1.3rem 1.55rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 1px 4px rgba(21,62,45,0.05);
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    flex-wrap: wrap;
}}
.page-header .ph-cat {{
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {C['primary']};
    margin-bottom: 0.2rem;
}}
.page-header h1 {{
    font-size: clamp(1.35rem, 2vw, 1.8rem);
    font-weight: 700;
    color: {C['forest']};
    margin: 0;
    padding: 0;
    line-height: 1.2;
}}
.page-header p {{
    color: {C['muted']};
    margin: 0.35rem 0 0;
    font-size: 0.94rem;
    max-width: 960px;
}}
.page-header .badges {{
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
    align-items: center;
}}

/* KPI Cards */
.kpi {{
    background: {C['white']};
    border: 1px solid {C['border']};
    border-radius: 14px;
    padding: 1.05rem 1.2rem;
    box-shadow: 0 1px 3px rgba(21,62,45,0.05);
    height: 100%;
    margin-bottom: 0.7rem;
}}
.kpi .l {{
    font-size: 0.74rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.045em;
    color: {C['muted']};
}}
.kpi .v {{
    font-size: clamp(1.3rem, 1.8vw, 1.7rem);
    font-weight: 700;
    color: {C['forest']};
    margin-top: 0.2rem;
    line-height: 1.2;
}}
.kpi .h {{
    font-size: 0.78rem;
    color: {C['muted']};
    margin-top: 0.22rem;
}}

/* Containers & Chart Cards */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {C['white']};
    border-radius: 14px;
    border-color: {C['border']};
    box-shadow: 0 1px 3px rgba(21,62,45,0.04);
}}
div[data-testid="stMetric"] {{
    background: {C['white']};
    border: 1px solid {C['border']};
    border-radius: 14px;
    padding: 0.95rem 1.1rem;
    box-shadow: 0 1px 3px rgba(21,62,45,0.05);
}}
.card-title {{
    font-weight: 600;
    color: {C['forest']};
    font-size: 1.06rem;
    margin-bottom: 0.12rem;
}}
.card-sub {{
    color: {C['muted']};
    font-size: 0.84rem;
    margin-bottom: 0.55rem;
}}

/* Status Badges */
.badge {{
    display: inline-block;
    padding: 0.2rem 0.72rem;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 600;
}}
.badge.ok {{ background: #DDF3E4; color: #153E2D !important; }}
.badge.warn {{ background: #FBF0D0; color: #8A5B00 !important; }}
.badge.info {{ background: #E3EEF4; color: #1E4D63 !important; }}
.badge.neutral {{ background: #E8EEE9; color: #354A3F !important; }}

/* Prediction Banner */
.pred-banner {{
    border-radius: 16px;
    padding: 1.25rem 1.4rem;
    color: #FFFFFF !important;
    margin-bottom: 0.85rem;
    box-shadow: 0 6px 16px rgba(21,62,45,0.14);
}}
.pred-banner * {{ color: #FFFFFF !important; }}
.pred-banner.pos {{ background: linear-gradient(135deg, {C['forest']}, {C['primary']}); }}
.pred-banner.neg {{ background: linear-gradient(135deg, #37474F, #607D8B); }}
.pred-banner .lab {{ font-size: 0.74rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }}
.pred-banner .big {{ font-size: 1.75rem; font-weight: 700; margin: 0.2rem 0; }}
.pred-banner .prob {{ font-size: 1.02rem; opacity: 0.95; }}

/* Info / Warning Callouts & Empty States */
.note {{
    background: #EBF4ED;
    border-left: 4px solid {C['primary']};
    padding: 0.75rem 1rem;
    border-radius: 10px;
    font-size: 0.87rem;
    color: {C['text']};
    margin: 0.45rem 0 0.85rem;
}}
.warnbox {{
    background: #FFF7E0;
    border-left: 4px solid {C['warning']};
    padding: 0.75rem 1rem;
    border-radius: 10px;
    font-size: 0.87rem;
    color: {C['text']};
    margin: 0.45rem 0 0.85rem;
}}
.empty {{
    text-align: center;
    padding: 2.2rem 1rem;
    border: 1.5px dashed {C['border']};
    border-radius: 16px;
    background: {C['white']};
}}
.empty .ic {{ font-size: 2.1rem; }}
.empty .tt {{ font-weight: 700; color: {C['forest']}; margin-top: 0.3rem; font-size: 1.05rem; }}
.empty .tx {{ color: {C['muted']}; font-size: 0.88rem; max-width: 600px; margin: 0.3rem auto 0; }}

button[kind="primary"] {{
    background: {C['primary']} !important;
    border-color: {C['primary']} !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}}
button[kind="primary"]:hover {{
    background: {C['forest']} !important;
    border-color: {C['forest']} !important;
}}

/* Responsive Tablet & Mobile Breakpoints */
@media (max-width: 1100px) {{
    div[data-testid="stHorizontalBlock"] {{
        flex-wrap: wrap !important;
        gap: 0.75rem !important;
    }}
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {{
        flex: 1 1 46% !important;
        min-width: 240px !important;
    }}
}}
@media (max-width: 768px) {{
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {{
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }}
}}
</style>
"""


# =============================================================================
# 2. REUSABLE UI HELPER FUNCTIONS (PAGE HEADERS, KPI CARDS, CHART CARDS, NOTES)
# =============================================================================
def inject_css() -> None:
    """Inject the Forest Green design system CSS into the current Streamlit page."""
    st.markdown(CSS, unsafe_allow_html=True)


def esc(val) -> str:
    """HTML-escape any dynamic string before rendering inside custom HTML cards."""
    return html.escape(str(val))


def badge(text: str, kind: str = "neutral") -> str:
    """Render a pill-shaped status badge ('ok', 'warn', 'info', or 'neutral')."""
    return f'<span class="badge {kind}">{esc(text)}</span>'


def page_header(
    title: str,
    subtitle: str = "",
    category: str = "UCI Online Shoppers Analytics",
    badges: list[tuple[str, str]] | None = None,
) -> None:
    b_html = "".join(badge(t, k) for t, k in (badges or []))
    b_wrap = f'<div class="badges">{b_html}</div>' if b_html else ""
    st.markdown(
        f'<div class="page-header">'
        f'<div><div class="ph-cat">{esc(category)}</div>'
        f'<h1>{esc(title)}</h1>'
        + (f"<p>{esc(subtitle)}</p>" if subtitle else "")
        + f"</div>{b_wrap}</div>",
        unsafe_allow_html=True,
    )


def kpi_row(items: list[tuple[str, str, str]], per_row: int = 3) -> None:
    """Render styled KPI cards: items = [(label, value, hint), ...]."""
    for i in range(0, len(items), per_row):
        chunk = items[i : i + per_row]
        cols = st.columns(len(chunk))
        for col, (label, value, hint) in zip(cols, chunk):
            col.markdown(
                f'<div class="kpi"><div class="l">{esc(label)}</div>'
                f'<div class="v">{esc(value)}</div>'
                f'<div class="h">{esc(hint)}</div></div>',
                unsafe_allow_html=True,
            )


@contextmanager
def chart_card(title: str, subtitle: str = ""):
    with st.container(border=True):
        st.markdown(
            f'<div class="card-title">{esc(title)}</div>'
            + (f'<div class="card-sub">{esc(subtitle)}</div>' if subtitle else ""),
            unsafe_allow_html=True,
        )
        yield


def note(text: str, warn: bool = False) -> None:
    st.markdown(f'<div class="{"warnbox" if warn else "note"}">{text}</div>', unsafe_allow_html=True)


def empty_state(title: str, text: str, icon: str = "📭") -> None:
    st.markdown(
        f'<div class="empty"><div class="ic">{esc(icon)}</div>'
        f'<div class="tt">{esc(title)}</div><div class="tx">{text}</div></div>',
        unsafe_allow_html=True,
    )


def apply_plotly_theme(fig, height: int = 320):
    """Apply consistent Forest Green & clean white styling to any Plotly figure."""
    fig.update_layout(
        height=height,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Inter, system-ui, sans-serif", color=C["text"], size=12),
        margin=dict(l=12, r=12, t=28, b=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False, linecolor=C["border"])
    fig.update_yaxes(gridcolor="#EEF3EF", linecolor=C["border"])
    return fig
