"""
Reusable SaaS UI Design System for the Streamlit + React Frontend.

Provides the Forest-Green & Sage SaaS aesthetic:
  - Custom branded sidebar (`ui.sidebar()`) with live Dataset, API & Model status badges
  - `.page-header` with status pills
  - `.kpi` metric grids (`ui.kpi_row()`)
  - `.card-title` / `.card-sub` container cards (`ui.chart_card()`)
  - `.pred.pos` / `.pred.neg` gradient prediction result banner (`ui.prediction_result_card()`)
  - `.note`, `.warnbox`, `.empty` states, and modal confirmation dialogs
"""
from __future__ import annotations

import html
from contextlib import contextmanager

import streamlit as st

COLORS = {
    "primary": "#2F855A",
    "forest": "#153E2D",
    "accent": "#A7C957",
    "background": "#F4F7F4",
    "card": "#FFFFFF",
    "text": "#263238",
    "muted": "#607D8B",
    "border": "#DCE6DF",
    "warning": "#D69E2E",
    "danger": "#C0392B",
}
C = COLORS

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Inter', system-ui, sans-serif; color: {C['text']}; }}
.stApp {{ background: {C['background']}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1400px; }}

/* Sidebar */
section[data-testid="stSidebar"] {{ background: {C['forest']}; }}
section[data-testid="stSidebar"] * {{ color: #E8F1EA; }}
section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,.15); }}
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] {{
    border-radius: 10px; padding: .48rem .75rem; margin: 2px 0; text-decoration: none;
    transition: background 0.15s ease;
}}
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:hover {{ background: rgba(255,255,255,.08); }}
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][aria-current="page"] {{
    background: {C['primary']} !important; font-weight: 600; box-shadow: inset 3px 0 0 {C['accent']};
}}
.brand {{ display:flex; gap:.75rem; align-items:center; padding:.25rem .2rem 1rem; }}
.brand .logo {{ background:{C['accent']}; color:{C['forest']} !important; width:40px; height:40px; border-radius:11px;
    display:flex; align-items:center; justify-content:center; font-weight:800; font-size:1.2rem; flex-shrink:0; }}
.brand .t1 {{ font-weight:700; font-size:1.02rem; line-height:1.15; color:#fff !important; }}
.brand .t2 {{ font-size:.73rem; opacity:.8; color:#DDF3E4 !important; }}
.side-status {{ font-size:.78rem; line-height:1.55; background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.1);
    border-radius:12px; padding:.7rem .8rem; }}

/* Page header */
.page-header {{
    background: #FFFFFF;
    border: 1px solid {C['border']};
    border-radius: 16px;
    padding: 1.15rem 1.35rem;
    margin-bottom: 1.15rem;
    box-shadow: 0 1px 3px rgba(21,62,45,.04);
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 1rem;
    flex-wrap: wrap;
}}
.page-header h1 {{ font-size: 1.55rem; font-weight: 700; color: {C['forest']}; margin: 0; padding: 0; line-height: 1.2; }}
.page-header p {{ color: {C['muted']}; margin: .3rem 0 0; font-size: .9rem; max-width: 880px; }}
.page-header .badges {{ display: flex; gap: .45rem; flex-wrap: wrap; align-items: center; }}

/* KPI Cards */
.kpi {{ background:#fff; border:1px solid {C['border']}; border-radius:14px; padding:.95rem 1.1rem;
    box-shadow:0 1px 3px rgba(21,62,45,.05); height:100%; margin-bottom: .65rem; }}
.kpi .l {{ font-size:.72rem; font-weight:600; text-transform:uppercase; letter-spacing:.045em; color:{C['muted']}; }}
.kpi .v {{ font-size:1.5rem; font-weight:700; color:{C['forest']}; margin-top:.18rem; line-height: 1.2; }}
.kpi .h {{ font-size:.74rem; color:{C['muted']}; margin-top:.2rem; }}

div[data-testid="stVerticalBlockBorderWrapper"] {{
    background:#fff; border-radius:14px; border-color:{C['border']};
    box-shadow: 0 1px 3px rgba(21,62,45,.04);
}}
div[data-testid="stMetric"] {{
    background: #FFFFFF; border: 1px solid {C['border']}; border-radius: 14px;
    padding: 0.85rem 1rem; box-shadow: 0 1px 3px rgba(21,62,45,0.05);
}}
.card-title {{ font-weight:600; color:{C['forest']}; font-size:1.02rem; margin-bottom:.1rem; }}
.card-sub {{ color:{C['muted']}; font-size:.8rem; margin-bottom:.5rem; }}

/* Badges */
.badge {{ display:inline-block; padding:.16rem .65rem; border-radius:999px; font-size:.73rem; font-weight:600; }}
.badge.ok {{ background:#DDF3E4; color:#1B5E3B !important; }}
.badge.warn {{ background:#FBF0D0; color:#8A5B00 !important; }}
.badge.err {{ background:#FADBD8; color:#922B21 !important; }}
.badge.info {{ background:#E3EEF4; color:#27566B !important; }}
.badge.neutral {{ background:#E8EEE9; color:#3E5247 !important; }}

/* Empty state */
.empty {{ text-align:center; padding:2.2rem 1rem; border:1.5px dashed {C['border']}; border-radius:16px; background:#fff; }}
.empty .ic {{ font-size:2.1rem; }}
.empty .tt {{ font-weight:700; color:{C['forest']}; margin-top:.3rem; font-size:1.05rem; }}
.empty .tx {{ color:{C['muted']}; font-size:.88rem; max-width:600px; margin:.3rem auto 0; }}

/* Prediction Result Hero Banner */
.pred {{ border-radius:16px; padding:1.25rem 1.35rem; color:#fff !important; margin-bottom:.85rem;
    box-shadow: 0 6px 18px rgba(21,62,45,.15); }}
.pred * {{ color:#fff !important; }}
.pred.pos {{ background:linear-gradient(135deg, {C['forest']}, {C['primary']}); }}
.pred.neg {{ background:linear-gradient(135deg, #37474F, #607D8B); }}
.pred .lab {{ font-size:.73rem; opacity:.88; text-transform:uppercase; letter-spacing:.06em; font-weight:600; }}
.pred .big {{ font-size:1.85rem; font-weight:700; margin:.2rem 0; }}
.pred .prob {{ font-size:1.05rem; opacity:.95; }}

/* Callout Notes */
.note {{ background:#EEF5EF; border-left:4px solid {C['primary']}; padding:.75rem 1rem; border-radius:10px;
    font-size:.85rem; color:{C['text']}; margin:.5rem 0 .85rem; }}
.warnbox {{ background:#FFF7E0; border-left:4px solid {C['warning']}; padding:.75rem 1rem; border-radius:10px;
    font-size:.85rem; color:{C['text']}; margin:.5rem 0 .85rem; }}

button[kind="primary"] {{ background:{C['primary']} !important; border-color:{C['primary']} !important; border-radius:10px !important; font-weight:600 !important; }}
button[kind="primary"]:hover {{ background:{C['forest']} !important; border-color:{C['forest']} !important; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape(str(x))


def badge(text: str, kind: str = "neutral") -> str:
    return f'<span class="badge {kind}">{esc(text)}</span>'


def page_header(title: str, subtitle: str = "", badges: list[tuple[str, str]] | None = None) -> None:
    b = "".join(badge(t, k) for t, k in (badges or []))
    b_div = f'<div class="badges">{b}</div>' if b else ""
    st.markdown(
        f'<div class="page-header"><div><h1>{esc(title)}</h1>'
        + (f"<p>{esc(subtitle)}</p>" if subtitle else "")
        + f"</div>{b_div}</div>",
        unsafe_allow_html=True,
    )


def kpi_row(items: list[tuple[str, str, str]], per_row: int = 4) -> None:
    """Render styled SaaS KPI cards: items = [(label, value, hint), ...]."""
    for i in range(0, len(items), per_row):
        cols = st.columns(per_row)
        for col, (label, value, hint) in zip(cols, items[i : i + per_row]):
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


def empty_state(title: str, text: str, icon: str = "📭") -> None:
    st.markdown(
        f'<div class="empty"><div class="ic">{esc(icon)}</div>'
        f'<div class="tt">{esc(title)}</div><div class="tx">{text}</div></div>',
        unsafe_allow_html=True,
    )


def note(text: str, warn: bool = False) -> None:
    st.markdown(f'<div class="{"warnbox" if warn else "note"}">{text}</div>', unsafe_allow_html=True)


def prediction_result_card(res: dict) -> None:
    pos = res.get("prediction") == "Yes"
    p_yes = float(res.get("probability", res.get("probabilities", {}).get("Yes", 0.0)))
    thr = float(res.get("threshold", 0.5))
    model_name = res.get("model_name", "Classifier")
    tier = res.get("intent_tier", "Session Estimate")
    st.markdown(
        f'<div class="pred {"pos" if pos else "neg"}">'
        f'<div class="lab">Model-Based Early-Session Estimate · {esc(tier)}</div>'
        f'<div class="big">{"Purchase Likely (Class 1 — Yes)" if pos else "No Purchase Expected (Class 0 — No)"}</div>'
        f'<div class="prob">Purchase probability: <b>{p_yes * 100:.1f}%</b> '
        f'(decision threshold: <b>{thr * 100:.0f}%</b>)</div>'
        f'<div style="font-size:.78rem;opacity:.88;margin-top:.55rem">'
        f'Active Pipeline: <b>{esc(model_name)}</b></div></div>',
        unsafe_allow_html=True,
    )


def model_summary_card(report: dict) -> None:
    m = report.get("test_metrics", {})
    tcfg = report.get("training_config", {})
    st.markdown(
        f'<div class="kpi">'
        f'<div class="l">Active Classifier · {esc(report.get("best_model", "Trained"))} ({esc(report.get("experiment_id", "default"))})</div>'
        f'<div class="h" style="font-size:.82rem;color:{C["text"]};margin-top:.35rem">'
        f'ROC-AUC <b>{m.get("roc_auc", 0):.4f}</b> · PR-AUC <b>{m.get("pr_auc", 0):.4f}</b> · '
        f'F1 <b>{m.get("f1", 0):.4f}</b> · Recall <b>{m.get("recall", 0):.4f}</b> · '
        f'Precision <b>{m.get("precision", 0):.4f}</b> · Threshold <b>{tcfg.get("threshold", 0.5):.2f}</b>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def sidebar(pages_list: list[tuple[object, str, str]], status_info: dict) -> None:
    """Render the custom Forest-Green SaaS sidebar with navigation links and live system status."""
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="logo">R</div>'
            '<div><div class="t1">Retail Analytics</div>'
            '<div class="t2">Purchase prediction & segmentation</div></div></div>',
            unsafe_allow_html=True,
        )
        for page_obj, label, icon in pages_list:
            st.page_link(page_obj, label=label, icon=icon)
        st.markdown("---")
        ds_badge = badge(status_info.get("dataset_label", "Retailrocket Loaded"), "ok")
        api_badge = badge(status_info.get("api_label", "FastAPI Online"), status_info.get("api_kind", "ok"))
        md_badge = badge(status_info.get("model_label", "Model Ready"), "ok")
        st.markdown(
            f'<div class="side-status">'
            f'<b>Dataset</b><br>{ds_badge}<br><span style="opacity:.82">{esc(status_info.get("dataset_sub", "2,755,641 clean events"))}</span>'
            f'<div style="height:.45rem"></div>'
            f'<b>Active ML Pipeline</b><br>{md_badge}<br><span style="opacity:.82">{esc(status_info.get("model_sub", "K-Means K=4 + Classifier"))}</span>'
            f'<div style="height:.45rem"></div>'
            f'<b>Backend Service</b><br>{api_badge}'
            f'</div>',
            unsafe_allow_html=True,
        )
