"""Reusable Streamlit UI components + design system (CSS)."""
from __future__ import annotations

import html
from contextlib import contextmanager

import streamlit as st

from config import settings

C = settings.COLORS

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Inter', system-ui, sans-serif; color: {C['text']}; }}
.stApp {{ background: {C['background']}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }}

/* Sidebar */
section[data-testid="stSidebar"] {{ background: {C['forest']}; }}
section[data-testid="stSidebar"] * {{ color: #E8F1EA; }}
section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,.15); }}
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] {{
    border-radius: 10px; padding: .45rem .7rem; margin: 1px 0; text-decoration: none; }}
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:hover {{ background: rgba(255,255,255,.08); }}
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][aria-current="page"] {{
    background: {C['primary']}; font-weight: 600; box-shadow: inset 3px 0 0 {C['accent']}; }}
.brand {{ display:flex; gap:.7rem; align-items:center; padding:.2rem .2rem 1rem; }}
.brand .logo {{ background:{C['accent']}; color:{C['forest']}; width:38px; height:38px; border-radius:10px;
    display:flex; align-items:center; justify-content:center; font-weight:700; font-size:1.2rem; }}
.brand .t1 {{ font-weight:700; font-size:1rem; line-height:1.1; color:#fff; }}
.brand .t2 {{ font-size:.72rem; opacity:.75; }}
.side-status {{ font-size:.78rem; line-height:1.5; background:rgba(255,255,255,.06); border-radius:10px; padding:.6rem .7rem; }}

/* Page header */
.page-header {{ margin-bottom: 1.1rem; }}
.page-header h1 {{ font-size: 1.7rem; font-weight: 700; color: {C['forest']}; margin: 0; padding: 0; }}
.page-header p {{ color: {C['muted']}; margin: .25rem 0 0; font-size: .95rem; }}

/* Cards */
.kpi {{ background:#fff; border:1px solid {C['border']}; border-radius:14px; padding:.95rem 1.05rem;
    box-shadow:0 1px 2px rgba(21,62,45,.05); height:100%; }}
.kpi .l {{ font-size:.72rem; font-weight:600; text-transform:uppercase; letter-spacing:.04em; color:{C['muted']}; }}
.kpi .v {{ font-size:1.55rem; font-weight:700; color:{C['forest']}; margin-top:.15rem; }}
.kpi .h {{ font-size:.72rem; color:{C['muted']}; margin-top:.1rem; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background:#fff; border-radius:14px; }}
.card-title {{ font-weight:600; color:{C['forest']}; font-size:1rem; margin-bottom:.1rem; }}
.card-sub {{ color:{C['muted']}; font-size:.8rem; margin-bottom:.4rem; }}

.badge {{ display:inline-block; padding:.12rem .6rem; border-radius:999px; font-size:.72rem; font-weight:600; }}
.badge.ok {{ background:#DDF3E4; color:#1B5E3B; }} .badge.warn {{ background:#FBF0D0; color:#8A5B00; }}
.badge.err {{ background:#FADBD8; color:#922B21; }} .badge.info {{ background:#E3EEF4; color:#27566B; }}
.badge.neutral {{ background:#E8EEE9; color:#3E5247; }}

.empty {{ text-align:center; padding:2.4rem 1rem; border:1.5px dashed {C['border']}; border-radius:16px; background:#fff; }}
.empty .ic {{ font-size:2.2rem; }} .empty .tt {{ font-weight:700; color:{C['forest']}; margin-top:.3rem; }}
.empty .tx {{ color:{C['muted']}; font-size:.9rem; max-width:620px; margin:.35rem auto 0; }}

.pred {{ border-radius:16px; padding:1.2rem 1.3rem; color:#fff; }}
.pred.pos {{ background:linear-gradient(135deg,{C['forest']},{C['primary']}); }}
.pred.neg {{ background:linear-gradient(135deg,#37474F,#607D8B); }}
.pred .lab {{ font-size:.75rem; opacity:.85; text-transform:uppercase; letter-spacing:.05em; }}
.pred .big {{ font-size:2rem; font-weight:700; }} .pred .prob {{ font-size:1.1rem; }}
.note {{ background:#EEF5EF; border-left:4px solid {C['primary']}; padding:.7rem .9rem; border-radius:8px; font-size:.85rem; }}
.warnbox {{ background:#FFF7E0; border-left:4px solid {C['warning']}; padding:.7rem .9rem; border-radius:8px; font-size:.85rem; }}
button[kind="primary"] {{ background:{C['primary']}; border-color:{C['primary']}; }}
button[kind="primary"]:hover {{ background:{C['forest']}; border-color:{C['forest']}; }}
@media (max-width: 768px) {{ .block-container {{ padding: 1rem .8rem 2rem; }} .page-header h1 {{ font-size:1.35rem; }}
    .kpi .v {{ font-size:1.25rem; }} }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape(str(x))


def badge(text: str, kind: str = "neutral") -> str:
    return f'<span class="badge {kind}">{esc(text)}</span>'


def page_header(title: str, subtitle: str = "", badges: list[tuple[str, str]] | None = None) -> None:
    b = " ".join(badge(t, k) for t, k in (badges or []))
    st.markdown(f'<div class="page-header"><h1>{esc(title)}</h1><p>{esc(subtitle)}</p>{b}</div>', unsafe_allow_html=True)


def kpi_row(items: list[tuple[str, str, str]], per_row: int = 4) -> None:
    """items: (label, value, help). Wraps to several rows on narrow layouts."""
    for i in range(0, len(items), per_row):
        cols = st.columns(per_row)
        for col, (label, value, hint) in zip(cols, items[i:i + per_row]):
            col.markdown(f'<div class="kpi"><div class="l">{esc(label)}</div><div class="v">{esc(value)}</div>'
                         f'<div class="h">{esc(hint)}</div></div>', unsafe_allow_html=True)


@contextmanager
def chart_card(title: str, subtitle: str = ""):
    with st.container(border=True):
        st.markdown(f'<div class="card-title">{esc(title)}</div>' + (f'<div class="card-sub">{esc(subtitle)}</div>' if subtitle else ""),
                    unsafe_allow_html=True)
        yield


def empty_state(title: str, message: str, icon: str = "📭") -> None:
    st.markdown(f'<div class="empty"><div class="ic">{icon}</div><div class="tt">{esc(title)}</div>'
                f'<div class="tx">{message}</div></div>', unsafe_allow_html=True)


def loading_state(message: str):
    """Context manager showing a spinner while an operation runs."""
    return st.spinner(message)


def note(text: str, warn: bool = False) -> None:
    st.markdown(f'<div class="{"warnbox" if warn else "note"}">{text}</div>', unsafe_allow_html=True)


def confirm_dialog(key: str, title: str, message: str, confirm_label: str = "Confirm") -> bool:
    """Opens a modal confirmation; returns True exactly once, on the rerun after the user confirms."""
    flag = f"_confirmed_{key}"
    if st.session_state.pop(flag, False):
        return True

    @st.dialog(title)
    def _dlg():
        st.write(message)
        c1, c2 = st.columns(2)
        if c1.button(confirm_label, type="primary", key=f"{key}_yes", use_container_width=True):
            st.session_state[flag] = True
            st.rerun()
        if c2.button("Cancel", key=f"{key}_no", use_container_width=True):
            st.rerun()

    _dlg()
    return False


def prediction_result_card(result: dict) -> None:
    pos = result["predicted_class"] == 1
    st.markdown(
        f'<div class="pred {"pos" if pos else "neg"}"><div class="lab">Model-based estimate</div>'
        f'<div class="big">{"Purchase likely" if pos else "No purchase expected"}</div>'
        f'<div class="prob">Purchase probability: <b>{result["probability"] * 100:.1f}%</b> '
        f'(decision threshold {result["threshold"] * 100:.0f}%)</div>'
        f'<div style="font-size:.78rem;opacity:.85;margin-top:.5rem">{esc(result["model"])} · {esc(result["timestamp"])} · '
        f'{esc(result["experiment_id"])}</div></div>', unsafe_allow_html=True)


def model_comparison_card(meta: dict) -> None:
    m = meta["metrics"]
    st.markdown(
        f'<div class="kpi"><div class="l">{esc(meta["model_name"])} · {esc(meta["created_at"])}</div>'
        f'<div class="h" style="font-size:.8rem;color:{C["text"]}">PR-AUC <b>{m["pr_auc"]:.3f}</b> · ROC-AUC <b>{m["roc_auc"]:.3f}</b> · '
        f'F1 <b>{m["f1"]:.3f}</b> · Recall <b>{m["recall"]:.3f}</b> · Precision <b>{m["precision"]:.3f}</b></div></div>',
        unsafe_allow_html=True)


def cluster_profile_card(cid, name: str, description: str, size: int, share: float, purchase_rate: float | None) -> None:
    pr = f" · purchase rate {purchase_rate:.2f}%" if purchase_rate is not None else ""
    st.markdown(f'<div class="kpi"><div class="l">Cluster {esc(cid)}</div><div class="v" style="font-size:1.05rem">{esc(name)}</div>'
                f'<div class="h">{size:,} members ({share:.1f}%){pr}</div><div class="h">{esc(description)}</div></div>',
                unsafe_allow_html=True)


def sidebar(current_status: dict) -> None:
    """Brand, navigation links and live dataset/model status indicators."""
    with st.sidebar:
        st.markdown('<div class="brand"><div class="logo">R</div><div><div class="t1">Retail Analytics</div>'
                    '<div class="t2">Purchase prediction & segmentation</div></div></div>', unsafe_allow_html=True)
        for key, title, icon in settings.PAGES:
            st.page_link(st.session_state["_pages"][key], label=title, icon=icon)
        st.markdown("---")
        ds, md = current_status["dataset"], current_status["model"]
        st.markdown(
            f'<div class="side-status"><b>Dataset</b><br>{badge(ds[0], ds[1])}<br><span style="opacity:.8">{esc(ds[2])}</span>'
            f'<div style="height:.5rem"></div><b>Model</b><br>{badge(md[0], md[1])}<br><span style="opacity:.8">{esc(md[2])}</span></div>',
            unsafe_allow_html=True)
