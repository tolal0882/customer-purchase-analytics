"""Shared, cached data access and app state for all pages (the only place pages touch the pipeline)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import settings
from src import data_loader, feature_engineering as fe, model_manager, pipeline

DEFAULT_CFG = {"timeout": settings.DEFAULT_SESSION_TIMEOUT_MIN, "dedup": settings.DEFAULT_REMOVE_DUPLICATES}


def cfg() -> dict:
    """Global processing configuration kept in st.session_state (changed on the Preprocessing page)."""
    if "cfg" not in st.session_state:
        st.session_state["cfg"] = dict(DEFAULT_CFG)
    return st.session_state["cfg"]


@st.cache_resource(show_spinner=False, max_entries=3)
def _core(timeout: float, dedup: bool):
    # cache_resource (not cache_data): the 2.7M-row tables are shared by reference, never copied per rerun.
    # Pages must treat them as read-only.
    return pipeline.build_core_tables(timeout, dedup)


def get_core():
    """Return CoreTables for the current config, or None if events.csv is missing (caller shows empty state)."""
    c = cfg()
    try:
        data_loader.events_path()
    except data_loader.DatasetNotFoundError:
        return None
    return _core(float(c["timeout"]), bool(c["dedup"]))


@st.cache_resource(show_spinner=False, max_entries=4)
def _snapshots(timeout, dedup, policy, n_events, cutoff, history):
    core = _core(timeout, dedup)
    return fe.build_prediction_snapshots(core.events, core.sessions, policy, n_events, cutoff, include_history=history)


def get_snapshots(policy: str, n_events: int, cutoff: float):
    c = cfg()
    return _snapshots(float(c["timeout"]), bool(c["dedup"]), policy, int(n_events), float(cutoff), True)


@st.cache_resource(show_spinner=False)
def get_raw_events():
    """Original events.csv exactly as stored (read-only, shared by reference)."""
    return data_loader.read_events_raw()


@st.cache_resource(show_spinner=False, max_entries=8)
def get_pipeline(exp_id: str):
    return model_manager.load_pipeline(exp_id)


def require_core():
    """Page guard: returns CoreTables or renders a setup empty-state and stops the page."""
    from components.streamlit_components import ui
    core = get_core()
    if core is None:
        ui.empty_state("Dataset not found",
                       f"{data_loader.SETUP_INSTRUCTION}<br><br>Looked in: <code>{settings.resolve_raw_dir()}</code>", "📂")
        st.stop()
    return core


def sidebar_status() -> dict:
    try:
        info = data_loader.discover_files()["events"]
    except Exception:
        info = None
    if info is None or not info.exists:
        ds = ("events.csv missing", "err", "See Data Preprocessing for setup")
    else:
        ds = ("Dataset ready", "ok", f"events.csv · session timeout {cfg()['timeout']:g} min")
    aid = model_manager.get_active_experiment_id()
    if aid:
        m = model_manager.load_experiment(aid)
        md = ("Model active", "ok", f"{m['model_name']} · PR-AUC {m['metrics']['pr_auc']:.3f}")
    else:
        md = ("No trained model", "warn", "Train one on Model Training")
    return {"dataset": ds, "model": md}


def fmt_int(n) -> str:
    return f"{int(n):,}"
