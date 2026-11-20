"""Page 3: Data Preprocessing (schema validation, quality, cleaning log, session construction)."""
import numpy as np
import pandas as pd
import streamlit as st

from app_pages import common
from components.streamlit_components import ui
from config import settings
from src import data_loader, data_validation, session_engineering
from src.utilities import human_bytes


@st.cache_data(show_spinner=False)
def _timeout_sensitivity(version: str, timeouts: tuple):
    core = common.get_core()
    rows = []
    for m in timeouts:
        n = session_engineering.count_sessions(core.events, m)
        rows.append({"timeout_min": m, "sessions": n, "avg_events_per_session": len(core.events) / n})
    return pd.DataFrame(rows)


def _setup_section():
    files = data_loader.discover_files()
    rows = [{"file": f.filename, "found": "✅" if f.exists else "❌", "size": human_bytes(f.size_bytes) if f.exists else "—",
             "role": "required" if k in settings.REQUIRED_FILES else "optional", "path": str(f.path)} for k, f in files.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if not files["events"].exists:
        ui.empty_state("events.csv not found", data_loader.SETUP_INSTRUCTION, "📂")
        up = st.file_uploader("Upload events.csv", type="csv", key="upl_events")
        if up is not None and st.button("Save uploaded file to data/raw/events.csv", type="primary"):
            dest = settings.RAW_DIR_CANDIDATES[0]
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "events.csv").write_bytes(up.getbuffer())
            st.success(f"Saved to {dest / 'events.csv'}. Reloading…")
            st.cache_resource.clear()
            st.rerun()
        st.stop()


def render():
    ui.page_header("Data Preprocessing", "Validate the schema, review data quality, apply logged cleaning steps and construct sessions. "
                                         "Nothing is removed silently and the raw files are never modified.")
    with ui.chart_card("1 · Dataset files", f"Raw folder in use: {settings.resolve_raw_dir()}"):
        _setup_section()

    raw = common.get_raw_events()
    with ui.chart_card("2 · Schema validation", "Expected columns: " + ", ".join(settings.EVENTS_COLUMNS)):
        sr = data_validation.validate_events_schema(raw)
        st.markdown(ui.badge("Schema valid", "ok") if sr.ok else ui.badge("Schema invalid", "err"), unsafe_allow_html=True)
        for m in sr.messages:
            st.warning(m)
        st.dataframe(pd.DataFrame({"column": list(sr.dtypes), "dtype_read": list(sr.dtypes.values())}), hide_index=True, use_container_width=True)
        if not sr.ok:
            st.stop()

    c = common.cfg()
    with ui.chart_card("3 · Processing settings", "Changing these rebuilds sessions and all downstream tables (results are cached on disk)."):
        with st.form("proc_form"):
            f1, f2 = st.columns(2)
            timeout = f1.number_input("Session timeout (minutes)", 1.0, 1440.0, float(c["timeout"]), 1.0,
                                      help="A new session starts after this much inactivity. 30 min is a common analytical assumption, NOT a field in the dataset.")
            dedup = f2.checkbox("Remove exact duplicate rows", value=bool(c["dedup"]), help="Duplicates are always counted and shown; this only controls removal.")
            go = st.form_submit_button("Apply & rebuild", type="primary")
        if go:
            st.session_state["cfg"] = {"timeout": timeout, "dedup": dedup}
            st.rerun()
        ui.note("The 30-minute threshold is an analytical assumption. Different thresholds change session counts and therefore labels and features; "
                "see the sensitivity table below.")

    with ui.loading_state("Building processed tables (first run reads and caches to data/processed/)…"):
        core = common.require_core()
    q, meta = core.quality, core.meta

    with ui.chart_card("4 · Data quality (raw file)"):
        ui.kpi_row([
            ("Raw rows", common.fmt_int(q["rows"]), "events.csv"),
            ("Duplicate rows", common.fmt_int(q["duplicate_rows"]), f"{q['duplicate_rows'] / q['rows'] * 100:.3f}% of rows"),
            ("Invalid timestamps", common.fmt_int(q["invalid_timestamps"]), "non-numeric / out of range"),
            ("Invalid event names", common.fmt_int(q["invalid_events"]), "not view/addtocart/transaction"),
        ])
        a, b = st.columns(2)
        a.markdown("**Missing values**")
        a.dataframe(pd.Series(q["missing_by_column"], name="missing").to_frame(), use_container_width=True)
        a.caption("`transactionid` is empty for every non-transaction event by design.")
        b.markdown("**Event categories & coverage**")
        b.dataframe(pd.Series(q["event_counts"], name="rows").to_frame(), use_container_width=True)
        b.caption(f"Timestamps {q['timestamp_min']} → {q['timestamp_max']} ({q['span_days']:.0f} days) · "
                  f"sorted in file: {'yes' if q['sorted_by_timestamp'] else 'no (sorted during processing)'}")
        st.caption(f"Transactions without transactionid: {q['transactions_without_transactionid']} · "
                   f"non-transactions with transactionid: {q['non_transactions_with_transactionid']}")

    with ui.chart_card("5 · Cleaning log (effect of every step)"):
        st.dataframe(core.cleaning_steps, use_container_width=True, hide_index=True)

    with ui.chart_card("6 · Before / after comparison"):
        clean = core.events
        comp = pd.DataFrame([
            {"metric": "Rows", "before (raw)": f"{len(raw):,}", "after (processed)": f"{len(clean):,}"},
            {"metric": "Memory", "before (raw)": f"{raw.memory_usage(deep=True).sum() / 1e6:.0f} MB", "after (processed)": f"{clean.memory_usage(deep=True).sum() / 1e6:.0f} MB (incl. session columns)"},
            {"metric": "timestamp dtype", "before (raw)": str(raw['timestamp'].dtype), "after (processed)": str(clean['timestamp'].dtype)},
            {"metric": "visitorid dtype", "before (raw)": str(raw['visitorid'].dtype), "after (processed)": str(clean['visitorid'].dtype)},
            {"metric": "event dtype", "before (raw)": str(raw['event'].dtype), "after (processed)": str(clean['event'].dtype)},
            {"metric": "transactionid dtype", "before (raw)": str(raw['transactionid'].dtype), "after (processed)": str(clean['transactionid'].dtype)},
            {"metric": "Columns", "before (raw)": ", ".join(raw.columns), "after (processed)": ", ".join(clean.columns)},
        ])
        st.dataframe(comp, use_container_width=True, hide_index=True)

    with ui.chart_card("7 · Session construction", f"Timeout = {meta['session_timeout_minutes']:g} min (configurable above)"):
        s = core.sessions
        single = (s["total_events"] == 1).mean() * 100
        ui.kpi_row([
            ("Sessions", common.fmt_int(len(s)), "engineered session_id"),
            ("Visitors", common.fmt_int(meta["n_visitors"]), ""),
            ("Mean events / session", f"{s['total_events'].mean():.2f}", f"median {s['total_events'].median():.0f}"),
            ("Single-event sessions", f"{single:.1f}%", "uninformative for prediction"),
        ])
        sens = _timeout_sensitivity(meta["dataset_version"], (5, 10, 15, 30, 60, 120))
        st.markdown("**Sensitivity to the timeout** (computed on the processed events)")
        st.dataframe(sens, use_container_width=True, hide_index=True)

    with ui.chart_card("8 · Processed data export", "Written to data/processed/ or downloaded; originals untouched."):
        c1, c2, c3 = st.columns(3)
        c1.download_button("⬇️ Customers CSV", core.customers.reset_index().to_csv(index=False).encode(), "customers.csv", "text/csv")
        c2.download_button("⬇️ Sessions CSV", core.sessions.reset_index().to_csv(index=False).encode(), "sessions.csv", "text/csv")
        if c3.button("💾 Write processed events CSV to data/processed/"):
            with ui.loading_state("Writing CSV…"):
                p = settings.PROCESSED_DIR / "events_processed.csv"
                core.events.to_csv(p, index=False)
            st.success(f"Wrote {p} ({human_bytes(p.stat().st_size)}, {len(core.events):,} rows)")
        st.caption(f"Dataset version: `{meta['dataset_version']}` · built {meta['built_at'][:19]} in {meta['build_seconds']}s "
                   "(0 s when loaded from cache)")
