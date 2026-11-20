"""Page 2: Dataset Explorer (raw + processed tables)."""
import numpy as np
import pandas as pd
import streamlit as st

from app_pages import common
from components.streamlit_components import ui
from config import settings
from src import data_loader, data_validation

DICTIONARY = {
    "timestamp": ("Epoch milliseconds of the event (UTC).", "raw"),
    "visitorid": ("Anonymous visitor identifier. Used for grouping only, never as a numeric model feature.", "raw"),
    "event": ("view, addtocart or transaction.", "raw"),
    "itemid": ("Product identifier. Not a model feature.", "raw"),
    "transactionid": ("Order id; only present for transaction events. Not a model feature.", "raw"),
    "session_id": ("ENGINEERED: gap-based session id (new session after the configured inactivity timeout).", "engineered"),
    "event_pos": ("ENGINEERED: 0-based position of the event inside its session.", "engineered"),
    "start_ts": ("Session start (epoch ms).", "engineered"), "end_ts": ("Session end (epoch ms).", "engineered"),
    "total_events": ("Events in the session / observed so far.", "engineered"),
    "total_views": ("Product views.", "engineered"), "add_to_cart_count": ("Add-to-cart events.", "engineered"),
    "transaction_count": ("Transaction rows in the session. RETROSPECTIVE; never a prediction input.", "engineered"),
    "unique_items_viewed": ("Distinct items viewed.", "engineered"), "unique_items_total": ("Distinct items in any event.", "engineered"),
    "session_duration_sec": ("Last minus first event, seconds.", "engineered"),
    "avg_time_between_events_sec": ("Duration/(events-1); 0 for single-event sessions.", "engineered"),
    "cart_interaction_ratio": ("add_to_cart_count / total_events.", "engineered"),
    "item_revisit_ratio": ("Share of views that revisit an item.", "engineered"),
    "first_cart_position": ("0-based position of first add-to-cart, -1 if none.", "engineered"),
    "ends_with_cart": ("Last event is add-to-cart.", "engineered"), "has_purchase": ("Session/visitor has ≥1 transaction. Used for LABELLING/comparison only.", "engineered"),
    "snapshot_ts": ("Timestamp of the prediction snapshot.", "engineered"), "target": ("1 = transaction after the snapshot.", "engineered"),
    "total_sessions": ("Sessions per visitor.", "engineered"), "total_add_to_cart": ("Add-to-cart events per visitor.", "engineered"),
    "total_transactions": ("Transaction rows per visitor.", "engineered"), "purchase_frequency": ("Purchase sessions / sessions.", "engineered"),
    "average_session_duration": ("Mean session duration (s).", "engineered"),
    "days_since_last_activity": ("Days between visitor's last event and dataset end.", "engineered"),
    "historical_cart_abandonment_rate": ("Cart sessions without purchase / cart sessions (proxy; no remove-from-cart event exists).", "engineered"),
}


def _dataset(name: str):
    core = common.get_core()
    if name == "Raw events.csv (as stored)":
        return common.get_raw_events()
    if core is None:
        return None
    if name == "Processed events (cleaned + sessions)":
        return core.events
    if name == "Session table":
        return core.sessions.reset_index()
    if name == "Customer table":
        return core.customers.reset_index()
    if name == "Prediction snapshots":
        return common.get_snapshots("n_events", settings.DEFAULT_SNAPSHOT_EVENTS, settings.DEFAULT_SNAPSHOT_CUTOFF_MIN).data.reset_index()
    return None


def render():
    ui.page_header("Dataset Explorer", "Inspect, search, sort, page through and export the original and processed datasets.")
    names = ["Raw events.csv (as stored)", "Processed events (cleaned + sessions)", "Session table", "Customer table",
             "Prediction snapshots"]
    name = st.selectbox("Dataset", names, key="exp_ds")
    if name == "Raw events.csv (as stored)" and not data_loader.discover_files()["events"].exists:
        ui.empty_state("events.csv missing", data_loader.SETUP_INSTRUCTION, "📂")
        return
    if name != "Raw events.csv (as stored)":
        common.require_core()
    with ui.loading_state("Loading dataset…"):
        df = _dataset(name)
    if df is None:
        ui.empty_state("Dataset not available", data_loader.SETUP_INSTRUCTION, "🗂️")
        return
    if name == "Prediction snapshots":
        st.caption("Default policy: snapshot after the 3rd non-transaction event (see Model Training for other policies).")

    k1, k2, k3 = st.columns(3)
    k1.metric("Rows", f"{len(df):,}")
    k2.metric("Columns", df.shape[1])
    k3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    t_data, t_stats, t_quality, t_dict = st.tabs(["Data", "Descriptive statistics", "Missing & duplicates", "Data dictionary"])
    with t_data:
        cols = st.multiselect("Columns to show", list(df.columns), default=list(df.columns), key=f"cols_{name}")
        if not cols:
            ui.empty_state("No columns selected", "Pick at least one column.", "🧱")
            return
        f1, f2, f3, f4 = st.columns([1.3, 1, 1.4, 1])
        fcol = f1.selectbox("Filter column", ["(none)"] + list(df.columns), key=f"fc_{name}")
        op = f2.selectbox("Operator", ["contains", "==", ">=", "<=", "is missing"], key=f"op_{name}")
        val = f3.text_input("Value", key=f"fv_{name}")
        sort_col = f4.selectbox("Sort by", ["(file order)"] + list(df.columns), key=f"sc_{name}")
        asc = st.checkbox("Ascending", value=True, key=f"asc_{name}")
        view = df
        try:
            if fcol != "(none)":
                s = view[fcol]
                if op == "is missing":
                    view = view[s.isna()]
                elif val != "":
                    if op == "contains":
                        view = view[s.astype("string").str.contains(val, case=False, na=False, regex=False)]
                    else:
                        num = pd.api.types.is_numeric_dtype(s)
                        v = float(val) if num else val
                        view = view[(s == v) if op == "==" else (s >= v) if op == ">=" else (s <= v)]
        except ValueError:
            st.error(f"“{val}” is not a valid value for column {fcol}.")
            return
        if sort_col != "(file order)":
            view = view.sort_values(sort_col, ascending=asc, kind="stable")
        total = len(view)
        if total == 0:
            ui.empty_state("No rows match", "Adjust or clear the filter.", "🔍")
            return
        p1, p2, p3 = st.columns([1, 1, 3])
        size = p1.selectbox("Rows per page", [25, 50, 100, 500], index=1, key=f"ps_{name}")
        pages = max(1, int(np.ceil(total / size)))
        page = p2.number_input("Page", 1, pages, 1, key=f"pg_{name}")
        p3.caption(f"{total:,} matching rows · page {page}/{pages}")
        page_df = view.iloc[(page - 1) * size: page * size][cols].copy()
        if "timestamp" in page_df.columns:
            page_df.insert(list(page_df.columns).index("timestamp") + 1, "datetime_utc", pd.to_datetime(page_df["timestamp"], unit="ms"))
        st.dataframe(page_df, use_container_width=True, hide_index=True)
        e1, e2 = st.columns(2)
        e1.download_button("⬇️ Export this page (CSV)", page_df.to_csv(index=False).encode(), f"{name[:20].strip()}_page.csv".replace(" ", "_"), "text/csv")
        cap = 500_000
        if total <= cap:
            e2.download_button(f"⬇️ Export all {total:,} filtered rows (CSV)", view[cols].to_csv(index=False).encode(),
                               "filtered_export.csv", "text/csv")
        else:
            e2.info(f"Filtered export is limited to {cap:,} rows; narrow the filter to export ({total:,} rows now).")
    with t_stats:
        num = df.select_dtypes("number")
        st.markdown("**Data types**")
        st.dataframe(df.dtypes.astype(str).rename("dtype").to_frame(), use_container_width=True)
        if len(num.columns):
            st.markdown("**Numeric summary**")
            st.dataframe(num.describe().T, use_container_width=True)
        cat = df.select_dtypes(exclude="number")
        if len(cat.columns):
            st.markdown("**Categorical summary**")
            st.dataframe(cat.describe().T, use_container_width=True)
    with t_quality:
        q = data_validation.describe_dataframe_quality(df)
        st.markdown("**Missing values per column**")
        st.dataframe(q[["dtype", "missing", "missing_pct", "unique"]], use_container_width=True)
        dup = int(df.duplicated().sum())
        st.metric("Fully duplicated rows", f"{dup:,}", f"{dup / max(len(df), 1) * 100:.3f}% of rows", delta_color="off")
    with t_dict:
        rows = [{"column": c, "dtype": str(df[c].dtype), "origin": DICTIONARY.get(c, ("—", "—"))[1],
                 "description": DICTIONARY.get(c, ("(no description)", "—"))[0]} for c in df.columns]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
