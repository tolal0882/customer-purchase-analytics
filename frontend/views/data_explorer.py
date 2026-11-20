"""
Page 2 – Data Explorer & Preprocessing Audit:
  - Tab 1: Dataset Explorer (preview, search/filter, sorting, pagination, column selection,
           data types, descriptive statistics, missing/duplicate summary, CSV export)
  - Tab 2: Data Preprocessing & Session Construction (schema validation, cleaning log,
           session timeout setting & sensitivity comparison, snapshot eligibility breakdown)
  - Tab 3: Data Dictionary
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import api_client

st.title("📊 Data Explorer & Preprocessing Audit")
st.caption("Inspect datasets, filter/sort/paginate rows, audit data cleaning & session construction, and export CSVs.")

try:
    ins = api_client.get_insights()
    schema = api_client.get_schema()
except api_client.APIError as e:
    api_client.show_error_and_stop(e)

tab_exp, tab_prep, tab_dict = st.tabs([
    "🗂️ 1. Dataset Explorer",
    "🧹 2. Data Preprocessing & Session Construction",
    "📖 3. Data Dictionary",
])

# =============================================================================
# TAB 1: DATASET EXPLORER (SEARCH, FILTER, SORT, PAGINATION, STATS, EXPORT)
# =============================================================================
with tab_exp:
    c_ds, c_lim = st.columns([2.4, 1])
    ds_choice = c_ds.selectbox(
        "Select Dataset to Explore",
        [
            "Early-Session Purchase Snapshots (clean.csv)",
            "Customer Behavioral Profiles (customer_profiles.csv)",
            "Session Behavioral Profiles (session_profiles.csv)",
            "Product Category Tree (category_tree.csv)",
        ],
    )
    ds_key = (
        "modeling" if ds_choice.startswith("Early")
        else "customers" if ds_choice.startswith("Customer")
        else "sessions" if ds_choice.startswith("Session")
        else "categories"
    )
    limit = int(c_lim.selectbox("Max rows to fetch", [500, 1500, 5000, 10000], index=1))

    data = api_client.get_data(limit=limit, dataset=ds_key)
    df = pd.DataFrame(data["rows"], columns=data["columns"])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Rows in Dataset", f"{data['total_rows']:,}")
    k2.metric("Loaded Sample Rows", f"{len(df):,}")
    k3.metric("Columns", len(df.columns))
    k4.metric("Duplicate Rows in Sample", f"{data.get('duplicate_rows', 0):,}")

    with st.container(border=True):
        f1, f2, f3, f4 = st.columns([1.4, 1.2, 1.2, 1])
        search_q = f1.text_input("Search text / value across rows", placeholder="e.g. Yes or visitorid")
        sort_col = f2.selectbox("Sort by column", list(df.columns))
        sort_asc = f3.selectbox("Sort order", ["Ascending", "Descending"]) == "Ascending"
        page_size = int(f4.selectbox("Rows per page", [20, 50, 100, 250], index=1))
        sel_cols = st.multiselect("Selected columns", list(df.columns), default=list(df.columns))

    view = df.copy()
    if search_q.strip():
        mask = view.astype(str).apply(lambda s: s.str.contains(search_q.strip(), case=False, na=False, regex=False)).any(axis=1)
        view = view[mask]
    view = view.sort_values(sort_col, ascending=sort_asc, kind="stable").reset_index(drop=True)

    total_pages = max(1, int(np.ceil(len(view) / page_size)))
    page_num = int(st.number_input(f"Page (1 to {total_pages})", min_value=1, max_value=total_pages, value=1))
    start_i = (page_num - 1) * page_size
    page_df = view.iloc[start_i : start_i + page_size][sel_cols or list(df.columns)]

    st.caption(f"Showing rows **{start_i + 1:,}–{min(start_i + page_size, len(view)):,}** of **{len(view):,}** matching rows.")
    st.dataframe(page_df, use_container_width=True, hide_index=True)

    st.download_button(
        f"⬇️ Export Filtered Dataset ({len(view):,} rows CSV)",
        view[sel_cols or list(df.columns)].to_csv(index=False).encode("utf-8"),
        file_name=f"retailrocket_{ds_key}_export.csv",
        mime="text/csv",
    )

    s1, s2 = st.columns([1.4, 1])
    num_cols = [c for c in view.select_dtypes(include="number").columns if c not in ("session_id", "visitorid", "session_start_ts")]
    with s1, st.container(border=True):
        st.subheader("Descriptive Statistics")
        if num_cols:
            st.dataframe(view[num_cols].describe().T.round(3), use_container_width=True)
    with s2, st.container(border=True):
        st.subheader("Column Data Types & Missing Values")
        qual_df = pd.DataFrame({
            "column": list(df.columns),
            "dtype": [data.get("dtypes", {}).get(c, str(df[c].dtype)) for c in df.columns],
            "missing": [int(df[c].isna().sum()) for c in df.columns],
            "unique": [int(df[c].nunique()) for c in df.columns],
        })
        st.dataframe(qual_df, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 2: DATA PREPROCESSING, CLEANING LOG & SESSION CONSTRUCTION
# =============================================================================
with tab_prep:
    st.subheader("1. Raw Dataset Inventory & Schema Validation")
    inv = ins.get("raw_files_inventory", [])
    if inv:
        st.dataframe(pd.DataFrame(inv), use_container_width=True, hide_index=True)

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Raw events.csv Rows", f"{ins.get('raw_rows', 2756101):,}", f"Memory: {ins.get('memory_raw_mb', 121.8)} MB")
    p2.metric("Exact Duplicates Removed", f"{ins.get('duplicates_removed', 460):,}", "0.017% of raw events")
    p3.metric("Clean Event Rows", f"{ins.get('clean_event_rows', 2755641):,}", f"Optimized: {ins.get('memory_clean_mb', 62.0)} MB")
    p4.metric("Invalid Timestamps / Events", f"{ins.get('invalid_rows_removed', 0):,}", "All timestamps valid UTC ms")

    c_left, c_right = st.columns(2)
    with c_left, st.container(border=True):
        st.subheader("2. Logged Data Cleaning Steps (No Silent Deletions)")
        steps = ins.get("cleaning_steps", [])
        if steps:
            st.dataframe(pd.DataFrame(steps), use_container_width=True, hide_index=True)
        raw_miss = ins.get("raw_missing_by_column", {})
        if raw_miss:
            st.caption(
                "**Raw missing-value audit:** "
                + ", ".join(f"`{k}`: {v:,}" for k, v in raw_miss.items())
                + " (`transactionid` is null on `view` and `addtocart` events by design)."
            )

    with c_right, st.container(border=True):
        st.subheader("3. Session Construction & Timeout Sensitivity")
        st.write(
            "Retailrocket does not include a native `session_id`. We sort events by `(visitorid, timestamp)` "
            "and start a new session whenever a visitor's inactivity gap exceeds the configured threshold "
            "(default **30 minutes**, an analytical assumption)."
        )
        sens = ins.get("timeout_sensitivity", [])
        if sens:
            st.dataframe(pd.DataFrame(sens), use_container_width=True, hide_index=True)

        with st.form("rebuild_sessions_form"):
            new_to = st.number_input(
                "Configure Session Inactivity Timeout (minutes)",
                min_value=5.0, max_value=180.0,
                value=float(ins.get("session_timeout_minutes", 30.0)), step=5.0,
            )
            rebuild_btn = st.form_submit_button("⚙️ Rebuild Sessions & Processed Datasets")
        if rebuild_btn:
            with st.spinner(f"Rebuilding sessions with {new_to:g}-minute timeout..."):
                try:
                    api_client.preprocess(timeout_minutes=new_to)
                    st.success(f"Rebuilt processed datasets with {new_to:g}-minute session timeout!")
                    st.rerun()
                except api_client.APIError as e:
                    st.error(str(e))

    with st.container(border=True):
        st.subheader("4. Leakage-Safe Early-Session Snapshot Eligibility Breakdown")
        elig = ins.get("snapshot_eligibility", {})
        if elig:
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Total Engineered Sessions", f"{elig.get('total_sessions', 0):,}")
            e2.metric("Excluded (<3 Non-Tx Events)", f"{elig.get('excluded_fewer_than_n_events', 0):,}")
            e3.metric("Excluded (Tx At/Before Snapshot)", f"{elig.get('excluded_transaction_at_or_before_snapshot', 0):,}", "Prevents outcome leakage")
            e4.metric("Usable Eligible Sessions", f"{elig.get('eligible_sessions_total', 0):,}", f"{elig.get('eligible_positive_rate_pct', 0):.2f}% positive rate")

# =============================================================================
# TAB 3: DATA DICTIONARY
# =============================================================================
with tab_dict:
    st.subheader("Retailrocket & Engineered Feature Dictionary")
    dict_rows = [
        {"Column": "timestamp", "Dataset": "raw/events.csv", "Origin": "Raw", "Description": "Unix epoch milliseconds of the interaction (UTC)."},
        {"Column": "visitorid", "Dataset": "raw & processed", "Origin": "Raw (Blocked from ML inputs)", "Description": "Unique visitor identifier. Used for grouping only, never as a numeric prediction feature."},
        {"Column": "event", "Dataset": "raw/events.csv", "Origin": "Raw", "Description": "Interaction category: view, addtocart, or transaction."},
        {"Column": "itemid", "Dataset": "raw/events.csv", "Origin": "Raw (Blocked from ML inputs)", "Description": "Unique product identifier."},
        {"Column": "transactionid", "Dataset": "raw/events.csv", "Origin": "Raw (Blocked from ML inputs)", "Description": "Transaction order ID (present only on transaction events)."},
        {"Column": "session_id", "Dataset": "clean.csv / session_profiles.csv", "Origin": "Engineered (Blocked from ML inputs)", "Description": "Generated session ID using the 30-minute inactivity threshold."},
        {"Column": "views_count", "Dataset": "clean.csv", "Origin": "Pre-Snapshot Input", "Description": "Product views observed in the first 3 non-transaction events."},
        {"Column": "unique_items_viewed", "Dataset": "clean.csv / profiles", "Origin": "Pre-Snapshot Input", "Description": "Distinct products viewed in the observation window."},
        {"Column": "add_to_cart_count", "Dataset": "clean.csv / session_profiles.csv", "Origin": "Pre-Snapshot Input", "Description": "Add-to-cart events in the observation window."},
        {"Column": "first_cart_position", "Dataset": "clean.csv", "Origin": "Pre-Snapshot Input", "Description": "0-based event position of first add-to-cart (-1 if none yet)."},
        {"Column": "elapsed_seconds", "Dataset": "clean.csv", "Origin": "Pre-Snapshot Input", "Description": "Seconds elapsed from session start to the 3rd event snapshot."},
        {"Column": "ends_with_cart", "Dataset": "clean.csv", "Origin": "Pre-Snapshot Input", "Description": "Yes if the 3rd event was an add-to-cart action, else No."},
        {"Column": "prior_sessions", "Dataset": "clean.csv", "Origin": "Historical Visitor Input", "Description": "Visitor's number of strictly earlier sessions prior to this session."},
        {"Column": "prior_purchases", "Dataset": "clean.csv", "Origin": "Historical Visitor Input", "Description": "Visitor's transactions in strictly earlier sessions only."},
        {"Column": "will_purchase", "Dataset": "clean.csv", "Origin": "Supervised Target", "Description": "Yes (1) if a transaction occurs strictly after the 3rd event in the session, else No (0)."},
    ]
    st.dataframe(pd.DataFrame(dict_rows), use_container_width=True, hide_index=True)
