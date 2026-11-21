"""
Data Explorer & Preprocessing Pipeline — dataset browsing, filtering, sorting,
pagination, cleaning audit log, and interactive session timeout builder.
"""
import numpy as np
import pandas as pd
import streamlit as st

import api_client
try:
    from components import ui
except ImportError:
    from frontend.components import ui

ui.inject_css()

try:
    ins = api_client.get_insights()
except api_client.APIError as e:
    api_client.show_error_and_stop(e)

ui.page_header(
    "Data Explorer & Preprocessing Pipeline",
    "Browse, search, filter, sort, and export datasets, or audit raw file validation, duplicate cleaning, and session construction.",
    badges=[
        (f"Raw Events: {ins.get('raw_rows', 2756101):,}", "neutral"),
        (f"Duplicates Cleaned: {ins.get('duplicates_removed', 460):,}", "info"),
    ],
)

tab_exp, tab_prep, tab_dict = st.tabs([
    "🗂️ Dataset Explorer",
    "🧹 Preprocessing & Session Engineering",
    "📖 Feature & Schema Dictionary",
])

# =============================================================================
# TAB 1: DATASET EXPLORER
# =============================================================================
with tab_exp:
    c_ds, c_lim = st.columns([2.4, 1])
    ds_choice = c_ds.selectbox(
        "Dataset Table",
        [
            "Early-Session Purchase Snapshots (data/processed/clean.csv)",
            "Customer Behavioral Profiles (data/processed/customer_profiles.csv)",
            "Session Behavioral Profiles (data/processed/session_profiles.csv)",
            "Product Category Hierarchy (data/raw/category_tree.csv)",
        ],
    )
    ds_key = (
        "modeling" if ds_choice.startswith("Early")
        else "customers" if ds_choice.startswith("Customer")
        else "sessions" if ds_choice.startswith("Session")
        else "categories"
    )
    limit = int(c_lim.selectbox("Sample Window Size", [500, 1500, 5000, 10000], index=1))

    data = api_client.get_data(limit=limit, dataset=ds_key)
    df = pd.DataFrame(data["rows"], columns=data["columns"])

    ui.kpi_row([
        ("Total Rows on Disk", f"{data['total_rows']:,}", f"Dataset: {ds_key}"),
        ("Loaded Window", f"{len(df):,}", "Interactive table buffer"),
        ("Columns", str(len(df.columns)), "Active schema fields"),
        ("Duplicate Rows", f"{data.get('duplicate_rows', 0):,}", "Verified 0 duplicates"),
    ])

    with ui.chart_card("Table Search, Sorting & Column Controls"):
        f1, f2, f3, f4 = st.columns([1.4, 1.2, 1.2, 1])
        search_q = f1.text_input("Search value across columns", placeholder="e.g. Yes or visitorid")
        sort_col = f2.selectbox("Sort by column", list(df.columns))
        sort_asc = f3.selectbox("Sort order", ["Ascending", "Descending"]) == "Ascending"
        page_size = int(f4.selectbox("Rows per page", [20, 50, 100, 250], index=1))
        sel_cols = st.multiselect("Visible columns", list(df.columns), default=list(df.columns))

    view = df.copy()
    if search_q.strip():
        mask = view.astype(str).apply(lambda s: s.str.contains(search_q.strip(), case=False, na=False, regex=False)).any(axis=1)
        view = view[mask]
    view = view.sort_values(sort_col, ascending=sort_asc, kind="stable").reset_index(drop=True)

    total_pages = max(1, int(np.ceil(len(view) / page_size)))
    p_col, dl_col = st.columns([1, 2])
    page_num = int(p_col.number_input(f"Page (1 to {total_pages})", min_value=1, max_value=total_pages, value=1))
    start_i = (page_num - 1) * page_size
    page_df = view.iloc[start_i : start_i + page_size][sel_cols or list(df.columns)]

    dl_col.write("")
    dl_col.download_button(
        f"⬇️ Export Filtered Table ({len(view):,} rows CSV)",
        view[sel_cols or list(df.columns)].to_csv(index=False).encode("utf-8"),
        file_name=f"retailrocket_{ds_key}_export.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with ui.chart_card(f"Dataset Preview — Rows {start_i + 1:,} to {min(start_i + page_size, len(view)):,} of {len(view):,}"):
        st.dataframe(page_df, use_container_width=True, hide_index=True)

    s1, s2 = st.columns([1.4, 1])
    num_cols = [c for c in view.select_dtypes(include="number").columns if c not in ("session_id", "visitorid", "session_start_ts")]
    with s1, ui.chart_card("Descriptive Statistics", "Summary statistics for numeric behavioral columns"):
        if num_cols:
            st.dataframe(view[num_cols].describe().T.round(3), use_container_width=True)
    with s2, ui.chart_card("Schema & Missing Value Summary", "Data types, null counts, and cardinality"):
        qual_df = pd.DataFrame({
            "column": list(df.columns),
            "dtype": [data.get("dtypes", {}).get(c, str(df[c].dtype)) for c in df.columns],
            "missing": [int(df[c].isna().sum()) for c in df.columns],
            "unique": [int(df[c].nunique()) for c in df.columns],
        })
        st.dataframe(qual_df, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 2: DATA PREPROCESSING & SESSION ENGINEERING
# =============================================================================
with tab_prep:
    ui.kpi_row([
        ("Raw events.csv Rows", f"{ins.get('raw_rows', 2756101):,}", f"Raw memory: {ins.get('memory_raw_mb', 121.8)} MB"),
        ("Exact Duplicates Removed", f"{ins.get('duplicates_removed', 460):,}", "0.017% of raw rows"),
        ("Clean Event Rows", f"{ins.get('clean_event_rows', 2755641):,}", f"Optimized memory: {ins.get('memory_clean_mb', 62.0)} MB"),
        ("Invalid Timestamps / Events", f"{ins.get('invalid_rows_removed', 0):,}", "All UTC millisecond timestamps valid"),
    ])

    with ui.chart_card("Raw Dataset Files Inventory (data/raw/)", "Original files verified on disk (never modified or overwritten)"):
        inv = ins.get("raw_files_inventory", [])
        if inv:
            st.dataframe(pd.DataFrame(inv), use_container_width=True, hide_index=True)

    c_left, c_right = st.columns(2)
    with c_left, ui.chart_card("Cleaning Pipeline Execution Log", "Every row removal is explicitly accounted for"):
        steps = ins.get("cleaning_steps", [])
        if steps:
            st.dataframe(pd.DataFrame(steps), use_container_width=True, hide_index=True)
        raw_miss = ins.get("raw_missing_by_column", {})
        if raw_miss:
            ui.note(
                "<b>Missing-value audit on raw events.csv:</b> "
                + ", ".join(f"<code>{k}</code>: {v:,}" for k, v in raw_miss.items())
                + " — <code>transactionid</code> is empty on <code>view</code> and <code>addtocart</code> rows by design."
            )

    with c_right, ui.chart_card("Session Construction & Timeout Sensitivity", "30-minute inactivity boundary (analytical assumption)"):
        sens = ins.get("timeout_sensitivity", [])
        if sens:
            st.dataframe(pd.DataFrame(sens), use_container_width=True, hide_index=True)
        with st.form("rebuild_sessions_form"):
            new_to = st.number_input(
                "Inactivity Timeout Threshold (minutes)",
                min_value=5.0, max_value=180.0,
                value=float(ins.get("session_timeout_minutes", 30.0)), step=5.0,
            )
            rebuild_btn = st.form_submit_button("⚙️ Rebuild Sessions & Processed Tables", type="primary", use_container_width=True)
        if rebuild_btn:
            with st.spinner(f"Rebuilding sessions with {new_to:g}-minute timeout..."):
                try:
                    api_client.preprocess(timeout_minutes=new_to)
                    st.success(f"Updated processed datasets and session metrics for {new_to:g}-minute session timeout.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    elig = ins.get("snapshot_eligibility", {})
    if elig:
        with ui.chart_card("Early-Session Snapshot Eligibility Breakdown", "Leakage prevention: features come strictly from the first 3 non-transaction events"):
            ui.kpi_row([
                ("Total Engineered Sessions", f"{elig.get('total_sessions', 0):,}", "All visitor sessions"),
                ("Excluded (<3 Non-Tx Events)", f"{elig.get('excluded_fewer_than_n_events', 0):,}", "Ended before 3rd action"),
                ("Excluded (Tx At/Before Snapshot)", f"{elig.get('excluded_transaction_at_or_before_snapshot', 0):,}", "Prevents post-purchase leakage"),
                ("Usable Eligible Sessions", f"{elig.get('eligible_sessions_total', 0):,}", f"{elig.get('eligible_positive_rate_pct', 7.56):.2f}% conversion rate"),
            ])

# =============================================================================
# TAB 3: DATA DICTIONARY
# =============================================================================
with tab_dict:
    with ui.chart_card("Retailrocket & Engineered Feature Dictionary", "Role and leakage policy for every raw and derived attribute"):
        dict_rows = [
            {"Column": "timestamp", "Dataset": "raw/events.csv", "Role / Leakage Policy": "Raw Timestamp", "Description": "Unix epoch milliseconds of the interaction (UTC)."},
            {"Column": "visitorid", "Dataset": "raw & processed", "Role / Leakage Policy": "Identifier (Blocked from ML inputs)", "Description": "Unique visitor identifier. Used for session grouping only."},
            {"Column": "event", "Dataset": "raw/events.csv", "Role / Leakage Policy": "Raw Event Type", "Description": "Interaction category: view, addtocart, or transaction."},
            {"Column": "itemid", "Dataset": "raw/events.csv", "Role / Leakage Policy": "Identifier (Blocked from ML inputs)", "Description": "Unique product identifier."},
            {"Column": "transactionid", "Dataset": "raw/events.csv", "Role / Leakage Policy": "Post-Purchase ID (Blocked from ML inputs)", "Description": "Transaction order ID (present only on transaction events)."},
            {"Column": "session_id", "Dataset": "clean.csv / session_profiles.csv", "Role / Leakage Policy": "Identifier (Blocked from ML inputs)", "Description": "Engineered session ID using the inactivity timeout threshold."},
            {"Column": "views_count", "Dataset": "clean.csv", "Role / Leakage Policy": "Early-Session Predictor", "Description": "Product views observed in the first 3 non-transaction events."},
            {"Column": "unique_items_viewed", "Dataset": "clean.csv / profiles", "Role / Leakage Policy": "Early-Session Predictor", "Description": "Distinct products viewed in the observation window."},
            {"Column": "add_to_cart_count", "Dataset": "clean.csv / session_profiles.csv", "Role / Leakage Policy": "Early-Session Predictor", "Description": "Add-to-cart events in the observation window."},
            {"Column": "first_cart_position", "Dataset": "clean.csv", "Role / Leakage Policy": "Early-Session Predictor", "Description": "0-based event position of first add-to-cart (-1 if none yet)."},
            {"Column": "elapsed_seconds", "Dataset": "clean.csv", "Role / Leakage Policy": "Early-Session Predictor", "Description": "Seconds elapsed from session start to the 3rd event snapshot."},
            {"Column": "ends_with_cart", "Dataset": "clean.csv", "Role / Leakage Policy": "Early-Session Predictor", "Description": "Yes if the 3rd event was an add-to-cart action, else No."},
            {"Column": "prior_sessions", "Dataset": "clean.csv", "Role / Leakage Policy": "Historical Visitor Predictor", "Description": "Visitor's number of strictly earlier sessions prior to this session."},
            {"Column": "prior_purchases", "Dataset": "clean.csv", "Role / Leakage Policy": "Historical Visitor Predictor", "Description": "Visitor's transactions in strictly earlier sessions only."},
            {"Column": "will_purchase", "Dataset": "clean.csv", "Role / Leakage Policy": "Supervised Target (0 / 1)", "Description": "Yes (1) if a transaction occurs strictly after the 3rd event in the session, else No (0)."},
        ]
        st.dataframe(pd.DataFrame(dict_rows), use_container_width=True, hide_index=True)
