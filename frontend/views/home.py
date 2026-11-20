"""
Page 1 – Home Dashboard: interactive filters (date range, event type),
real calculated Retailrocket KPIs, daily/weekly trends, conversion funnel,
top product interactions, and customer segment overview.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

import api_client
import config

st.title(f"{config.PROJECT_ICON} {config.PROJECT_TITLE}")
st.caption(config.PROJECT_SUBTITLE)
st.write(config.PROJECT_DESCRIPTION)

try:
    ins = api_client.get_insights()
    schema = api_client.get_schema()
except api_client.APIError as e:
    api_client.show_error_and_stop(e)

# =============================================================================
# 1. INTERACTIVE DASHBOARD FILTERS (DATE RANGE & EVENT TYPES)
# =============================================================================
daily_all = pd.DataFrame(ins.get("daily_trends", []))
with st.container(border=True):
    st.markdown("**Dashboard Filters**")
    f1, f2, f3 = st.columns([1.4, 1.4, 1])
    if not daily_all.empty:
        daily_all["dt"] = pd.to_datetime(daily_all["date"])
        d_min, d_max = daily_all["dt"].min().date(), daily_all["dt"].max().date()
        dr = f1.date_input("Date Range", (d_min, d_max), min_value=d_min, max_value=d_max)
    else:
        dr = None
    sel_events = f2.multiselect("Event Types", ["view", "addtocart", "transaction"], default=["view", "addtocart", "transaction"])
    gran = f3.radio("Trend Granularity", ["Daily", "Weekly"], horizontal=True)

if not daily_all.empty and isinstance(dr, tuple) and len(dr) == 2:
    mask = (daily_all["dt"].dt.date >= dr[0]) & (daily_all["dt"].dt.date <= dr[1])
    daily_filt = daily_all.loc[mask].copy()
else:
    daily_filt = daily_all.copy()

filt_views = int(daily_filt["view"].sum()) if ("view" in sel_events and not daily_filt.empty) else 0
filt_carts = int(daily_filt["addtocart"].sum()) if ("addtocart" in sel_events and not daily_filt.empty) else 0
filt_txs = int(daily_filt["transaction"].sum()) if ("transaction" in sel_events and not daily_filt.empty) else 0
filt_total = filt_views + filt_carts + filt_txs

# =============================================================================
# 2. REAL CALCULATED KPI CARDS
# =============================================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Events (Filtered)", f"{filt_total:,}", f"{ins.get('duplicates_removed', 0)} duplicates cleaned")
c2.metric("Unique Visitors", f"{ins.get('unique_visitors', 0):,}", f"{ins.get('returning_visitors', 0):,} returning visitors")
c3.metric("Unique Sessions", f"{ins.get('total_sessions', 0):,}", f"{ins.get('session_timeout_minutes', 30):g}-min inactivity timeout")
c4.metric("Avg Session Duration", f"{ins.get('avg_session_duration_sec', 0):.0f} sec", f"Avg {ins.get('avg_events_per_session', 0):.2f} events/session")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Product Views", f"{filt_views:,}", "event = view")
c6.metric("Add-to-Cart Events", f"{filt_carts:,}", f"Cart abandonment {ins.get('cart_abandonment_rate_pct', 0):.1f}%")
c7.metric("Transactions", f"{filt_txs:,}", f"Overall session tx rate {ins.get('session_conversion_rate_pct', 0):.2f}%")
test_m = ins.get("test_metrics", {})
c8.metric("Active ML Classifier", ins.get("best_model") or "Trained ✅", f"Test ROC-AUC: {test_m.get('roc_auc', '—')} · F1: {test_m.get('f1', '—')}")

# =============================================================================
# 3. EVENT DISTRIBUTION, FUNNEL & DAILY/WEEKLY EVENT TRENDS
# =============================================================================
a, b = st.columns(2)
with a, st.container(border=True):
    st.subheader("Event Distribution")
    ev_df = pd.DataFrame([
        {"event": "view", "count": filt_views},
        {"event": "addtocart", "count": filt_carts},
        {"event": "transaction", "count": filt_txs},
    ])
    ev_df = ev_df[ev_df["event"].isin(sel_events)]
    fig_ev = px.bar(
        ev_df, x="event", y="count", color="event", text_auto=".3s",
        color_discrete_map={"view": "#2F855A", "addtocart": "#D69E2E", "transaction": "#153E2D"},
    )
    fig_ev.update_layout(height=300, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_ev, use_container_width=True)

with b, st.container(border=True):
    st.subheader("Event Funnel: View → Add to Cart → Transaction")
    funnel_df = pd.DataFrame(ins.get("session_funnel", []))
    if not funnel_df.empty:
        fig_f = px.funnel(funnel_df, x="count", y="stage", color_discrete_sequence=["#2F855A"])
        fig_f.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_f, use_container_width=True)

if not daily_filt.empty and sel_events:
    with st.container(border=True):
        st.subheader(f"{gran} Event & Purchase Trends")
        tdf = daily_filt.copy()
        if gran == "Weekly":
            tdf["period"] = tdf["dt"].dt.to_period("W").dt.start_time
            tdf = tdf.groupby("period")[["view", "addtocart", "transaction"]].sum().reset_index()
        else:
            tdf["period"] = tdf["dt"]
        t1, t2 = st.columns(2)
        with t1:
            non_tx = [e for e in sel_events if e != "transaction"] or sel_events
            long_df = tdf.melt("period", non_tx, var_name="event", value_name="count")
            fig_tr = px.line(
                long_df, x="period", y="count", color="event",
                title="Browsing & Cart Activity Over Time",
                color_discrete_map={"view": "#2F855A", "addtocart": "#D69E2E", "transaction": "#153E2D"},
            )
            fig_tr.update_layout(height=300, margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig_tr, use_container_width=True)
        with t2:
            fig_tx = px.line(
                tdf, x="period", y="transaction",
                title="Purchase Transactions Over Time",
                color_discrete_sequence=["#153E2D"],
            )
            fig_tx.update_layout(height=300, margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig_tx, use_container_width=True)

# =============================================================================
# 4. TOP PRODUCT INTERACTIONS & CUSTOMER SEGMENTS OVERVIEW
# =============================================================================
p1, p2 = st.columns(2)
with p1, st.container(border=True):
    st.subheader("Top 15 Product Interactions")
    top_df = pd.DataFrame(ins.get("top_items", []))
    if not top_df.empty:
        long_top = top_df.melt("itemid", ["view", "addtocart", "transaction"], var_name="event", value_name="count")
        fig_top = px.bar(
            long_top, x="count", y="itemid", color="event", orientation="h",
            color_discrete_map={"view": "#2F855A", "addtocart": "#D69E2E", "transaction": "#153E2D"},
        )
        fig_top.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_top, use_container_width=True)

with p2, st.container(border=True):
    st.subheader("Customer Segments Overview (K-Means)")
    segs = ins.get("segments_overview", [])
    if segs:
        sdf = pd.DataFrame(segs)[["cluster_id", "segment_name", "size", "share_pct", "conversion_rate_pct"]]
        sdf.columns = ["Cluster", "Segment Name", "Customers", "Share (%)", "Conversion (%)"]
        st.dataframe(sdf, use_container_width=True, hide_index=True)
