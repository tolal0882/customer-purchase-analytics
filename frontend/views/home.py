"""
Dashboard — Real KPIs and charts computed from the Retailrocket events.
"""
import pandas as pd
import plotly.express as px
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

test_m = ins.get("test_metrics", {})
ui.page_header(
    "Executive Analytics Dashboard",
    "Real KPIs, conversion funnel, event trends, and behavioral segments computed from 2,755,641 cleaned Retailrocket events.",
    badges=[
        ("Dataset: Retailrocket (May–Sep 2015)", "info"),
        (f"Classifier: {ins.get('best_model', 'Logistic Regression')} (AUC {test_m.get('roc_auc', 0.848):.3f})", "neutral"),
    ],
)

# =============================================================================
# 1. INTERACTIVE DASHBOARD FILTERS
# =============================================================================
daily_all = pd.DataFrame(ins.get("daily_trends", []))
with st.container(border=True):
    st.markdown('<div class="card-title">Interactive Filters</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns([1.4, 1.4, 1])
    if not daily_all.empty:
        daily_all["dt"] = pd.to_datetime(daily_all["date"])
        d_min, d_max = daily_all["dt"].min().date(), daily_all["dt"].max().date()
        dr = f1.date_input("Date range", (d_min, d_max), min_value=d_min, max_value=d_max)
    else:
        dr = None
    sel_events = f2.multiselect("Event types", ["view", "addtocart", "transaction"], default=["view", "addtocart", "transaction"])
    gran = f3.radio("Trend granularity", ["Daily", "Weekly"], horizontal=True)

if not sel_events:
    ui.empty_state("No event type selected", "Select at least one event type above.", "🎛️")
    st.stop()

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
# 2. STYLED KPI CARDS ROW
# =============================================================================
ui.kpi_row([
    ("Total Events", f"{filt_total:,}", f"{ins.get('duplicates_removed', 460):,} exact duplicates removed"),
    ("Unique Visitors", f"{ins.get('unique_visitors', 1407580):,}", f"{ins.get('returning_visitors', 0):,} returning visitors"),
    ("Unique Sessions", f"{ins.get('total_sessions', 1761675):,}", f"engineered, {ins.get('session_timeout_minutes', 30):g}-min timeout"),
    ("Avg Session Duration", f"{ins.get('avg_session_duration_sec', 0):.0f} s", f"mean {ins.get('avg_events_per_session', 1.56):.2f} events / session"),
    ("Product Views", f"{filt_views:,}", "event = view"),
    ("Add-to-Cart Events", f"{filt_carts:,}", f"cart abandonment {ins.get('cart_abandonment_rate_pct', 68.5):.1f}%"),
    ("Transactions", f"{filt_txs:,}", f"{ins.get('converting_sessions', 0):,} purchase sessions"),
    ("Session Purchase Rate", f"{ins.get('session_conversion_rate_pct', 0.90):.2f}%", f"Test F1 {test_m.get('f1', 0.431):.3f} · ROC-AUC {test_m.get('roc_auc', 0.848):.3f}"),
])

# =============================================================================
# 3. EVENT DISTRIBUTION, FUNNEL & DAILY/WEEKLY EVENT TRENDS
# =============================================================================
a, b = st.columns(2)
with a, ui.chart_card("Event Distribution", "Filtered interaction volume by event category"):
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
    fig_ev.update_layout(height=300, showlegend=False, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_ev, use_container_width=True)

with b, ui.chart_card("Event Funnel: View → Add to Cart → Transaction", "Session-level progression from browsing to checkout"):
    funnel_df = pd.DataFrame(ins.get("session_funnel", []))
    if not funnel_df.empty:
        fig_f = px.funnel(funnel_df, x="count", y="stage", color_discrete_sequence=["#2F855A"])
        fig_f.update_layout(height=300, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_f, use_container_width=True)

if not daily_filt.empty and sel_events:
    tdf = daily_filt.copy()
    if gran == "Weekly":
        tdf["period"] = tdf["dt"].dt.to_period("W").dt.start_time
        tdf = tdf.groupby("period")[["view", "addtocart", "transaction"]].sum().reset_index()
    else:
        tdf["period"] = tdf["dt"]

    t1, t2 = st.columns(2)
    with t1, ui.chart_card(f"{gran} Event Trends Over Time", "Views and add-to-cart activity per period"):
        non_tx = [e for e in sel_events if e != "transaction"] or sel_events
        long_df = tdf.melt("period", non_tx, var_name="event", value_name="count")
        fig_tr = px.line(
            long_df, x="period", y="count", color="event",
            color_discrete_map={"view": "#2F855A", "addtocart": "#D69E2E", "transaction": "#153E2D"},
        )
        fig_tr.update_layout(height=290, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_tr, use_container_width=True)

    with t2, ui.chart_card(f"{gran} Purchase Trend", "Completed transaction volume over time"):
        fig_tx = px.line(tdf, x="period", y="transaction", color_discrete_sequence=["#153E2D"])
        fig_tx.update_layout(height=290, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_tx, use_container_width=True)

# =============================================================================
# 4. TOP PRODUCT INTERACTIONS & CUSTOMER SEGMENTS OVERVIEW
# =============================================================================
p1, p2 = st.columns(2)
with p1, ui.chart_card("Top Product Interactions", "Top 15 items ranked by total interactions"):
    top_df = pd.DataFrame(ins.get("top_items", []))
    if not top_df.empty:
        long_top = top_df.melt("itemid", ["view", "addtocart", "transaction"], var_name="event", value_name="count")
        fig_top = px.bar(
            long_top, x="count", y="itemid", color="event", orientation="h",
            color_discrete_map={"view": "#2F855A", "addtocart": "#D69E2E", "transaction": "#153E2D"},
        )
        fig_top.update_layout(height=330, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_top, use_container_width=True)

with p2, ui.chart_card("Customer Segments Overview", "Active K-Means behavioral clusters and post-clustering purchase rates"):
    segs = ins.get("segments_overview", [])
    if segs:
        sdf = pd.DataFrame(segs)[["cluster_id", "segment_name", "size", "share_pct", "conversion_rate_pct"]]
        sdf.columns = ["Cluster", "Segment Name", "Customers", "Share (%)", "Conversion (%)"]
        st.dataframe(sdf, use_container_width=True, hide_index=True)
