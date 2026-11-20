"""Page 1: Dashboard (all numbers computed from the filtered events)."""
import pandas as pd
import streamlit as st

from app_pages import common
from components.streamlit_components import ui
from src import analytics, charts
from config import settings


def render():
    ui.page_header("Dashboard", "Real KPIs and charts computed from the Retailrocket events; every widget responds to the filters.")
    core = common.require_core()
    ev_all = core.events

    tmin = pd.to_datetime(ev_all["timestamp"].min(), unit="ms").date()
    tmax = pd.to_datetime(ev_all["timestamp"].max(), unit="ms").date()
    with st.container(border=True):
        st.markdown('<div class="card-title">Filters</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([1.3, 1.2, 1, 1.3])
        dr = c1.date_input("Date range", (tmin, tmax), min_value=tmin, max_value=tmax, key="dash_dates")
        types = c2.multiselect("Event type", settings.VALID_EVENTS, default=settings.VALID_EVENTS, key="dash_types")
        vtxt = c3.text_input("Visitor ID (optional)", key="dash_visitor", placeholder="e.g. 257597")
        cats = None
        if "categoryid" in ev_all.columns and ev_all["categoryid"].notna().any():
            top = ev_all["categoryid"].value_counts().head(200).index.astype(int).tolist()
            cats = c4.multiselect("Product category", top, key="dash_cats", help="Top 200 categories by events (as-of join)")
        else:
            c4.caption("Category filter needs item properties: enable it on Data Preprocessing.")
    if isinstance(dr, tuple) and len(dr) == 2:
        start, end = dr
    else:
        start = end = dr[0] if isinstance(dr, tuple) else dr
    vids = None
    if vtxt.strip():
        if vtxt.strip().isdigit():
            vids = [int(vtxt.strip())]
        else:
            st.error("Visitor ID must be a number.")
            st.stop()
    if not types:
        ui.empty_state("No event type selected", "Select at least one event type.", "🎛️")
        st.stop()

    ev = analytics.filter_events(ev_all, start, end, types, vids, cats)
    if ev.empty:
        ui.empty_state("No events match the filters", "Widen the date range or clear the visitor/category filters.", "🔍")
        st.stop()
    k = analytics.compute_kpis(ev)
    ui.kpi_row([
        ("Total events", common.fmt_int(k["total_events"]), "rows after filters"),
        ("Unique visitors", common.fmt_int(k["unique_visitors"]), "distinct visitorid"),
        ("Unique sessions", common.fmt_int(k["unique_sessions"]), f"engineered, {common.cfg()['timeout']:g}-min timeout"),
        ("Product views", common.fmt_int(k["views"]), "event = view"),
        ("Add-to-cart events", common.fmt_int(k["addtocart"]), "event = addtocart"),
        ("Transactions", common.fmt_int(k["transactions"]), "transaction rows (items)"),
        ("Session purchase rate", f"{k['session_purchase_rate']:.2f}%", f"{common.fmt_int(k['purchase_sessions'])} sessions with a purchase"),
        ("Avg session duration", f"{k['avg_session_duration_sec']:.0f} s", "mean of (last − first event); 1-event sessions = 0"),
    ])
    st.caption("Session metrics count sessions that contain at least one of the selected events. "
               "“Transaction rate” = sessions with a transaction ÷ sessions.")

    a, b = st.columns(2)
    with a, ui.chart_card("Event distribution"):
        st.plotly_chart(charts.bar(analytics.event_distribution(ev), "event", "count", color="event"), use_container_width=True)
    with b, ui.chart_card("Event funnel: view → add to cart → transaction"):
        level = st.radio("Count by", ["visitors", "sessions", "events"], horizontal=True, key="dash_funnel", label_visibility="collapsed")
        f = analytics.funnel(ev, level)
        st.plotly_chart(charts.funnel_fig(f), use_container_width=True)

    with ui.chart_card("Event trends over time", "Daily or weekly counts per event type"):
        freq = st.radio("Granularity", ["Daily", "Weekly"], horizontal=True, key="dash_freq", label_visibility="collapsed")
        ts = analytics.time_series(ev, "D" if freq == "Daily" else "W")
        st.plotly_chart(charts.line(ts[ts["event"] != "transaction"], "period", "count", color="event"), use_container_width=True)
    a, b = st.columns(2)
    with a, ui.chart_card("Purchase trend", "Transactions and unique buyers per period"):
        tx = ev[ev["event"] == "transaction"]
        if tx.empty:
            ui.empty_state("No transactions in selection", "Include the transaction event type or widen the date range.", "🧾")
        else:
            d = pd.DataFrame({"period": pd.to_datetime(tx["timestamp"], unit="ms").dt.to_period("D" if freq == "Daily" else "W").dt.start_time,
                              "visitorid": tx["visitorid"].to_numpy()})
            g = d.groupby("period").agg(transactions=("visitorid", "size"), buyers=("visitorid", "nunique")).reset_index()
            st.plotly_chart(charts.line(g.melt("period", var_name="metric", value_name="count"), "period", "count", color="metric"),
                            use_container_width=True)
    with b, ui.chart_card("Visitor activity", "Events per visitor (log-scale bars)"):
        act = analytics.visitor_activity_distribution(ev)
        h = analytics.hist_counts(act, [0, 1, 2, 3, 5, 10, 20, 50, 100, 1e9])
        h["bin"] = ["1", "2", "3", "4–5", "6–10", "11–20", "21–50", "51–100", ">100"][: len(h)]
        st.plotly_chart(charts.hist_bar(h, "Events per visitor"), use_container_width=True)
    a, b = st.columns(2)
    with a, ui.chart_card("Top product interactions", "Top 15 items by selected event"):
        by = st.selectbox("Rank by", ["view", "addtocart", "transaction"], key="dash_topby", label_visibility="collapsed")
        ti = analytics.top_items(ev, 15, by)
        ti["item"] = ti["itemid"].astype(str)
        st.plotly_chart(charts.bar(ti.melt("item", ["view", "addtocart", "transaction"], var_name="event", value_name="count"),
                                   "count", "item", color="event", orientation="h", text_auto=False), use_container_width=True)
    with b, ui.chart_card("Session behaviour", "Events per session"):
        ss = analytics.session_stats(ev)
        h = analytics.hist_counts(ss["events"], [0, 1, 2, 3, 5, 10, 20, 50, 1e9])
        h["bin"] = ["1", "2", "3", "4–5", "6–10", "11–20", "21–50", ">50"][: len(h)]
        st.plotly_chart(charts.hist_bar(h, "Events per session"), use_container_width=True)
