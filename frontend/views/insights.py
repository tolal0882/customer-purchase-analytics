"""
Exploratory Data Analysis & Customer Insights — 6-tab visual analysis of events,
visitors, sessions, purchase funnel, products/categories, and segment action playbook.
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
    cust_data = api_client.get_data(limit=2000, dataset="customers")
    sess_data = api_client.get_data(limit=2000, dataset="sessions")
except api_client.APIError as e:
    api_client.show_error_and_stop(e)

ui.page_header(
    "Exploratory Data Analysis & Behavioral Insights",
    "Event, visitor, session, purchase conversion, and product hierarchy analysis across the processed Retailrocket dataset.",
    badges=[
        (f"Visitors: {ins.get('unique_visitors', 1407580):,}", "ok"),
        (f"Sessions: {ins.get('total_sessions', 1761675):,}", "info"),
        (f"Items: {ins.get('unique_items', 235061):,}", "neutral"),
    ],
)

cust_df = pd.DataFrame(cust_data["rows"], columns=cust_data["columns"])
sess_df = pd.DataFrame(sess_data["rows"], columns=sess_data["columns"])

tabs = st.tabs([
    "A · Events",
    "B · Visitors",
    "C · Sessions",
    "D · Purchases & Cart Abandonment",
    "E · Products & Categories",
    "F · Segment Action Playbook",
])

with tabs[0]:
    c1, c2 = st.columns(2)
    with c1, ui.chart_card("Event Distribution", "Count of view, addtocart, and transaction events"):
        ev_c = ins.get("event_counts", {})
        ev_df = pd.DataFrame([{"event": k, "count": v} for k, v in ev_c.items()])
        fig_ev = px.bar(
            ev_df, x="event", y="count", color="event", text_auto=".3s",
            color_discrete_map={"view": "#2F855A", "addtocart": "#D69E2E", "transaction": "#153E2D"},
        )
        fig_ev.update_layout(height=300, showlegend=False, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_ev, use_container_width=True)

    with c2, ui.chart_card("Within-Session Event Transition Matrix (%)", "Row-normalized probability of the next event within the same session"):
        trans = ins.get("event_transitions", {})
        if trans:
            t_df = pd.DataFrame(trans).T
            t_pct = (t_df.div(t_df.sum(axis=1).replace(0, 1), axis=0) * 100).round(2)
            fig_tr = px.imshow(
                t_pct, text_auto=True, color_continuous_scale="Greens",
                labels={"x": "Next Event", "y": "Current Event", "color": "Transition (%)"},
            )
            fig_tr.update_layout(height=300, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_tr, use_container_width=True)

with tabs[1]:
    tot_v = ins.get("unique_visitors", 1)
    ret_v = ins.get("returning_visitors", 0)
    ui.kpi_row([
        ("Total Unique Visitors", f"{tot_v:,}", "Distinct visitorid"),
        ("Returning Visitors (≥2 Sessions)", f"{ret_v:,}", f"{ret_v / max(tot_v, 1) * 100:.1f}% of visitors"),
        ("Single-Session Visitors", f"{tot_v - ret_v:,}", f"{(tot_v - ret_v) / max(tot_v, 1) * 100:.1f}% of visitors"),
    ], per_row=3)

    if not cust_df.empty:
        a, b = st.columns(2)
        with a, ui.chart_card("Sessions per Engaged Customer", "Distribution split by conversion outcome"):
            fig_vs = px.histogram(cust_df, x="total_sessions", color="converted", nbins=30, color_discrete_map={"Yes": "#2F855A", "No": "#94A3B8"})
            fig_vs.update_layout(height=290, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_vs, use_container_width=True)
        with b, ui.chart_card("Product Views vs Unique Items Explored", "Browsing depth vs breadth across visitors"):
            fig_vv = px.scatter(
                cust_df.head(1000), x="total_views", y="unique_items_viewed", color="converted", opacity=0.68,
                color_discrete_map={"Yes": "#2F855A", "No": "#94A3B8"},
            )
            fig_vv.update_layout(height=290, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_vv, use_container_width=True)

with tabs[2]:
    ui.kpi_row([
        ("Total Engineered Sessions", f"{ins.get('total_sessions', 0):,}", "30-min inactivity threshold"),
        ("Single-Event Bounce Sessions", f"{ins.get('single_event_sessions', 0):,}", f"{ins.get('single_event_sessions_pct', 0):.1f}% of sessions"),
        ("Mean Session Duration", f"{ins.get('avg_session_duration_sec', 0):.1f} s", f"Mean {ins.get('avg_events_per_session', 0):.2f} events/session"),
    ], per_row=3)

    t1, t2 = st.columns(2)
    with t1, ui.chart_card("Hourly Activity Profile (UTC)", "Interaction volume by hour of day"):
        h_df = pd.DataFrame(ins.get("hourly_events", []))
        if not h_df.empty:
            fig_h = px.line(h_df, x="hour", y="events", markers=True, color_discrete_sequence=["#2F855A"])
            fig_h.update_layout(height=275, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_h, use_container_width=True)
    with t2, ui.chart_card("Activity by Day of Week", "Interaction volume across weekdays"):
        w_df = pd.DataFrame(ins.get("weekday_events", []))
        if not w_df.empty:
            fig_w = px.bar(w_df, x="day", y="events", color_discrete_sequence=["#153E2D"])
            fig_w.update_layout(height=275, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_w, use_container_width=True)

with tabs[3]:
    p1, p2 = st.columns(2)
    with p1, ui.chart_card("Conversion Rate by Early Add-to-Cart Actions", "Impact of cart additions during the first 3 session events"):
        cart_df = pd.DataFrame(ins.get("conversion_by_early_carts", []))
        if not cart_df.empty:
            cart_df["label"] = cart_df["add_to_cart_count"].astype(str) + " early carts"
            fig_c = px.bar(
                cart_df, x="label", y="conversion_rate", text="conversion_rate",
                color="conversion_rate", color_continuous_scale="Greens",
                labels={"label": "Early Add-to-Carts", "conversion_rate": "Conversion Rate (%)"},
            )
            fig_c.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_c.update_layout(height=300, coloraxis_showscale=False, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_c, use_container_width=True)

    with p2, ui.chart_card("Top Predictive Drivers of Purchase Conversion", "Feature importance from the active classification pipeline"):
        imp = ins.get("feature_importance", {})
        if imp:
            imp_s = pd.Series(imp).sort_values(ascending=True)
            fig_imp = px.bar(x=imp_s.values, y=imp_s.index, orientation="h", color_discrete_sequence=["#2F855A"])
            fig_imp.update_layout(height=300, xaxis_title="Importance", yaxis_title="Feature", plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_imp, use_container_width=True)

with tabs[4]:
    ui.kpi_row([
        ("Distinct Products Interacted", f"{ins.get('unique_items', 0):,}", "Unique itemid in events.csv"),
        ("Categories in category_tree.csv", f"{ins.get('total_categories', 0):,}", "Total category nodes"),
        ("Root Product Categories", f"{ins.get('root_categories', 0):,}", "Top-level parentid = NaN"),
    ], per_row=3)

    top_df = pd.DataFrame(ins.get("top_items", []))
    if not top_df.empty:
        with ui.chart_card("Top 15 Most Interacted Products", "Breakdown of views, cart additions, and purchases"):
            long_top = top_df.melt("itemid", ["view", "addtocart", "transaction"], var_name="event", value_name="count")
            fig_top = px.bar(
                long_top, x="count", y="itemid", color="event", orientation="h",
                color_discrete_map={"view": "#2F855A", "addtocart": "#D69E2E", "transaction": "#153E2D"},
            )
            fig_top.update_layout(height=330, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_top, use_container_width=True)

with tabs[5]:
    with ui.chart_card("Operational Playbook by Customer Segment", "Recommended marketing & UX actions for each K-Means cluster"):
        segs = ins.get("segments_overview", [])
        if segs:
            play_df = pd.DataFrame([
                {
                    "Segment": f"Cluster {s['cluster_id']}: {s['segment_name']}",
                    "Customers": s["size"],
                    "Share (%)": s["share_pct"],
                    "Conversion Rate (%)": s["conversion_rate_pct"],
                    "Behavioral Profile": s["description"],
                    "Recommended Operational Action": s["recommended_action"],
                }
                for s in segs
            ])
            st.dataframe(play_df, use_container_width=True, hide_index=True)
