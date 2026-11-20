"""
Page 7 – Exploratory Data Analysis (EDA) & Customer Insights:
  A. Event Analysis (distribution, daily frequency, within-session event transitions)
  B. Visitor Analysis (returning vs single-session visitors, engagement distribution)
  C. Session Analysis (duration, events/views per session, hourly & weekday patterns)
  D. Purchase & Cart Abandonment Analysis (conversion funnel, early cart impact)
  E. Product & Category Analysis (top products, category hierarchy)
  F. Actionable Business Playbook by Customer Segment
"""
import pandas as pd
import plotly.express as px
import streamlit as st

import api_client

st.title("💡 Exploratory Data Analysis & Customer Insights")
st.caption("Comprehensive 6-part analytical exploration of the Retailrocket dataset and actionable business recommendations.")

try:
    ins = api_client.get_insights()
    cust_data = api_client.get_data(limit=2000, dataset="customers")
    sess_data = api_client.get_data(limit=2000, dataset="sessions")
except api_client.APIError as e:
    api_client.show_error_and_stop(e)

cust_df = pd.DataFrame(cust_data["rows"], columns=cust_data["columns"])
sess_df = pd.DataFrame(sess_data["rows"], columns=sess_data["columns"])

tabs = st.tabs([
    "A · Event Analysis",
    "B · Visitor Analysis",
    "C · Session Analysis",
    "D · Purchase & Abandonment",
    "E · Product & Category Analysis",
    "F · Segment Action Playbook",
])

# =============================================================================
# A. EVENT ANALYSIS (DISTRIBUTION, TRENDS & TRANSITION MATRIX)
# =============================================================================
with tabs[0]:
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.subheader("Event Distribution")
        ev_c = ins.get("event_counts", {})
        ev_df = pd.DataFrame([{"event": k, "count": v} for k, v in ev_c.items()])
        fig_ev = px.bar(
            ev_df, x="event", y="count", color="event", text_auto=".3s",
            color_discrete_map={"view": "#2F855A", "addtocart": "#D69E2E", "transaction": "#153E2D"},
        )
        fig_ev.update_layout(height=310, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_ev, use_container_width=True)

    with c2, st.container(border=True):
        st.subheader("Within-Session Event Transition Matrix")
        trans = ins.get("event_transitions", {})
        if trans:
            t_df = pd.DataFrame(trans).T
            t_pct = (t_df.div(t_df.sum(axis=1).replace(0, 1), axis=0) * 100).round(2)
            fig_tr = px.imshow(
                t_pct, text_auto=True, color_continuous_scale="Greens",
                labels={"x": "Next Event in Session", "y": "Current Event", "color": "Transition (%)"},
            )
            fig_tr.update_layout(height=310, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_tr, use_container_width=True)
            st.caption("Row-normalized transition percentages between consecutive events in the same session.")

# =============================================================================
# B. VISITOR ANALYSIS
# =============================================================================
with tabs[1]:
    v1, v2, v3 = st.columns(3)
    tot_v = ins.get("unique_visitors", 1)
    ret_v = ins.get("returning_visitors", 0)
    v1.metric("Total Unique Visitors", f"{tot_v:,}")
    v2.metric("Returning Visitors (≥2 Sessions)", f"{ret_v:,}", f"{ret_v / max(tot_v, 1) * 100:.1f}% of visitors")
    v3.metric("Single-Session Visitors", f"{tot_v - ret_v:,}", f"{(tot_v - ret_v) / max(tot_v, 1) * 100:.1f}% of visitors")

    if not cust_df.empty:
        a, b = st.columns(2)
        with a, st.container(border=True):
            st.subheader("Sessions per Engaged Customer")
            fig_vs = px.histogram(cust_df, x="total_sessions", color="converted", nbins=30, color_discrete_map={"Yes": "#2F855A", "No": "#94A3B8"})
            fig_vs.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_vs, use_container_width=True)
        with b, st.container(border=True):
            st.subheader("Product Views vs Unique Items Explored")
            fig_vv = px.scatter(
                cust_df.head(1000), x="total_views", y="unique_items_viewed", color="converted", opacity=0.65,
                color_discrete_map={"Yes": "#2F855A", "No": "#94A3B8"},
            )
            fig_vv.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_vv, use_container_width=True)

# =============================================================================
# C. SESSION ANALYSIS (DURATION, EVENTS & TEMPORAL PATTERNS)
# =============================================================================
with tabs[2]:
    s1, s2, s3 = st.columns(3)
    s1.metric("Total Engineered Sessions", f"{ins.get('total_sessions', 0):,}")
    s2.metric("Single-Event Bounce Sessions", f"{ins.get('single_event_sessions', 0):,}", f"{ins.get('single_event_sessions_pct', 0):.1f}%")
    s3.metric("Mean Session Duration", f"{ins.get('avg_session_duration_sec', 0):.1f} sec", f"Mean {ins.get('avg_events_per_session', 0):.2f} events/session")

    t1, t2 = st.columns(2)
    with t1, st.container(border=True):
        st.subheader("Hourly Browsing & Shopping Activity (UTC)")
        h_df = pd.DataFrame(ins.get("hourly_events", []))
        if not h_df.empty:
            fig_h = px.line(h_df, x="hour", y="events", markers=True, color_discrete_sequence=["#2F855A"])
            fig_h.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_h, use_container_width=True)
    with t2, st.container(border=True):
        st.subheader("Activity by Day of Week")
        w_df = pd.DataFrame(ins.get("weekday_events", []))
        if not w_df.empty:
            fig_w = px.bar(w_df, x="day", y="events", color_discrete_sequence=["#153E2D"])
            fig_w.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_w, use_container_width=True)

    if not sess_df.empty:
        with st.container(border=True):
            st.subheader("Multi-Event Session Duration & Events Distribution")
            fig_sd = px.box(sess_df, x="converted", y="total_events", color="converted", color_discrete_map={"Yes": "#2F855A", "No": "#94A3B8"})
            fig_sd.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_sd, use_container_width=True)

# =============================================================================
# D. PURCHASE & CART ABANDONMENT ANALYSIS
# =============================================================================
with tabs[3]:
    p1, p2 = st.columns(2)
    with p1, st.container(border=True):
        st.subheader("Impact of Early Add-to-Cart Actions on Conversion")
        cart_df = pd.DataFrame(ins.get("conversion_by_early_carts", []))
        if not cart_df.empty:
            cart_df["label"] = cart_df["add_to_cart_count"].astype(str) + " early carts"
            fig_c = px.bar(
                cart_df, x="label", y="conversion_rate", text="conversion_rate",
                color="conversion_rate", color_continuous_scale="Greens",
                labels={"label": "Add-to-Carts in First 3 Events", "conversion_rate": "Conversion Rate (%)"},
            )
            fig_c.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_c.update_layout(height=310, coloraxis_showscale=False, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_c, use_container_width=True)

    with p2, st.container(border=True):
        st.subheader("Top Predictive Drivers of Purchase Conversion")
        imp = ins.get("feature_importance", {})
        if imp:
            imp_s = pd.Series(imp).sort_values(ascending=True)
            fig_imp = px.bar(x=imp_s.values, y=imp_s.index, orientation="h", color_discrete_sequence=["#2F855A"])
            fig_imp.update_layout(height=310, xaxis_title="Importance", yaxis_title="Feature", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_imp, use_container_width=True)

# =============================================================================
# E. PRODUCT & CATEGORY ANALYSIS
# =============================================================================
with tabs[4]:
    pr1, pr2 = st.columns([1.4, 1])
    with pr1, st.container(border=True):
        st.subheader("Top 15 Most Interacted Products")
        top_df = pd.DataFrame(ins.get("top_items", []))
        if not top_df.empty:
            long_top = top_df.melt("itemid", ["view", "addtocart", "transaction"], var_name="event", value_name="count")
            fig_top = px.bar(
                long_top, x="count", y="itemid", color="event", orientation="h",
                color_discrete_map={"view": "#2F855A", "addtocart": "#D69E2E", "transaction": "#153E2D"},
            )
            fig_top.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_top, use_container_width=True)
    with pr2, st.container(border=True):
        st.subheader("Product & Category Hierarchy Summary")
        st.metric("Distinct Products Interacted", f"{ins.get('unique_items', 0):,}")
        st.metric("Categories in category_tree.csv", f"{ins.get('total_categories', 0):,}", f"{ins.get('root_categories', 0)} root categories")
        if not top_df.empty:
            st.dataframe(top_df, use_container_width=True, hide_index=True)

# =============================================================================
# F. ACTIONABLE PLAYBOOK BY CUSTOMER SEGMENT
# =============================================================================
with tabs[5]:
    with st.container(border=True):
        st.subheader("Actionable Business Playbook by Customer Segment")
        segs = ins.get("segments_overview", [])
        if segs:
            play_df = pd.DataFrame([
                {
                    "Segment": f"Cluster {s['cluster_id']}: {s['segment_name']}",
                    "Customers": s["size"],
                    "Share (%)": s["share_pct"],
                    "Conversion Rate (%)": s["conversion_rate_pct"],
                    "Behavioral Profile": s["description"],
                    "Recommended Marketing & UX Action": s["recommended_action"],
                }
                for s in segs
            ])
            st.dataframe(play_df, use_container_width=True, hide_index=True)
