"""
Page 7 – Customer Insights: analyze e-commerce purchasing patterns,
conversion drivers, temporal shopping trends, and actionable business playbooks.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

import api_client

st.title("💡 Customer Insights & Business Recommendations")
st.caption("Actionable e-commerce conversion patterns derived from 2.75M Retailrocket interactions and our ML models.")

try:
    ins = api_client.get_insights()
except api_client.APIError as e:
    api_client.show_error_and_stop(e)

# =============================================================================
# 1. CONVERSION & ABANDONMENT HEADLINE METRICS
# =============================================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Overall Session Conversion", f"{ins.get('session_conversion_rate_pct', 0):.2f}%", "Across all 1.76M sessions")
c2.metric("Engaged Session Conversion", f"{ins.get('modeling_positive_rate_pct', 0):.2f}%", "Sessions with ≥3 early interactions")
c3.metric("Cart Abandonment Rate", f"{ins.get('cart_abandonment_rate_pct', 0):.1f}%", "Cart sessions without purchase")
c4.metric("Add-to-Cart Events", f"{ins.get('event_counts', {}).get('addtocart', 0):,}", f"Transactions: {ins.get('event_counts', {}).get('transaction', 0):,}")

# =============================================================================
# 2. EARLY ADD-TO-CART IMPACT & FEATURE DRIVERS OF CONVERSION
# =============================================================================
left, right = st.columns(2)
with left, st.container(border=True):
    st.subheader("Impact of Early Add-to-Cart Actions on Conversion")
    cart_df = pd.DataFrame(ins.get("conversion_by_early_carts", []))
    if not cart_df.empty:
        cart_df["add_to_cart_count"] = cart_df["add_to_cart_count"].astype(str) + " cart actions"
        fig_cart = px.bar(
            cart_df, x="add_to_cart_count", y="conversion_rate", text="conversion_rate",
            color="conversion_rate", color_continuous_scale="Greens",
            labels={"add_to_cart_count": "Add-to-Cart Actions in First 3 Events", "conversion_rate": "Conversion Rate (%)"},
        )
        fig_cart.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_cart.update_layout(height=320, coloraxis_showscale=False, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_cart, use_container_width=True)
    st.caption("Insight: Sessions with even 1 add-to-cart action in their first 3 interactions convert at a dramatically higher rate than view-only sessions.")

with right, st.container(border=True):
    st.subheader("Top Predictive Drivers of Purchase Conversion")
    imp = ins.get("feature_importance", {})
    if imp:
        imp_s = pd.Series(imp).sort_values(ascending=True).tail(10)
        fig_imp = px.bar(
            x=imp_s.values, y=imp_s.index, orientation="h",
            labels={"x": "Model Feature Importance", "y": "Feature"},
            color_discrete_sequence=["#2F855A"],
        )
        fig_imp.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_imp, use_container_width=True)
    st.caption("Derived from the trained supervised classification pipeline (`models/model.joblib`).")

# =============================================================================
# 3. TEMPORAL SHOPPING PATTERNS (HOURLY & DAY-OF-WEEK ACTIVITY)
# =============================================================================
t1, t2 = st.columns(2)
with t1, st.container(border=True):
    st.subheader("Hourly Browsing & Shopping Activity (UTC)")
    h_df = pd.DataFrame(ins.get("hourly_events", []))
    if not h_df.empty:
        fig_h = px.line(h_df, x="hour", y="events", markers=True, color_discrete_sequence=["#2F855A"])
        fig_h.update_layout(height=280, xaxis_title="Hour of Day (UTC)", yaxis_title="Total Events", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_h, use_container_width=True)

with t2, st.container(border=True):
    st.subheader("Browsing Activity by Day of Week")
    w_df = pd.DataFrame(ins.get("weekday_events", []))
    if not w_df.empty:
        fig_w = px.bar(w_df, x="day", y="events", color_discrete_sequence=["#153E2D"])
        fig_w.update_layout(height=280, xaxis_title="Day of Week", yaxis_title="Total Events", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_w, use_container_width=True)

# =============================================================================
# 4. ACTIONABLE PLAYBOOK BY K-MEANS CUSTOMER SEGMENT
# =============================================================================
with st.container(border=True):
    st.subheader("Actionable Business Playbook by Customer Segment")
    segs = ins.get("segments_overview", [])
    if segs:
        play_df = pd.DataFrame([
            {
                "Segment": f"Cluster {s['cluster_id']}: {s['segment_name']}",
                "Share (%)": s["share_pct"],
                "Conversion Rate (%)": s["conversion_rate_pct"],
                "Behavioral Profile": s["description"],
                "Recommended Marketing & UX Action": s["recommended_action"],
            }
            for s in segs
        ])
        st.dataframe(play_df, use_container_width=True, hide_index=True)
