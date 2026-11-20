"""
Page 4 – Customer Segmentation: K-Means unsupervised clustering,
interactive React segment profile cards, 2-D PCA scatter visualization,
and an interactive tool to classify a customer profile into a segment.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

import api_client
try:
    from components import react_bridge
except ImportError:
    from frontend.components import react_bridge

st.title("👥 Customer Segmentation (K-Means)")
st.caption(
    "Group engaged customers by historical browsing and purchasing behavior using K-Means clustering "
    "(`log1p` skew transformation + `StandardScaler` + 2-D PCA)."
)

# =============================================================================
# 1. NUMBER OF CLUSTERS (K) SELECTOR & KPI METRICS
# =============================================================================
c_k, c_info = st.columns([1, 3])
k_choice = int(c_k.slider("Number of Customer Segments (K)", min_value=2, max_value=8, value=4))

try:
    seg = api_client.get_segments(k=k_choice)
except api_client.APIError as e:
    api_client.show_error_and_stop(e)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Algorithm", seg.get("algorithm", "K-Means"), f"K = {seg['n_clusters']} clusters")
m2.metric("Customers Segmented", f"{seg['n_customers']:,}", "Engaged multi-event visitors")
m3.metric("Silhouette Score", f"{seg['silhouette_score']:.4f}", f"Davies–Bouldin: {seg['davies_bouldin_index']:.3f}")
pca_var = sum(seg.get("pca_explained_variance", [0, 0])) * 100
m4.metric("2-D PCA Variance Explained", f"{pca_var:.1f}%", f"Inertia: {seg['inertia']:,.0f}")

# =============================================================================
# 2. INTERACTIVE SEGMENT PROFILE CARDS (REACT COMPONENT + NATIVE FALLBACK)
# =============================================================================
profiles = seg.get("profiles", [])
with st.container(border=True):
    st.subheader("Customer Segment Profiles")
    st.caption("Click a segment card to inspect its feature z-scores and recommended marketing strategy.")
    selected_cid = st.session_state.get("selected_cluster_id", 0)
    if react_bridge.is_available():
        out = react_bridge.cluster_explorer(profiles, selected_id=selected_cid, key=f"seg_cards_{k_choice}")
        if out and isinstance(out, dict) and out.get("selected") is not None:
            selected_cid = int(out["selected"])
            st.session_state["selected_cluster_id"] = selected_cid
    else:
        cols = st.columns(min(len(profiles), 4))
        for col, p in zip(cols * 2, profiles):
            with col:
                st.markdown(
                    f"**Cluster {p['cluster_id']}: {p['segment_name']}**  \n"
                    f"`{p['size']:,}` customers (`{p['share_pct']}%`) · Conversion: `{p['conversion_rate_pct']}%`  \n"
                    f"_{p['description']}_"
                )

    active_prof = next((p for p in profiles if p["cluster_id"] == selected_cid), profiles[0] if profiles else None)
    if active_prof:
        st.info(
            f"**Cluster {active_prof['cluster_id']} – {active_prof['segment_name']}** "
            f"(`{active_prof['size']:,}` customers, `{active_prof['share_pct']}%` of base, "
            f"`{active_prof['conversion_rate_pct']}%` conversion rate)  \n"
            f"**Profile:** {active_prof['description']}  \n"
            f"**Recommended Business Action:** {active_prof['recommended_action']}"
        )

# =============================================================================
# 3. 2-D PCA SCATTER VISUALIZATION & SEGMENT CONVERSION COMPARISON
# =============================================================================
left, right = st.columns([1.35, 1])
name_map = {p["cluster_id"]: f"C{p['cluster_id']}: {p['segment_name']}" for p in profiles}

with left:
    with st.container(border=True):
        st.subheader("2-D PCA Projection of Customer Segments")
        pca_df = pd.DataFrame(seg.get("pca_sample", []))
        if not pca_df.empty:
            pca_df["Segment"] = pca_df["cluster"].map(name_map)
            ev = seg.get("pca_explained_variance", [0.0, 0.0])
            fig_pca = px.scatter(
                pca_df, x="pc1", y="pc2", color="Segment", opacity=0.65,
                hover_data=["converted"],
                labels={
                    "pc1": f"PC1 ({ev[0] * 100:.1f}% variance)",
                    "pc2": f"PC2 ({ev[1] * 100:.1f}% variance)",
                },
            )
            fig_pca.update_traces(marker=dict(size=6))
            fig_pca.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_pca, use_container_width=True)

with right:
    with st.container(border=True):
        st.subheader("Conversion Rate by Segment (%)")
        st.caption("Purchase conversion is compared *after* clustering (not used as a clustering input).")
        p_df = pd.DataFrame(profiles)
        if not p_df.empty:
            p_df["Segment"] = p_df["cluster_id"].map(name_map)
            fig_conv = px.bar(
                p_df, x="conversion_rate_pct", y="Segment", orientation="h",
                text="conversion_rate_pct", color="Segment",
                labels={"conversion_rate_pct": "Conversion Rate (%)"},
            )
            fig_conv.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_conv.update_layout(height=380, showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_conv, use_container_width=True)

# =============================================================================
# 4. SEGMENT FEATURE AVERAGES TABLE & INTERACTIVE SEGMENT ASSIGNER
# =============================================================================
with st.container(border=True):
    st.subheader("Segment Behavioral Feature Averages")
    rows = []
    for p in profiles:
        rows.append({
            "Cluster": p["cluster_id"],
            "Segment Name": p["segment_name"],
            "Customers": p["size"],
            "Share (%)": p["share_pct"],
            "Conversion (%)": p["conversion_rate_pct"],
            **p["feature_means"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with st.expander("🔍 Interactive Tool: Assign a New Customer Profile to a Segment"):
    with st.form("seg_predict_form"):
        c1, c2, c3, c4 = st.columns(4)
        inp = {
            "total_sessions": c1.number_input("Total Sessions", 1, 500, 3),
            "total_views": c2.number_input("Total Product Views", 1, 2000, 12),
            "unique_items_viewed": c3.number_input("Unique Items Viewed", 1, 1000, 8),
            "total_add_to_cart": c4.number_input("Total Add-to-Carts", 0, 500, 2),
            "avg_session_duration_sec": c1.number_input("Avg Session Duration (sec)", 0.0, 10000.0, 240.0, 10.0),
            "avg_items_per_session": c2.number_input("Avg Items / Session", 1.0, 200.0, 3.0, 0.5),
            "cart_abandonment_rate": c3.slider("Cart Abandonment Rate (0–1)", 0.0, 1.0, 0.5, 0.05),
            "recency_days": c4.number_input("Days Since Last Visit", 0.0, 200.0, 15.0, 1.0),
        }
        run_seg = st.form_submit_button("Assign Customer Segment", type="primary")
    if run_seg:
        res_seg = api_client.predict_segment(inp)
        st.success(
            f"**Assigned Segment:** Cluster {res_seg['cluster_id']} – **{res_seg['segment_name']}** "
            f"(Segment conversion rate: `{res_seg['segment_conversion_rate_pct']}%`)  \n"
            f"**Recommended Action:** {res_seg['recommended_action']}"
        )
