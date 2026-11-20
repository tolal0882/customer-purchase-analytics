"""
Page 4 – Customer & Session Segmentation (K-Means Clustering):
  - Supports BOTH Customer Segmentation (visitor behavior profiles) AND Session Segmentation
  - Configurable features, K (2..8), random seed, scaling ('standard', 'robust', 'minmax'), log1p transform
  - Elbow Method & Silhouette comparison charts
  - 2-D PCA scatter visualization, cluster size & share, z-score profile cards (React + native),
    cluster comparison, post-clustering purchase rate, and CSV export of cluster assignments.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

import api_client
try:
    from components import react_bridge
except ImportError:
    from frontend.components import react_bridge

st.title("👥 Customer & Session Segmentation (K-Means)")
st.caption(
    "Group visitors and shopping sessions into data-driven behavioral clusters using K-Means. "
    "Note: Clusters are behavior-based analytical groups, not verified demographic or psychological customer types."
)

scope_tab = st.radio(
    "Select Segmentation Level",
    ["👥 Customer Segmentation (Visitor Profiles)", "🧾 Session Segmentation (Session Behavior)"],
    horizontal=True,
)
scope = "customer" if scope_tab.startswith("👥") else "session"

try:
    seg = api_client.get_segments(scope=scope)
except api_client.APIError as e:
    api_client.show_error_and_stop(e)

# =============================================================================
# 1. INTERACTIVE K-MEANS CONTROLS (FEATURES, K, SEED, SCALING, LOG1P)
# =============================================================================
avail_feats = seg.get("available_features") or seg.get("features", [])
cur_feats = [f for f in seg.get("features", avail_feats) if f in avail_feats] or avail_feats

with st.expander("⚙️ Configure K-Means Clustering Parameters (Features, K, Scaling, Random Seed)", expanded=False):
    with st.form(f"kmeans_cfg_{scope}"):
        sel_feats = st.multiselect(
            "Selected Behavioral Features (visitorid/session_id & post-purchase outcomes excluded)",
            options=avail_feats,
            default=cur_feats,
        )
        c1, c2, c3, c4 = st.columns(4)
        k_val = int(c1.slider("Number of Clusters (K)", min_value=2, max_value=8, value=int(seg.get("n_clusters", 4))))
        scaler_val = c2.selectbox("Numerical Scaling", ["standard", "robust", "minmax"], index=["standard", "robust", "minmax"].index(seg.get("scaler", "standard")))
        log_val = bool(c3.checkbox("Apply log1p to skewed features", value=bool(seg.get("use_log1p", True))))
        seed_val = int(c4.number_input("Random Seed", min_value=0, max_value=9999, value=int(seg.get("random_state", 42))))
        fit_btn = st.form_submit_button("🧩 Fit & Update K-Means Segmentation", type="primary")

if fit_btn:
    if len(sel_feats) < 2:
        st.error("Please select at least 2 clustering features.")
    else:
        with st.spinner(f"Fitting K-Means (K={k_val}, scaler={scaler_val})..."):
            seg = api_client.fit_segments(
                n_clusters=k_val,
                scope=scope,
                features=sel_feats,
                scaler=scaler_val,
                use_log1p=log_val,
                random_state=seed_val,
            )

# =============================================================================
# 2. HEADLINE CLUSTERING METRICS & ELBOW / SILHOUETTE CHARTS
# =============================================================================
m1, m2, m3, m4 = st.columns(4)
m1.metric("Clusters (K)", seg["n_clusters"], f"Scaler: {seg.get('scaler', 'standard')} · Seed: {seg.get('random_state', 42)}")
m2.metric("Records Clustered", f"{seg['n_customers']:,}", f"Scope: {scope.capitalize()}s")
m3.metric("Silhouette Score", f"{seg['silhouette_score']:.4f}", f"Davies–Bouldin: {seg['davies_bouldin_index']:.3f}")
pca_var = sum(seg.get("pca_explained_variance", [0, 0])) * 100
m4.metric("2-D PCA Variance", f"{pca_var:.1f}%", f"Inertia: {seg['inertia']:,.0f}")

k_df = pd.DataFrame(seg.get("k_evaluation", []))
if not k_df.empty:
    e1, e2 = st.columns(2)
    with e1, st.container(border=True):
        st.subheader("Elbow Method (Inertia vs K)")
        fig_el = px.line(k_df, x="k", y="inertia", markers=True, color_discrete_sequence=["#2F855A"])
        fig_el.update_layout(height=270, xaxis_title="K (Number of Clusters)", yaxis_title="Inertia (Within-Cluster SSE)", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_el, use_container_width=True)
    with e2, st.container(border=True):
        st.subheader("Silhouette Score Comparison Across K")
        fig_sil = px.bar(k_df, x="k", y="silhouette", text="silhouette", color_discrete_sequence=["#A7C957"])
        fig_sil.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_sil.update_layout(height=270, xaxis_title="K (Number of Clusters)", yaxis_title="Silhouette Score", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_sil, use_container_width=True)

# =============================================================================
# 3. INTERACTIVE CLUSTER PROFILES (REACT COMPONENT + NATIVE FALLBACK)
# =============================================================================
profiles = seg.get("profiles", [])
name_map = {p["cluster_id"]: f"C{p['cluster_id']}: {p['segment_name'].split(' (')[0]}" for p in profiles}

with st.container(border=True):
    st.subheader("Data-Driven Cluster Profiles (Generated from Actual Feature Z-Scores)")
    selected_cid = st.session_state.get(f"sel_cid_{scope}", 0)
    if react_bridge.is_available():
        out = react_bridge.cluster_explorer(profiles, selected_id=selected_cid, key=f"seg_cards_{scope}_{seg['n_clusters']}")
        if out and isinstance(out, dict) and out.get("selected") is not None:
            selected_cid = int(out["selected"])
            st.session_state[f"sel_cid_{scope}"] = selected_cid

    active_prof = next((p for p in profiles if p["cluster_id"] == selected_cid), profiles[0] if profiles else None)
    if active_prof:
        st.info(
            f"**Cluster {active_prof['cluster_id']} – {active_prof['segment_name']}**  \n"
            f"**Size:** `{active_prof['size']:,}` ({active_prof['share_pct']}% of population) · "
            f"**Post-Clustering Conversion Rate:** `{active_prof['conversion_rate_pct']}%`  \n"
            f"**Behavioral Explanation:** {active_prof['description']}  \n"
            f"**Recommended Business Action:** {active_prof['recommended_action']}"
        )

# =============================================================================
# 4. 2-D PCA PROJECTION, CLUSTER SIZE & POST-CLUSTERING PURCHASE BEHAVIOR
# =============================================================================
p_left, p_mid, p_right = st.columns([1.3, 1, 1])
with p_left, st.container(border=True):
    st.subheader("2-D PCA Cluster Visualization")
    pca_df = pd.DataFrame(seg.get("pca_sample", []))
    if not pca_df.empty:
        pca_df["Segment"] = pca_df["cluster"].map(name_map)
        ev = seg.get("pca_explained_variance", [0.0, 0.0])
        fig_pca = px.scatter(
            pca_df, x="pc1", y="pc2", color="Segment", opacity=0.65,
            hover_data=["id", "converted"],
            labels={"pc1": f"PC1 ({ev[0]*100:.1f}%)", "pc2": f"PC2 ({ev[1]*100:.1f}%)"},
        )
        fig_pca.update_traces(marker=dict(size=6))
        fig_pca.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_pca, use_container_width=True)

p_df = pd.DataFrame(profiles)
if not p_df.empty:
    p_df["Segment"] = p_df["cluster_id"].map(name_map)
    with p_mid, st.container(border=True):
        st.subheader("Cluster Size & Share")
        fig_sz = px.bar(
            p_df, x="Segment", y="size", text="share_pct", color="Segment",
            labels={"size": "Count"},
        )
        fig_sz.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_sz.update_layout(height=340, showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_sz, use_container_width=True)

    with p_right, st.container(border=True):
        st.subheader("Purchase Rate by Cluster (%)")
        st.caption("Compared *after* clustering (not used as input).")
        fig_conv = px.bar(
            p_df, x="conversion_rate_pct", y="Segment", orientation="h",
            text="conversion_rate_pct", color="Segment",
            labels={"conversion_rate_pct": "Purchase Rate (%)"},
        )
        fig_conv.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_conv.update_layout(height=340, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_conv, use_container_width=True)

# =============================================================================
# 5. CLUSTER PROFILE TABLE, Z-SCORE COMPARISON & CSV EXPORT
# =============================================================================
with st.container(border=True):
    st.subheader("Cluster Profile Table (Feature Means & Z-Score Comparison)")
    rows = [
        {
            "Cluster": p["cluster_id"],
            "Generated Segment Name": p["segment_name"],
            "Size": p["size"],
            "Share (%)": p["share_pct"],
            "Purchase Rate (%)": p["conversion_rate_pct"],
            **p["feature_means"],
        }
        for p in profiles
    ]
    prof_tbl = pd.DataFrame(rows)
    st.dataframe(prof_tbl, use_container_width=True, hide_index=True)

    c_exp1, c_exp2 = st.columns(2)
    c_exp1.download_button(
        f"⬇️ Export {scope.capitalize()} Cluster Profiles (CSV)",
        prof_tbl.to_csv(index=False).encode("utf-8"),
        file_name=f"{scope}_cluster_profiles.csv",
        mime="text/csv",
        use_container_width=True,
    )
    if not pca_df.empty:
        c_exp2.download_button(
            f"⬇️ Export {scope.capitalize()} Cluster Assignments & PCA Coordinates (CSV)",
            pca_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{scope}_cluster_assignments.csv",
            mime="text/csv",
            use_container_width=True,
        )

if scope == "customer":
    with st.expander("🔍 Interactive Customer Segment Assigner"):
        with st.form("seg_predict_form"):
            c1, c2, c3, c4 = st.columns(4)
            inp = {
                "total_sessions": c1.number_input("Total Sessions", 1, 500, 3),
                "total_views": c2.number_input("Total Product Views", 1, 2000, 12),
                "unique_items_viewed": c3.number_input("Unique Items Viewed", 1, 1000, 8),
                "total_add_to_cart": c4.number_input("Total Add-to-Carts", 0, 500, 2),
                "avg_session_duration_sec": c1.number_input("Avg Session Duration (sec)", 0.0, 10000.0, 240.0, 10.0),
                "avg_items_per_session": c2.number_input("Avg Items / Session", 1.0, 200.0, 3.0, 0.5),
                "avg_cart_additions_per_session": c3.number_input("Avg Carts / Session", 0.0, 50.0, 0.7, 0.1),
                "cart_abandonment_rate": c4.slider("Cart Abandonment Rate", 0.0, 1.0, 0.5, 0.05),
                "recency_days": st.number_input("Days Since Last Activity", 0.0, 200.0, 15.0, 1.0),
            }
            run_seg = st.form_submit_button("Assign Customer Segment", type="primary")
        if run_seg:
            res_seg = api_client.predict_segment(inp)
            st.success(
                f"**Assigned Segment:** Cluster {res_seg['cluster_id']} – **{res_seg['segment_name']}** "
                f"(Conversion rate: `{res_seg['segment_conversion_rate_pct']}%`)  \n"
                f"**Recommended Action:** {res_seg['recommended_action']}"
            )
