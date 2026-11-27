"""
PAGE 5 — Session Segmentation (Section 7 & Section 8):
  - Unsupervised K-Means Clustering on Session Behavior
  - Controls:
      * Feature selection (9 default behavioral features + optional PageValues, Month, VisitorType, Weekend)
      * Revenue is strictly excluded from clustering inputs
      * Number of clusters (K)
      * Random seed
      * Maximum iterations
      * Scaling configuration & log1p skewness option
  - Displays:
      * Elbow Method (Inertia vs K)
      * Silhouette Score comparison across K
      * Cluster size & percentage
      * 2-D PCA visualization
      * Cluster profile table & average behavior by cluster
      * Post-clustering Revenue (purchase rate) analysis for interpretation only
      * Data-driven cluster interpretations
      * CSV export of cluster assignments
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src import clustering, data_loader, data_validation, ui

ui.inject_css()
df_raw = data_loader.require_dataset()

ui.page_header(
    title="Session Behavior Segmentation (K-Means Clustering)",
    subtitle="Discover groups of shopping sessions with similar browsing behavior using K-Means, Elbow & Silhouette evaluation, 2-D PCA, and post-clustering purchase rate analysis.",
    category="Page 5 · Unsupervised Learning (Task 2)",
    badges=[
        ("Algorithm: K-Means Clustering", "ok"),
        ("Target 'Revenue' Excluded from Inputs", "info"),
        ("Unit of Analysis: Shopping Sessions", "neutral"),
    ],
)

ui.note(
    "<b>Terminology & Methodological Safeguard:</b> Each row in the UCI dataset represents an individual "
    "<b>shopping session</b> rather than a verified long-term customer identity. Therefore, clusters are "
    "interpreted as <b>Session Behavior Segments</b>. Furthermore, <code>Revenue</code> is <b>never included</b> "
    "in the K-Means input features; purchase rates are compared <i>after</i> clustering strictly for interpretation."
)

# =============================================================================
# 1. CLUSTERING CONFIGURATION CONTROLS
# =============================================================================
all_allowed_clustering_features = (
    data_validation.DEFAULT_CLUSTERING_FEATURES + data_validation.OPTIONAL_CLUSTERING_FEATURES
)

with ui.chart_card(
    "Configure K-Means Session Segmentation Parameters",
    "Start with the 9 core behavioral features or optionally include PageValues, Month, VisitorType, and Weekend.",
):
    with st.form("kmeans_session_form"):
        sel_feats = st.multiselect(
            "Select Clustering Input Features (Revenue is strictly forbidden and blocked)",
            options=all_allowed_clustering_features,
            default=data_validation.DEFAULT_CLUSTERING_FEATURES,
        )
        c1, c2, c3, c4, c5 = st.columns(5)
        k_clusters = int(c1.slider("Number of Clusters (K)", min_value=2, max_value=8, value=4))
        random_seed = int(c2.number_input("Random Seed", min_value=0, max_value=9999, value=42))
        max_iter = int(c3.number_input("Max Iterations", min_value=50, max_value=1000, value=300, step=50))
        scaler_choice = c4.selectbox(
            "Numerical Scaling",
            options=["standard", "robust", "minmax"],
            index=0,
            format_func=lambda s: {"standard": "StandardScaler", "robust": "RobustScaler", "minmax": "MinMaxScaler"}[s],
        )
        use_log1p = bool(c5.checkbox("log1p Skewed Features", value=True))
        drop_dups = bool(st.checkbox("Remove 125 Exact Duplicate Sessions Before Clustering", value=bool(st.session_state.get("prep_drop_duplicates", False))))

        run_km_btn = st.form_submit_button("Run K-Means Session Segmentation", icon=":material/hub:", type="primary", use_container_width=True)

if run_km_btn or "session_clustering_result" not in st.session_state:
    if len(sel_feats) < 2:
        st.error("Please select at least 2 features for K-Means clustering.")
        st.stop()
    with st.spinner(f"Evaluating K=2..8 (Elbow & Silhouette) and fitting K-Means (K={k_clusters})..."):
        st.session_state["session_clustering_result"] = clustering.run_session_segmentation(
            df_raw,
            selected_features=sel_feats,
            n_clusters=k_clusters,
            k_min=2,
            k_max=8,
            random_state=random_seed,
            max_iter=max_iter,
            scaler_name=scaler_choice,
            use_log1p=use_log1p,
            drop_duplicates=drop_dups,
        )

seg_res = st.session_state["session_clustering_result"]
pca_var = seg_res["pca_explained_variance_pct"]

# =============================================================================
# 2. CLUSTERING SUMMARY KPIs & ELBOW / SILHOUETTE CHARTS
# =============================================================================
ui.kpi_row(
    [
        ("Session Segments (K)", f"K = {seg_res['n_clusters']}", f"Seed: {seg_res['random_state']} · Max iter: {seg_res['max_iter']}"),
        ("Sessions Clustered", f"{seg_res['n_sessions']:,}", f"{len(seg_res['selected_features'])} behavioral features"),
        ("Silhouette Score", f"{seg_res['silhouette_score']:.4f}", f"Davies–Bouldin Index: {seg_res['davies_bouldin_index']:.3f}"),
        ("Within-Cluster Inertia", f"{seg_res['inertia']:,.1f}", "Sum of squared distances to centroids"),
        ("2-D PCA Variance Explained", f"{sum(pca_var):.1f}%", f"PC1: {pca_var[0]:.1f}% · PC2: {pca_var[1]:.1f}%"),
        ("Revenue Excluded from Inputs", "Verified ✅", "Used only for post-clustering interpretation"),
    ],
    per_row=3,
)

k_df = pd.DataFrame(seg_res["k_evaluation"])
e1, e2 = st.columns(2)
with e1, ui.chart_card("1. Elbow Method (Inertia vs. Number of Clusters K)", "Within-cluster sum of squared errors across K = 2 to 8"):
    fig_elbow = px.line(
        k_df,
        x="k",
        y="inertia",
        markers=True,
        color_discrete_sequence=["#2F855A"],
        labels={"k": "Number of Clusters (K)", "inertia": "Inertia (Within-Cluster SSE)"},
    )
    fig_elbow.add_vline(x=seg_res["n_clusters"], line_dash="dash", line_color="#153E2D")
    ui.apply_plotly_theme(fig_elbow, height=300)
    st.plotly_chart(fig_elbow, use_container_width=True)

with e2, ui.chart_card("2. Silhouette Score Comparison Across K", "Cluster cohesion and separation quality across K = 2 to 8 (higher is better)"):
    fig_sil = px.bar(
        k_df,
        x="k",
        y="silhouette",
        text="silhouette",
        color="silhouette",
        color_continuous_scale=["#A7C957", "#2F855A", "#153E2D"],
        labels={"k": "Number of Clusters (K)", "silhouette": "Silhouette Score"},
    )
    fig_sil.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    ui.apply_plotly_theme(fig_sil, height=300)
    fig_sil.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_sil, use_container_width=True)

# =============================================================================
# 3. 2-D PCA VISUALIZATION, CLUSTER SIZE & POST-CLUSTERING PURCHASE RATE
# =============================================================================
profiles = seg_res["profiles"]
prof_df = pd.DataFrame(profiles)
prof_df["Short_Label"] = [f"Segment {p['cluster_id']}" for p in profiles]

p1, p2 = st.columns([1.35, 1])
with p1, ui.chart_card("3. Two-Dimensional PCA Projection of Session Behavior Segments", "Sessions projected onto the first two principal components of the scaled feature matrix"):
    pca_df = pd.DataFrame(seg_res["pca_sample"])
    fig_pca = px.scatter(
        pca_df,
        x="PCA_1",
        y="PCA_2",
        color="Session_Segment",
        opacity=0.68,
        hover_data=["ProductRelated", "BounceRates", "ExitRates", "VisitorType", "Revenue"],
        color_discrete_sequence=ui.PLOTLY_COLORWAY,
        labels={
            "PCA_1": f"Principal Component 1 ({pca_var[0]:.1f}% variance)",
            "PCA_2": f"Principal Component 2 ({pca_var[1]:.1f}% variance)",
        },
    )
    fig_pca.update_traces(marker=dict(size=6))
    ui.apply_plotly_theme(fig_pca, height=360)
    st.plotly_chart(fig_pca, use_container_width=True)

with p2, ui.chart_card("4. Cluster Size & Session Percentage", "Number of sessions and share (%) assigned to each segment"):
    fig_sz = px.bar(
        prof_df,
        x="Short_Label",
        y="size",
        color="segment_name",
        text="share_pct",
        hover_data=["segment_name", "size"],
        color_discrete_sequence=ui.PLOTLY_COLORWAY,
        labels={"Short_Label": "Segment", "size": "Session Count"},
    )
    fig_sz.update_traces(texttemplate="%{y:,} (%{text:.1f}%)", textposition="outside")
    ui.apply_plotly_theme(fig_sz, height=360)
    fig_sz.update_layout(showlegend=False)
    st.plotly_chart(fig_sz, use_container_width=True)

r1, r2 = st.columns(2)
with r1, ui.chart_card(
    "5. Post-Clustering Revenue Analysis (Purchase Rate by Segment)",
    "Revenue is evaluated AFTER clustering for interpretation only — never used as a K-Means input",
):
    fig_rev = px.bar(
        prof_df,
        x="purchase_rate_posthoc_pct",
        y="segment_name",
        orientation="h",
        text="purchase_rate_posthoc_pct",
        color="purchase_rate_posthoc_pct",
        hover_data=["size", "purchase_sessions_posthoc"],
        color_continuous_scale=["#A7C957", "#2F855A", "#153E2D"],
        labels={"purchase_rate_posthoc_pct": "Post-Hoc Purchase Rate (%)", "segment_name": "Session Segment"},
    )
    fig_rev.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    ui.apply_plotly_theme(fig_rev, height=320)
    fig_rev.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_rev, use_container_width=True)

with r2, ui.chart_card("6. Standardized Feature Profile Comparison (Z-Scores by Segment)", "How each segment deviates from the dataset average across core browsing dimensions"):
    z_rows = []
    for p in profiles:
        for feat_name, z_val in p["feature_z_scores"].items():
            z_rows.append({
                "Segment": f"Seg {p['cluster_id']}",
                "Feature": feat_name,
                "Z-Score": z_val,
            })
    z_df = pd.DataFrame(z_rows)
    fig_z = px.bar(
        z_df,
        x="Feature",
        y="Z-Score",
        color="Segment",
        barmode="group",
        color_discrete_sequence=ui.PLOTLY_COLORWAY,
    )
    ui.apply_plotly_theme(fig_z, height=320)
    st.plotly_chart(fig_z, use_container_width=True)

# =============================================================================
# 4. DATA-DRIVEN CLUSTER INTERPRETATIONS & PROFILE TABLE
# =============================================================================
with ui.chart_card("Data-Driven Session Behavior Segment Interpretations", "Generated dynamically from actual cluster centroid means and z-scores"):
    cols = st.columns(2)
    for idx, p in enumerate(profiles):
        with cols[idx % 2]:
            st.markdown(
                f'<div class="kpi">'
                f'<div class="l">Cluster {p["cluster_id"]} · {p["size"]:,} sessions ({p["share_pct"]}%) · '
                f'Post-Hoc Purchase Rate: <b>{p["purchase_rate_posthoc_pct"]}%</b></div>'
                f'<div class="v" style="font-size:1.12rem;margin-top:0.25rem">{ui.esc(p["segment_name"])}</div>'
                f'<div class="h" style="font-size:0.84rem;color:#263238;margin-top:0.35rem">{p["interpretation"]}</div>'
                f'<div class="h" style="margin-top:0.3rem">Dominant VisitorType: <code>{ui.esc(p["dominant_visitor_type"])}</code> · '
                f'Dominant Month: <code>{ui.esc(p["dominant_month"])}</code></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

with ui.chart_card("Cluster Profile Table (Average Feature Values & Post-Clustering Revenue Rate)", "Complete numerical comparison across all K session behavior segments"):
    table_rows = []
    for p in profiles:
        table_rows.append({
            "Cluster ID": p["cluster_id"],
            "Session Behavior Segment": p["segment_name"],
            "Sessions": p["size"],
            "Share (%)": p["share_pct"],
            "Post-Hoc Purchase Sessions": p["purchase_sessions_posthoc"],
            "Post-Hoc Purchase Rate (%)": p["purchase_rate_posthoc_pct"],
            "Dominant VisitorType": p["dominant_visitor_type"],
            **p["feature_means"],
        })
    profile_table_df = pd.DataFrame(table_rows)
    st.dataframe(profile_table_df, use_container_width=True, hide_index=True)

    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "Export Cluster Profile Summary Table (CSV)",
        profile_table_df.to_csv(index=False).encode("utf-8"),
        file_name="session_segment_profiles.csv",
        mime="text/csv",
        icon=":material/download:",
        use_container_width=True,
    )
    assign_df: pd.DataFrame = seg_res["assignments_df"]
    dl2.download_button(
        f"Export All Session Cluster Assignments ({len(assign_df):,} rows CSV)",
        assign_df.to_csv(index=False).encode("utf-8"),
        file_name="online_shoppers_cluster_assignments.csv",
        mime="text/csv",
        icon=":material/download:",
        use_container_width=True,
    )
