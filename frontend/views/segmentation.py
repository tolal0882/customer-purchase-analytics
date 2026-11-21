"""
Customer & Session Segmentation (K-Means) — Dual-scope behavioral clustering
with configurable features, K, scaling, log1p, Elbow/Silhouette evaluation,
React ClusterExplorer cards, 2-D PCA projection, and CSV export.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

import api_client
try:
    from components import react_bridge, ui
except ImportError:
    from frontend.components import react_bridge, ui

ui.inject_css()

ui.page_header(
    "Customer & Session Segmentation",
    "K-Means behavioral clustering on customer profiles and session profiles with Elbow, Silhouette, 2-D PCA, and data-derived z-score personas.",
    badges=[
        ("Algorithm: K-Means (Customer & Session)", "info"),
        ("Preprocessing: log1p + Scaler", "neutral"),
    ],
)

t_cust, t_sess = st.tabs(["👥 Customers (Visitor Profiles)", "🧾 Sessions (Session Behavior)"])


def _render_segmentation_scope(scope: str):
    try:
        seg = api_client.get_segments(scope=scope)
    except api_client.APIError as e:
        api_client.show_error_and_stop(e)

    avail_feats = seg.get("available_features") or seg.get("features", [])
    cur_feats = [f for f in seg.get("features", avail_feats) if f in avail_feats] or avail_feats

    with ui.chart_card(
        f"Configure {scope.capitalize()}-Level K-Means Clustering",
        "visitorid / session_id and purchase outcomes are never used as clustering inputs",
    ):
        with st.form(f"kmeans_cfg_{scope}"):
            sel_feats = st.multiselect("Clustering features", options=avail_feats, default=cur_feats)
            c1, c2, c3, c4 = st.columns(4)
            k_val = int(c1.slider("K (clusters)", min_value=2, max_value=8, value=int(seg.get("n_clusters", 4))))
            scaler_val = c2.selectbox(
                "Scaling", ["standard", "robust", "minmax"],
                index=["standard", "robust", "minmax"].index(seg.get("scaler", "standard")),
            )
            log_val = bool(c3.checkbox("log1p-transform skewed features", value=bool(seg.get("use_log1p", True))))
            seed_val = int(c4.number_input("Random seed", min_value=0, max_value=9999, value=int(seg.get("random_state", 42))))
            fit_btn = st.form_submit_button("🧩 Fit & Save K-Means Model", type="primary", use_container_width=True)

    if fit_btn:
        if len(sel_feats) < 2:
            st.error("Select at least 2 clustering features.")
        else:
            with st.spinner(f"Fitting {scope} K-Means (K={k_val}, scaler={scaler_val})..."):
                seg = api_client.fit_segments(
                    n_clusters=k_val,
                    scope=scope,
                    features=sel_feats,
                    scaler=scaler_val,
                    use_log1p=log_val,
                    random_state=seed_val,
                )

    pca_var = sum(seg.get("pca_explained_variance", [0, 0])) * 100
    ui.kpi_row([
        ("Clusters (K)", str(seg["n_clusters"]), f"Scaler: {seg.get('scaler', 'standard')} · Seed: {seg.get('random_state', 42)}"),
        ("Records Clustered", f"{seg['n_customers']:,}", f"Scope: {scope} profiles"),
        ("Silhouette Score", f"{seg['silhouette_score']:.4f}", f"Davies–Bouldin: {seg['davies_bouldin_index']:.3f}"),
        ("2-D PCA Variance", f"{pca_var:.1f}%", f"Inertia: {seg['inertia']:,.0f}"),
    ])

    k_df = pd.DataFrame(seg.get("k_evaluation", []))
    if not k_df.empty:
        e1, e2 = st.columns(2)
        with e1, ui.chart_card("Elbow Method", "Within-cluster sum of squared errors (Inertia) across K = 2..8"):
            fig_el = px.line(k_df, x="k", y="inertia", markers=True, color_discrete_sequence=["#2F855A"])
            fig_el.update_layout(height=265, xaxis_title="K (Clusters)", yaxis_title="Inertia", plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_el, use_container_width=True)
        with e2, ui.chart_card("Silhouette Score by K", "Cluster separation quality across K = 2..8 (higher is better)"):
            fig_sil = px.bar(k_df, x="k", y="silhouette", text="silhouette", color_discrete_sequence=["#A7C957"])
            fig_sil.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_sil.update_layout(height=265, xaxis_title="K (Clusters)", yaxis_title="Silhouette Score", plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_sil, use_container_width=True)

    profiles = seg.get("profiles", [])
    name_map = {p["cluster_id"]: f"C{p['cluster_id']}: {p['segment_name'].split(' (')[0]}" for p in profiles}

    with ui.chart_card("Cluster Profiles (Click a Segment Card to Inspect)", "Segment names and z-score bars are derived directly from cluster feature means"):
        selected_cid = st.session_state.get(f"sel_cid_{scope}", 0)
        if react_bridge.is_available():
            out = react_bridge.cluster_explorer(profiles, selected_id=selected_cid, key=f"seg_cards_{scope}_{seg['n_clusters']}")
            if out and isinstance(out, dict) and out.get("selected") is not None:
                selected_cid = int(out["selected"])
                st.session_state[f"sel_cid_{scope}"] = selected_cid

        active_prof = next((p for p in profiles if p["cluster_id"] == selected_cid), profiles[0] if profiles else None)
        if active_prof:
            ui.note(
                f"<b>Cluster {active_prof['cluster_id']} — {ui.esc(active_prof['segment_name'])}</b> · "
                f"<b>{active_prof['size']:,}</b> members ({active_prof['share_pct']}%) · "
                f"Post-clustering purchase rate: <b>{active_prof['conversion_rate_pct']}%</b><br>"
                f"<b>Behavioral Profile:</b> {ui.esc(active_prof['description'])}<br>"
                f"<b>Recommended Action:</b> {ui.esc(active_prof['recommended_action'])}"
            )

    p_left, p_mid, p_right = st.columns([1.3, 1, 1])
    with p_left, ui.chart_card("2-D PCA Cluster Projection", "Principal component scatter of clustered records"):
        pca_df = pd.DataFrame(seg.get("pca_sample", []))
        if not pca_df.empty:
            pca_df["Segment"] = pca_df["cluster"].map(name_map)
            ev = seg.get("pca_explained_variance", [0.0, 0.0])
            fig_pca = px.scatter(
                pca_df, x="pc1", y="pc2", color="Segment", opacity=0.68,
                hover_data=["id", "converted"],
                labels={"pc1": f"PC1 ({ev[0]*100:.1f}%)", "pc2": f"PC2 ({ev[1]*100:.1f}%)"},
            )
            fig_pca.update_traces(marker=dict(size=6))
            fig_pca.update_layout(height=330, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_pca, use_container_width=True)

    p_df = pd.DataFrame(profiles)
    if not p_df.empty:
        p_df["Segment"] = p_df["cluster_id"].map(name_map)
        with p_mid, ui.chart_card("Cluster Size & Population Share", "Distribution of records across clusters"):
            fig_sz = px.bar(p_df, x="Segment", y="size", text="share_pct", color="Segment", labels={"size": "Count"})
            fig_sz.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_sz.update_layout(height=330, showlegend=False, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_sz, use_container_width=True)

        with p_right, ui.chart_card("Post-Clustering Purchase Rate (%)", "Evaluated after clustering (not used as input)"):
            fig_conv = px.bar(
                p_df, x="conversion_rate_pct", y="Segment", orientation="h",
                text="conversion_rate_pct", color="Segment",
                labels={"conversion_rate_pct": "Purchase Rate (%)"},
            )
            fig_conv.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_conv.update_layout(height=330, showlegend=False, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_conv, use_container_width=True)

    with ui.chart_card("Cluster Summary Table & CSV Export", "Feature means across all clusters"):
        rows = [
            {
                "Cluster": p["cluster_id"],
                "Segment Name": p["segment_name"],
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
                f"⬇️ Export {scope.capitalize()} Cluster Assignments (CSV)",
                pca_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{scope}_cluster_assignments.csv",
                mime="text/csv",
                use_container_width=True,
            )


with t_cust:
    _render_segmentation_scope("customer")

with t_sess:
    _render_segmentation_scope("session")
