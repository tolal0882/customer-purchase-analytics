"""Page 5: Customer & Session Segmentation (K-Means)."""
import numpy as np
import pandas as pd
import streamlit as st

from app_pages import common
from components.streamlit_components import react_components as rc, ui
from config import settings
from src import charts, clustering as cl, model_manager


def _controls(kind: str, df: pd.DataFrame, default_feats: list[str], optional_feats: list[str]):
    all_feats = default_feats + [f for f in optional_feats if f in df.columns]
    c1, c2 = st.columns([2, 1])
    feats = c1.multiselect("Clustering features", all_feats, default=default_feats, key=f"{kind}_feats",
                           help="visitorid / session_id are never used as inputs.")
    outcome = [f for f in feats if f in cl.OUTCOME_FEATURES]
    if outcome:
        ui.note(f"<b>{', '.join(outcome)}</b> encode purchases. Using them as inputs makes the later purchase-rate comparison partly circular.", warn=True)
    sk = cl.skewed_features(df, feats) if feats else []
    logf = c2.multiselect("log1p-transform (skewed)", feats, default=sk, key=f"{kind}_log",
                          help="Pre-selected where |skewness| > 1 (measured on this data).")
    s1, s2, s3, s4 = st.columns(4)
    scaler = s1.selectbox("Scaling", list(cl.SCALERS), key=f"{kind}_scaler", help="Applied after imputation and log transform; fitted on the fit sample only.")
    k = s2.slider("K (clusters)", 2, 10, 4, key=f"{kind}_k")
    seed = s3.number_input("Random seed", 0, 10_000, settings.DEFAULT_RANDOM_SEED, key=f"{kind}_seed")
    max_rows = s4.number_input("Max rows to fit", 10_000, 1_000_000, 200_000, 10_000, key=f"{kind}_maxrows",
                               help="Larger populations are randomly sampled (seeded) for fitting; ALL rows are then assigned.")
    return feats, logf, scaler, int(k), int(seed), int(max_rows)


def _results(kind: str, df: pd.DataFrame, idcol: str, label: str):
    r: cl.ClusterResult = st.session_state.get(f"seg_{kind}")
    ka = st.session_state.get(f"ka_{kind}")
    if ka is not None:
        a, b = st.columns(2)
        with a, ui.chart_card("Elbow method", f"Inertia vs K on a {min(60_000, len(df)):,}-row seeded sample"):
            st.plotly_chart(charts.elbow_fig(ka), use_container_width=True)
        with b, ui.chart_card("Silhouette score by K", "Higher is better; computed on a ≤20,000-point sample"):
            st.plotly_chart(charts.silhouette_fig(ka), use_container_width=True)
        st.dataframe(ka.round(4), use_container_width=True, hide_index=True)
    if r is None:
        ui.empty_state("No clustering fitted yet", "Choose features and K, then press <b>Fit K-Means</b>. Optionally run the elbow/silhouette analysis first.", "🧩")
        return
    ui.kpi_row([("Clusters (K)", str(r.k), f"seed {r.seed}, scaler {r.scaler}"), (f"{label} clustered", common.fmt_int(len(r.labels)), f"fit on {common.fmt_int(r.fit_rows)} rows"),
                ("Silhouette", f"{r.silhouette:.3f}", "sampled; −1…1"), ("Inertia", f"{r.inertia:,.0f}", f"Davies–Bouldin {r.davies_bouldin:.2f}")])
    ui.note("Clusters are <b>behaviour-based analytical groups</b> found by K-Means, not verified psychological or demographic customer types. "
            "Names below are generated from each cluster's most deviating features (z-scores) and may change with settings.")
    a, b = st.columns([1.4, 1])
    with a, ui.chart_card("PCA projection (2-D)", f"{len(r.pca_coords):,} sampled points · explained variance {sum(r.pca_explained) * 100:.1f}%"):
        st.plotly_chart(charts.pca_fig(r.pca_coords, r.names, r.pca_explained), use_container_width=True)
    with b, ui.chart_card("Cluster size and share"):
        st.plotly_chart(charts.cluster_size_fig(r.profile, r.names), use_container_width=True)

    clusters = [{"id": int(c), "name": r.names[c], "description": r.descriptions[c], "size": int(r.profile.loc[c, "size"]),
                 "share_pct": float(r.profile.loc[c, "share_pct"]),
                 "purchase_rate_pct": float(r.profile.loc[c, "purchase_rate_pct"]) if "purchase_rate_pct" in r.profile else None,
                 "z": {f: float(r.z_profile.loc[c, f]) for f in r.features}} for c in r.profile.index]
    with ui.chart_card("Cluster profiles", "Bars = cluster mean vs overall mean in standard deviations (click a card to select)"):
        sel = st.session_state.get(f"sel_{kind}")
        out = rc.cluster_explorer(clusters, sel, key=f"cx_{kind}")
        if out is not None:
            sel = out.get("selected", sel)
            st.session_state[f"sel_{kind}"] = sel
        elif not rc.react_available():
            st.caption("React build not found – showing native cards. Run `npm install && npm run build` in components/react.")
            cols = st.columns(min(len(clusters), 4))
            for col, c in zip(cols * 3, clusters):
                with col:
                    ui.cluster_profile_card(c["id"], c["name"], c["description"], c["size"], c["share_pct"], c["purchase_rate_pct"])
    if sel is not None and sel in r.profile.index:
        st.markdown(f"**Selected: cluster {sel} – {r.names[sel]}**")
        d = pd.DataFrame({"cluster mean": r.profile.loc[sel, r.features], "overall mean": df[r.features].mean(),
                          "z-score": r.z_profile.loc[sel]})
        st.dataframe(d.round(3), use_container_width=True)

    with ui.chart_card("Cluster profile table (means per cluster)"):
        t = r.profile.copy()
        t.insert(0, "name", [r.names[c] for c in t.index])
        st.dataframe(t.round(3), use_container_width=True)
        st.plotly_chart(charts.z_heatmap(r.z_profile, r.names), use_container_width=True)
    if "purchase_rate_pct" in r.profile:
        with ui.chart_card("Purchase behaviour by cluster", "has_purchase is NOT a clustering input (unless you selected an outcome feature above): compared after clustering"):
            pr = r.profile.reset_index()
            pr["cluster"] = pr["cluster"].map(lambda c: f"{c}: {r.names[c]}")
            st.plotly_chart(charts.bar(pr, "cluster", "purchase_rate_pct", color=None, height=320), use_container_width=True)
            overall = df["has_purchase"].mean() * 100
            st.caption(f"Overall purchase rate in the clustered population: {overall:.2f}%")

    with ui.chart_card("Cluster comparison"):
        picks = st.multiselect("Clusters to compare", list(r.profile.index), default=list(r.profile.index)[:2], key=f"{kind}_cmp",
                               format_func=lambda c: f"{c}: {r.names[c]}")
        if len(picks) >= 2:
            z = r.z_profile.loc[picks].T.reset_index().melt("index", var_name="cluster", value_name="z")
            z["cluster"] = z["cluster"].map(lambda c: f"{c}: {r.names[c]}")
            fig = charts.bar(z, "index", "z", color="cluster", color_map={}, text_auto=False, height=340)
            fig.update_layout(barmode="group", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Select at least two clusters.")

    e1, e2 = st.columns(2)
    out_df = r.labels.rename("cluster").reset_index()
    out_df.columns = [idcol, "cluster"]
    out_df["cluster_name"] = out_df["cluster"].map(r.names)
    e1.download_button("⬇️ Export cluster assignments (CSV)", out_df.to_csv(index=False).encode(), f"{kind}_cluster_assignments.csv", "text/csv")
    if e2.button("💾 Save clustering model (joblib)", key=f"save_{kind}"):
        p = model_manager.save_cluster_bundle(kind, cl.result_to_bundle(r), {"k": r.k, "features": r.features, "silhouette": r.silhouette,
                                                                         "names": {str(k): v for k, v in r.names.items()}})
        st.success(f"Saved {p.name}")


def _tab(kind: str, df_all: pd.DataFrame, idcol: str, label: str, defaults, optional, filt):
    df = filt(df_all)
    if len(df) < 50:
        ui.empty_state("Too few rows", "Relax the activity filter.", "🧮")
        return
    st.caption(f"{len(df):,} of {len(df_all):,} {label.lower()} in scope.")
    feats, logf, scaler, k, seed, max_rows = _controls(kind, df, defaults, optional)
    b1, b2, _ = st.columns([1, 1, 3])
    with st.expander("Elbow / silhouette range"):
        r1, r2 = st.columns(2)
        kmin = r1.number_input("K min", 2, 10, 2, key=f"{kind}_kmin")
        kmax = r2.number_input("K max", 3, 12, 8, key=f"{kind}_kmax")
    if not feats:
        ui.empty_state("No features selected", "Select at least one clustering feature.", "🧱")
        return
    if b1.button("📈 Run elbow & silhouette", key=f"{kind}_ka_btn"):
        bar = st.progress(0.0, "Starting…")
        st.session_state[f"ka_{kind}"] = cl.k_range_analysis(df, feats, logf, scaler, int(kmin), int(max(kmax, kmin + 1)), seed,
                                                              progress=lambda f, m: bar.progress(f, m))
        bar.empty()
    if b2.button("🧩 Fit & Save K-Means", type="primary", key=f"{kind}_fit_btn"):
        with ui.loading_state("Fitting & saving K-Means…"):
            r = cl.fit_kmeans(df, feats, k, seed, logf, scaler, max_rows)
            st.session_state[f"seg_{kind}"] = r
            st.session_state.pop(f"sel_{kind}", None)
            p = model_manager.save_cluster_bundle(
                kind,
                cl.result_to_bundle(r),
                {
                    "k": r.k,
                    "features": r.features,
                    "scaler": r.scaler,
                    "seed": r.seed,
                    "fit_rows": r.fit_rows,
                    "total_rows": len(r.labels),
                    "silhouette": r.silhouette,
                    "inertia": r.inertia,
                    "davies_bouldin": r.davies_bouldin,
                    "names": {str(c): v for c, v in r.names.items()},
                },
            )
            st.success(f"Fitted and saved `{p.name}` to `models/{kind}_clustering/`.")
    saved_runs = model_manager.list_cluster_bundles(kind)
    if saved_runs:
        with st.expander(f"Saved {kind} clustering models ({len(saved_runs)} in models/{kind}_clustering/)"):
            rows = [
                {
                    "file": m["file"],
                    "K": m.get("k"),
                    "silhouette": round(m["silhouette"], 4) if m.get("silhouette") is not None else None,
                    "scaler": m.get("scaler", "—"),
                    "features": len(m.get("features", [])),
                }
                for m in saved_runs
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    _results(kind, df, idcol, label)


def render():
    ui.page_header("Segmentation", "K-Means customer (visitor) segmentation and session segmentation with elbow, silhouette, PCA and data-derived profiles.")
    core = common.require_core()
    t1, t2 = st.tabs(["👥 Customers (visitors)", "🧾 Sessions"])
    with t1:
        ui.note("<b>Customer-level</b>: one row per visitorid built from that visitor's whole history (descriptive segmentation, not prediction). "
                "Most visitors have a single short session, so a minimum-activity filter is applied.")
        a, b = st.columns(2)
        ms = a.number_input("Keep visitors with at least … sessions", 1, 50, settings.DEFAULT_MIN_CUSTOMER_SESSIONS, key="cust_ms")
        me = b.number_input("…OR at least … events", 1, 500, settings.DEFAULT_MIN_CUSTOMER_EVENTS, key="cust_me")
        _tab("customer", core.customers, "visitorid", "Customers", cl.CUSTOMER_DEFAULT_FEATURES, cl.CUSTOMER_OPTIONAL_FEATURES,
             lambda d: d[(d["total_sessions"] >= ms) | (d["total_events"] >= me)])
    with t2:
        ui.note("<b>Session-level</b>: one row per engineered session (full-session behaviour, retrospective). Segment names come from the actual cluster profiles.")
        mev = st.number_input("Keep sessions with at least … events", 1, 50, 2, key="sess_me")
        _tab("session", core.sessions, "session_id", "Sessions", cl.SESSION_DEFAULT_FEATURES, [],
             lambda d: d[d["total_events"] >= mev])
