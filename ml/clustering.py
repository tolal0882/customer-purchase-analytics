"""
Unsupervised Learning Component – Customer Segmentation using K-Means.

Run standalone from the project root:
    python -m ml.clustering
Or automatically as part of:
    python -m ml.train

Outputs:
    models/clustering.joblib          fitted preprocessing + K-Means + PCA bundle
    models/clustering_metrics.json    Elbow inertia, Silhouette scores, segment profiles, 2-D PCA points
    reports/figures/elbow_silhouette.png
    reports/figures/clusters_pca.png
"""
from __future__ import annotations

import json
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from ml import config


# =============================================================================
# 1. PREPROCESSING FOR SKEWED CUSTOMER BEHAVIOR FEATURES
#    - E-commerce counts (views, sessions, duration) are heavily right-skewed.
#    - Applies median imputation -> log1p(max(x, 0)) -> StandardScaler.
# =============================================================================
def _transform_features(df: pd.DataFrame, features: list[str], imputer=None, scaler=None):
    X_raw = df[features].copy()
    if imputer is None:
        imputer = SimpleImputer(strategy="median").fit(X_raw)
    X_imp = imputer.transform(X_raw)
    X_log = np.log1p(np.clip(X_imp, 0, None))
    if scaler is None:
        scaler = StandardScaler().fit(X_log)
    X_scaled = scaler.transform(X_log)
    return X_scaled, imputer, scaler


# =============================================================================
# 2. ELBOW METHOD (INERTIA) & SILHOUETTE ANALYSIS ACROSS K = 2..8
# =============================================================================
def evaluate_k_range(X_scaled: np.ndarray, k_values=config.K_RANGE) -> list[dict]:
    """Compute Inertia (Elbow), Silhouette Score, and Davies-Bouldin Index for each K."""
    sil_sample = min(8_000, len(X_scaled))
    rows = []
    for k in k_values:
        km = KMeans(n_clusters=int(k), n_init=10, random_state=config.RANDOM_STATE)
        labels = km.fit_predict(X_scaled)
        sil = float(silhouette_score(X_scaled, labels, sample_size=sil_sample, random_state=config.RANDOM_STATE))
        dbi = float(davies_bouldin_score(X_scaled, labels))
        rows.append({
            "k": int(k),
            "inertia": round(float(km.inertia_), 2),
            "silhouette": round(sil, 4),
            "davies_bouldin": round(dbi, 4),
        })
    return rows


# =============================================================================
# 3. DATA-DRIVEN SEGMENT PROFILING & BUSINESS NAMING
#    - Names and marketing recommendations are derived from each cluster's
#      actual feature z-scores and post-clustering purchase conversion rate.
# =============================================================================
def _assign_segment_identity(z_row: pd.Series, conv_rate: float, overall_conv: float) -> tuple[str, str, str]:
    """Generate a clear business segment name, description, and recommended action from cluster statistics."""
    if z_row.get("total_add_to_cart", 0) > 0.6 and z_row.get("cart_abandonment_rate", 0) > 0.5:
        return (
            "Hesitant Cart Abandoners",
            "Frequently add products to cart but leave without completing checkout.",
            "Trigger exit-intent reminders, free-shipping thresholds, or time-limited cart discounts.",
        )
    if z_row.get("total_sessions", 0) > 0.6 or z_row.get("total_views", 0) > 0.7:
        if conv_rate >= overall_conv * 1.2:
            return (
                "High-Value Loyal Shoppers",
                "Returning multi-session visitors with deep product exploration and strong purchase conversion.",
                "Enroll in VIP loyalty tiers, early-access drops, and personalized cross-sell bundles.",
            )
        return (
            "Deep Window Explorers",
            "High browsing volume and long sessions across many items, but below-average checkout rate.",
            "Show social proof, product comparison tables, and price-drop alerts on viewed items.",
        )
    if z_row.get("recency_days", 0) > 0.5:
        return (
            "Dormant / At-Risk Visitors",
            "Visitors whose last browsing activity occurred long ago with low recent engagement.",
            "Run win-back re-engagement campaigns featuring trending products in their viewed history.",
        )
    if z_row.get("avg_session_duration_sec", 0) < -0.3:
        return (
            "Quick Bouncers / Casual Browsers",
            "Short, low-interaction visits with minimal product depth and very low cart activity.",
            "Optimize landing-page load speed, highlight top bestsellers, and simplify search filters.",
        )
    return (
        "Focused Everyday Shoppers",
        "Balanced session duration and moderate product comparisons with steady baseline activity.",
        "Offer relevant product recommendations and streamlined one-click checkout prompts.",
    )


def build_profiles(df: pd.DataFrame, labels: np.ndarray, features: list[str]) -> list[dict]:
    """Build detailed per-cluster profiles from actual customer data."""
    work = df.copy()
    work["cluster"] = labels
    overall_means = work[features].mean()
    overall_stds = work[features].std().replace(0, 1.0)
    overall_conv = float((work["converted"] == "Yes").mean() * 100) if "converted" in work.columns else 0.0

    profiles = []
    used_names: set[str] = set()
    for cid, grp in work.groupby("cluster", sort=True):
        means = grp[features].mean()
        z_scores = ((means - overall_means) / overall_stds).round(3)
        conv_pct = float((grp["converted"] == "Yes").mean() * 100) if "converted" in grp.columns else 0.0
        name, desc, action = _assign_segment_identity(z_scores, conv_pct, overall_conv)
        if name in used_names:
            name = f"{name} (Segment {int(cid) + 1})"
        used_names.add(name)

        profiles.append({
            "cluster_id": int(cid),
            "segment_name": name,
            "description": desc,
            "recommended_action": action,
            "size": int(len(grp)),
            "share_pct": round(float(len(grp) / len(work) * 100), 2),
            "conversion_rate_pct": round(conv_pct, 2),
            "avg_transactions": round(float(grp["total_transactions"].mean()), 3) if "total_transactions" in grp.columns else 0.0,
            "feature_means": {f: round(float(means[f]), 2) for f in features},
            "feature_z_scores": {f: round(float(z_scores[f]), 3) for f in features},
        })
    return profiles


# =============================================================================
# 4. SAVE CLUSTERING FIGURES (reports/figures/elbow_silhouette.png & clusters_pca.png)
# =============================================================================
def save_clustering_figures(k_table: list[dict], pca_points: list[dict], profiles: list[dict]) -> list[str]:
    out = config.FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)
    saved = []

    # Figure 1: Elbow & Silhouette side-by-side
    kdf = pd.DataFrame(k_table)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(kdf["k"], kdf["inertia"], marker="o", color="#2F855A", linewidth=2)
    axes[0].set_title("Elbow Method (Inertia vs K)")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Inertia (Within-Cluster SSE)")

    axes[1].bar(kdf["k"].astype(str), kdf["silhouette"], color="#4472C4")
    axes[1].set_title("Silhouette Score vs K")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Silhouette Score")
    fig.tight_layout()
    fig.savefig(out / "elbow_silhouette.png", dpi=150)
    plt.close(fig)
    saved.append("elbow_silhouette.png")

    # Figure 2: 2-D PCA Cluster Scatter
    pdf = pd.DataFrame(pca_points)
    name_map = {p["cluster_id"]: p["segment_name"] for p in profiles}
    fig, ax = plt.subplots(figsize=(7, 5))
    for cid, grp in pdf.groupby("cluster"):
        ax.scatter(grp["pc1"], grp["pc2"], s=14, alpha=0.6, label=f"C{cid}: {name_map.get(cid, cid)}")
    ax.set_title("Customer Segments – 2D PCA Projection")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out / "clusters_pca.png", dpi=150)
    plt.close(fig)
    saved.append("clusters_pca.png")
    return saved


# =============================================================================
# 5. MAIN TRAINING & PREDICTION FUNCTIONS
# =============================================================================
def run(n_clusters: int = config.DEFAULT_K_CLUSTERS, verbose: bool = True) -> dict:
    """Fit K-Means customer segmentation, evaluate K=2..8, and save artifacts."""
    if not config.CUSTOMER_PROFILES_PATH.exists():
        from ml.data import make_dataset
        make_dataset.main()

    df = pd.read_csv(config.CUSTOMER_PROFILES_PATH)
    features = [f for f in config.CLUSTERING_FEATURES if f in df.columns]
    X_scaled, imputer, scaler = _transform_features(df, features)

    k_table = evaluate_k_range(X_scaled, config.K_RANGE)
    km = KMeans(n_clusters=int(n_clusters), n_init=10, random_state=config.RANDOM_STATE)
    labels = km.fit_predict(X_scaled)

    sil_sample = min(8_000, len(X_scaled))
    sil = float(silhouette_score(X_scaled, labels, sample_size=sil_sample, random_state=config.RANDOM_STATE))
    dbi = float(davies_bouldin_score(X_scaled, labels))

    pca = PCA(n_components=2, random_state=config.RANDOM_STATE).fit(X_scaled)
    coords = pca.transform(X_scaled)

    # Store up to 3,000 points in JSON for fast interactive scatter rendering
    rng = np.random.RandomState(config.RANDOM_STATE)
    sample_idx = rng.choice(len(df), size=min(3_000, len(df)), replace=False)
    pca_points = [
        {
            "pc1": round(float(coords[i, 0]), 3),
            "pc2": round(float(coords[i, 1]), 3),
            "cluster": int(labels[i]),
            "converted": str(df.iloc[i].get("converted", "No")),
        }
        for i in sample_idx
    ]

    profiles = build_profiles(df, labels, features)
    figures = save_clustering_figures(k_table, pca_points, profiles)

    report = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "algorithm": "K-Means",
        "n_clusters": int(n_clusters),
        "n_customers": int(len(df)),
        "features": features,
        "inertia": round(float(km.inertia_), 2),
        "silhouette_score": round(sil, 4),
        "davies_bouldin_index": round(dbi, 4),
        "pca_explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "k_evaluation": k_table,
        "profiles": profiles,
        "pca_sample": pca_points,
        "figures": figures,
    }

    config.CLUSTERING_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "kmeans": km,
            "imputer": imputer,
            "scaler": scaler,
            "pca": pca,
            "features": features,
            "profiles": {p["cluster_id"]: p for p in profiles},
        },
        config.CLUSTERING_MODEL_PATH,
    )
    config.CLUSTERING_METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if verbose:
        print(f"  K-Means (K={n_clusters}) trained on {len(df):,} customers · Silhouette = {sil:.4f}")
        print("  Saved ➜ models/clustering.joblib, models/clustering_metrics.json")
    return report


def predict_segment(customer_features: dict) -> dict:
    """Assign a single customer behavior dictionary to its nearest K-Means segment."""
    if not config.CLUSTERING_MODEL_PATH.exists():
        run(verbose=False)
    bundle = joblib.load(config.CLUSTERING_MODEL_PATH)
    feats = bundle["features"]
    row = pd.DataFrame([{f: float(customer_features.get(f, 0.0)) for f in feats}])
    X_imp = bundle["imputer"].transform(row)
    X_log = np.log1p(np.clip(X_imp, 0, None))
    X_scaled = bundle["scaler"].transform(X_log)
    cid = int(bundle["kmeans"].predict(X_scaled)[0])
    xy = bundle["pca"].transform(X_scaled)[0]
    prof = bundle["profiles"].get(cid, {})
    return {
        "cluster_id": cid,
        "segment_name": prof.get("segment_name", f"Segment {cid}"),
        "description": prof.get("description", ""),
        "recommended_action": prof.get("recommended_action", ""),
        "segment_conversion_rate_pct": prof.get("conversion_rate_pct", 0.0),
        "pca_point": {"pc1": round(float(xy[0]), 3), "pc2": round(float(xy[1]), 3)},
    }


if __name__ == "__main__":
    run()
