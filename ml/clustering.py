"""
Unsupervised Learning Component – Customer & Session Segmentation using K-Means.

Supports:
  - Customer Segmentation (scope="customer" on data/processed/customer_profiles.csv)
  - Session Segmentation  (scope="session"  on data/processed/session_profiles.csv)
  - Configurable K (2..10), features, scaling ('standard', 'robust', 'minmax'),
    log1p skew handling, and random_state.
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
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from ml import config

SCALER_MAP = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "minmax": MinMaxScaler,
}


# =============================================================================
# 1. PREPROCESSING FOR SKEWED BEHAVIORAL FEATURES
# =============================================================================
def _transform_features(
    df: pd.DataFrame,
    features: list[str],
    scaler_name: str = "standard",
    use_log1p: bool = True,
    imputer=None,
    scaler=None,
):
    X_raw = df[features].copy()
    if imputer is None:
        imputer = SimpleImputer(strategy="median").fit(X_raw)
    X_imp = imputer.transform(X_raw)
    X_proc = np.log1p(np.clip(X_imp, 0, None)) if use_log1p else X_imp
    if scaler is None:
        scaler_cls = SCALER_MAP.get(scaler_name, StandardScaler)
        scaler = scaler_cls().fit(X_proc)
    X_scaled = scaler.transform(X_proc)
    return X_scaled, imputer, scaler


# =============================================================================
# 2. ELBOW METHOD (INERTIA) & SILHOUETTE ANALYSIS ACROSS K = 2..8
# =============================================================================
def evaluate_k_range(
    X_scaled: np.ndarray,
    k_values=config.K_RANGE,
    random_state: int = config.RANDOM_STATE,
) -> list[dict]:
    """Compute Inertia (Elbow), Silhouette Score, and Davies-Bouldin Index for each K."""
    sil_sample = min(8_000, len(X_scaled))
    rows = []
    for k in k_values:
        km = KMeans(n_clusters=int(k), n_init=10, random_state=int(random_state))
        labels = km.fit_predict(X_scaled)
        sil = float(silhouette_score(X_scaled, labels, sample_size=sil_sample, random_state=int(random_state)))
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
# =============================================================================
def _assign_segment_identity(
    z_row: pd.Series,
    conv_rate: float,
    overall_conv: float,
    scope: str = "customer",
) -> tuple[str, str, str]:
    """Generate data-driven segment names and business recommendations from z-scores."""
    top_feats = z_row.reindex(z_row.abs().sort_values(ascending=False).index)
    f1 = top_feats.index[0] if len(top_feats) > 0 else "activity"
    v1 = float(top_feats.iloc[0]) if len(top_feats) > 0 else 0.0
    f2 = top_feats.index[1] if len(top_feats) > 1 else ""
    v2 = float(top_feats.iloc[1]) if len(top_feats) > 1 else 0.0
    z_desc = f"{'High' if v1 >= 0 else 'Low'} {f1.replace('_', ' ')} ({v1:+.2f} SD)"
    if f2:
        z_desc += f" · {'High' if v2 >= 0 else 'Low'} {f2.replace('_', ' ')} ({v2:+.2f} SD)"

    if scope == "session":
        if z_row.get("add_to_cart_count", 0) > 0.5 or z_row.get("cart_interaction_ratio", 0) > 0.5:
            return (
                f"Cart-Focused Sessions ({z_desc})",
                "Sessions with high add-to-cart intensity and strong purchase readiness.",
                "Streamline checkout flow, show trust badges, and avoid distracting pop-ups.",
            )
        if z_row.get("total_views", 0) > 0.6 or z_row.get("unique_items_viewed", 0) > 0.6:
            return (
                f"High Product Exploration Sessions ({z_desc})",
                "Extended multi-item browsing sessions comparing many products.",
                "Provide comparison widgets, personalized filters, and 'save for later' prompts.",
            )
        if z_row.get("item_revisit_ratio", 0) > 0.5:
            return (
                f"Focused Item-Revisit Sessions ({z_desc})",
                "Sessions repeatedly viewing the same product before deciding.",
                "Display stock urgency, reviews, and Q&A on the revisited product page.",
            )
        return (
            f"Short Browsing Sessions ({z_desc})",
            "Brief browsing sessions with rapid transitions and lower exploration depth.",
            "Highlight clear category navigation and top-selling hero products immediately.",
        )

    # Customer-level naming
    if z_row.get("total_add_to_cart", 0) > 0.5 and z_row.get("cart_abandonment_rate", 0) > 0.4:
        return (
            f"Hesitant Cart Abandoners ({z_desc})",
            "Customers who frequently add products to cart but leave without completing checkout.",
            "Trigger exit-intent reminders, free-shipping thresholds, or time-limited cart incentives.",
        )
    if z_row.get("total_sessions", 0) > 0.5 or z_row.get("total_views", 0) > 0.6:
        if conv_rate >= overall_conv * 1.15:
            return (
                f"High-Value Loyal Shoppers ({z_desc})",
                "Returning multi-session customers with deep exploration and strong conversion.",
                "Enroll in VIP loyalty rewards, early-access drops, and cross-sell bundles.",
            )
        return (
            f"Deep Window Explorers ({z_desc})",
            "High browsing volume across many sessions and items, but moderate checkout rate.",
            "Show social proof, product comparison tables, and price-drop alerts on viewed items.",
        )
    if z_row.get("recency_days", 0) > 0.5:
        return (
            f"Dormant / At-Risk Visitors ({z_desc})",
            "Customers whose last activity occurred long ago with low recent engagement.",
            "Run win-back re-engagement campaigns featuring trending items from their history.",
        )
    if z_row.get("avg_session_duration_sec", 0) < -0.25:
        return (
            f"Quick Bouncers / Casual Browsers ({z_desc})",
            "Short, low-interaction visits with minimal product depth and low cart activity.",
            "Optimize landing-page speed, highlight top bestsellers, and simplify search filters.",
        )
    return (
        f"Focused Everyday Shoppers ({z_desc})",
        "Balanced session duration and steady product exploration near the population mean.",
        "Offer relevant product recommendations and streamlined one-click checkout prompts.",
    )


def build_profiles(df: pd.DataFrame, labels: np.ndarray, features: list[str], scope: str = "customer") -> list[dict]:
    """Build detailed per-cluster profiles from actual data."""
    work = df.copy()
    work["cluster"] = labels
    overall_means = work[features].mean()
    overall_stds = work[features].std().replace(0, 1.0)
    overall_conv = float((work["converted"] == "Yes").mean() * 100) if "converted" in work.columns else 0.0
    tx_col = "total_transactions" if "total_transactions" in work.columns else "transaction_count"

    profiles = []
    used_names: set[str] = set()
    for cid, grp in work.groupby("cluster", sort=True):
        means = grp[features].mean()
        z_scores = ((means - overall_means) / overall_stds).round(3)
        conv_pct = float((grp["converted"] == "Yes").mean() * 100) if "converted" in grp.columns else 0.0
        name, desc, action = _assign_segment_identity(z_scores, conv_pct, overall_conv, scope=scope)
        if name in used_names:
            name = f"{name} [#{int(cid)}]"
        used_names.add(name)

        profiles.append({
            "cluster_id": int(cid),
            "segment_name": name,
            "description": desc,
            "recommended_action": action,
            "size": int(len(grp)),
            "share_pct": round(float(len(grp) / len(work) * 100), 2),
            "conversion_rate_pct": round(conv_pct, 2),
            "avg_transactions": round(float(grp[tx_col].mean()), 3) if tx_col in grp.columns else 0.0,
            "feature_means": {f: round(float(means[f]), 2) for f in features},
            "feature_z_scores": {f: round(float(z_scores[f]), 3) for f in features},
        })
    return profiles


# =============================================================================
# 4. SAVE CLUSTERING FIGURES
# =============================================================================
def save_clustering_figures(k_table: list[dict], pca_points: list[dict], profiles: list[dict]) -> list[str]:
    out = config.FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)
    saved = []

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

    pdf = pd.DataFrame(pca_points)
    name_map = {p["cluster_id"]: p["segment_name"].split(" (")[0] for p in profiles}
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
# 5. MAIN TRAINING & PREDICTION FUNCTIONS (CUSTOMER & SESSION SEGMENTATION)
# =============================================================================
def run(
    n_clusters: int = config.DEFAULT_K_CLUSTERS,
    scope: str = "customer",
    features: list[str] | None = None,
    scaler_name: str = "standard",
    use_log1p: bool = True,
    random_state: int = config.RANDOM_STATE,
    verbose: bool = True,
) -> dict:
    """Fit K-Means segmentation (for 'customer' or 'session' scope) and save artifacts."""
    data_path = config.SESSION_PROFILES_PATH if scope == "session" else config.CUSTOMER_PROFILES_PATH
    if not data_path.exists():
        from ml.data import make_dataset
        make_dataset.main()

    df = pd.read_csv(data_path)
    default_feats = config.SESSION_CLUSTERING_FEATURES if scope == "session" else config.CUSTOMER_CLUSTERING_FEATURES
    chosen_feats = [f for f in (features or default_feats) if f in df.columns and f not in ("visitorid", "session_id")]
    if len(chosen_feats) < 2:
        chosen_feats = [f for f in default_feats if f in df.columns]

    X_scaled, imputer, scaler = _transform_features(
        df, chosen_feats, scaler_name=scaler_name, use_log1p=use_log1p
    )

    k_table = evaluate_k_range(X_scaled, config.K_RANGE, random_state=int(random_state))
    km = KMeans(n_clusters=int(n_clusters), n_init=10, random_state=int(random_state))
    labels = km.fit_predict(X_scaled)

    sil_sample = min(8_000, len(X_scaled))
    sil = float(silhouette_score(X_scaled, labels, sample_size=sil_sample, random_state=int(random_state)))
    dbi = float(davies_bouldin_score(X_scaled, labels))

    pca = PCA(n_components=2, random_state=int(random_state)).fit(X_scaled)
    coords = pca.transform(X_scaled)

    rng = np.random.RandomState(int(random_state))
    sample_idx = rng.choice(len(df), size=min(3_000, len(df)), replace=False)
    id_col = "session_id" if scope == "session" else "visitorid"
    pca_points = [
        {
            "id": int(df.iloc[i].get(id_col, i)),
            "pc1": round(float(coords[i, 0]), 3),
            "pc2": round(float(coords[i, 1]), 3),
            "cluster": int(labels[i]),
            "converted": str(df.iloc[i].get("converted", "No")),
        }
        for i in sample_idx
    ]

    profiles = build_profiles(df, labels, chosen_feats, scope=scope)
    figures = save_clustering_figures(k_table, pca_points, profiles) if scope == "customer" else []

    report = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "scope": scope,
        "algorithm": "K-Means",
        "n_clusters": int(n_clusters),
        "n_customers": int(len(df)),
        "scaler": scaler_name,
        "use_log1p": bool(use_log1p),
        "random_state": int(random_state),
        "features": chosen_feats,
        "available_features": [f for f in default_feats if f in df.columns],
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
    if scope == "customer":
        joblib.dump(
            {
                "kmeans": km,
                "imputer": imputer,
                "scaler": scaler,
                "pca": pca,
                "use_log1p": bool(use_log1p),
                "features": chosen_feats,
                "profiles": {p["cluster_id"]: p for p in profiles},
            },
            config.CLUSTERING_MODEL_PATH,
        )
        config.CLUSTERING_METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        config.SESSION_CLUSTERING_METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if verbose:
        print(f"  K-Means ({scope}, K={n_clusters}) trained on {len(df):,} records · Silhouette = {sil:.4f}")
    return report


def predict_segment(customer_features: dict) -> dict:
    """Assign a single customer behavior dictionary to its nearest K-Means segment."""
    if not config.CLUSTERING_MODEL_PATH.exists():
        run(verbose=False)
    try:
        bundle = joblib.load(config.CLUSTERING_MODEL_PATH)
    except Exception:
        run(verbose=False)
        bundle = joblib.load(config.CLUSTERING_MODEL_PATH)
    feats = bundle["features"]
    row = pd.DataFrame([{f: float(customer_features.get(f, 0.0)) for f in feats}])
    X_imp = bundle["imputer"].transform(row)
    use_log = bundle.get("use_log1p", True)
    X_proc = np.log1p(np.clip(X_imp, 0, None)) if use_log else X_imp
    X_scaled = bundle["scaler"].transform(X_proc)
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
    run(scope="customer")
    run(scope="session")
