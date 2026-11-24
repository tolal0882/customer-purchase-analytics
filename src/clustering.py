"""
Unsupervised Machine Learning Module — Task 2: Session Behavior Segmentation (Section 7):
  - Algorithm: K-Means Clustering
  - Default 9 behavioral features + optional PageValues, Month, VisitorType, Weekend
  - CRITICAL RULE: Never includes Revenue in K-Means input features.
  - Evaluates K range (2..10) with Inertia (Elbow Method) and Silhouette Score
  - Fits K-Means with configurable K, random_state, max_iter, scaler, and log1p option
  - Computes 2-D PCA coordinates for visualization
  - Generates data-driven session behavior segment descriptions from actual cluster feature z-scores
  - Post-clustering Revenue analysis (for interpretation only, never used as input)
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score

from src import data_validation, model_manager, preprocessing


# =============================================================================
# 1. DATA-DRIVEN SESSION BEHAVIOR SEGMENT PROFILER (SECTION 7)
#    Names and describes each cluster strictly from its actual centroid feature
#    means and standardized z-scores—never using hardcoded customer loyalty
#    labels (because rows represent single shopping sessions).
# =============================================================================
def _generate_data_driven_segment_profile(
    cluster_id: int,
    z_scores: dict[str, float],
    means: dict[str, float],
) -> tuple[str, str]:
    """Generate a descriptive session behavior segment label and narrative strictly from actual
    cluster z-scores and feature means (never hardcoded customer loyalty labels)."""
    prod_z = z_scores.get("ProductRelated", 0.0)
    prod_dur_z = z_scores.get("ProductRelated_Duration", 0.0)
    bounce_z = z_scores.get("BounceRates", 0.0)
    exit_z = z_scores.get("ExitRates", 0.0)
    admin_z = z_scores.get("Administrative", 0.0)
    info_z = z_scores.get("Informational", 0.0)
    spec_z = z_scores.get("SpecialDay", 0.0)
    pv_z = z_scores.get("PageValues", 0.0)

    # Pick dominant behavioral trait from actual z-scores
    if bounce_z > 0.9 or exit_z > 0.9:
        label = f"High-Bounce / Quick-Exit Sessions (Segment {cluster_id})"
    elif pv_z > 0.9:
        label = f"High Page-Value Transaction-Proximity Sessions (Segment {cluster_id})"
    elif spec_z > 1.0:
        label = f"Special-Day / Holiday Shopping Sessions (Segment {cluster_id})"
    elif max(prod_z, prod_dur_z) > 0.75 and max(admin_z, info_z) > 0.65:
        label = f"Deep Multi-Section Browsing Sessions (Segment {cluster_id})"
    elif max(prod_z, prod_dur_z) > 0.65:
        label = f"Extended Product-Focused Sessions (Segment {cluster_id})"
    elif max(admin_z, info_z) > 0.65:
        label = f"Administrative & Informational Browsing Sessions (Segment {cluster_id})"
    elif bounce_z < -0.25 and prod_z >= -0.2:
        label = f"Low-Bounce Focused Product Browsing Sessions (Segment {cluster_id})"
    else:
        label = f"Light / Brief Browsing Sessions (Segment {cluster_id})"

    # Build factual explanation citing actual cluster averages
    parts = []
    if "ProductRelated" in means:
        parts.append(f"averages **{means['ProductRelated']:.1f}** product pages")
    if "ProductRelated_Duration" in means:
        parts.append(f"**{means['ProductRelated_Duration']:.0f}s** product browsing duration")
    if "Administrative" in means:
        parts.append(f"**{means['Administrative']:.1f}** administrative pages")
    if "BounceRates" in means and "ExitRates" in means:
        parts.append(f"bounce rate **{means['BounceRates']*100:.2f}%** and exit rate **{means['ExitRates']*100:.2f}%**")

    sorted_z = sorted(z_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    z_summary = ", ".join(f"`{k}` ({v:+.2f}σ)" for k, v in sorted_z)
    description = (
        f"This session behavior segment {', '.join(parts)}. "
        f"Most distinctive standardized deviations vs. dataset mean: {z_summary}."
    )
    return label, description


# =============================================================================
# 2. COMPLETE K-MEANS SESSION SEGMENTATION WORKFLOW (SECTION 7)
#    Steps:
#      1. Validate feature selection and strictly block 'Revenue'
#      2. Preprocess & scale selected features
#      3. Evaluate multiple K values (Inertia for Elbow + Silhouette Score)
#      4. Train K-Means with chosen K & assign cluster labels
#      5. Project scaled features onto 2-D PCA for visualization
#      6. Build data-driven cluster profiles & compare post-hoc Revenue rates
# =============================================================================
def run_session_segmentation(
    df: pd.DataFrame,
    selected_features: list[str] | None = None,
    n_clusters: int = 4,
    k_min: int = 2,
    k_max: int = 8,
    random_state: int = 42,
    max_iter: int = 300,
    scaler_name: str = "standard",
    use_log1p: bool = True,
    drop_duplicates: bool = False,
) -> dict:
    """Execute the full K-Means session behavior segmentation workflow (Section 7)."""
    # 2.1 Validate features & enforce strict Revenue exclusion rule
    features = list(selected_features or data_validation.DEFAULT_CLUSTERING_FEATURES)
    if data_validation.TARGET_COL in features:
        raise ValueError("CRITICAL RULE: 'Revenue' must never be included in clustering features.")
    if len(features) < 2:
        raise ValueError("Please select at least 2 features for K-Means clustering.")

    # 2.2 Clean dataset & transform clustering features (scaling + optional encoding)
    clean_df, clean_info = preprocessing.apply_cleaning_options(df, drop_duplicates=drop_duplicates)
    X_in = preprocessing.prepare_features_for_transform(clean_df, features)

    preprocessor = preprocessing.build_clustering_preprocessor(
        features, scaler_name=scaler_name, use_log1p=use_log1p
    )
    X_trans = preprocessor.fit_transform(X_in)

    # 2.3 Evaluate K range (K = k_min..k_max) for Elbow Method (Inertia) & Silhouette Score
    k_eval_rows = []
    sil_sample_size = min(5000, len(X_trans))
    k_values = list(range(max(2, int(k_min)), max(int(k_max) + 1, int(n_clusters) + 1)))
    for k in k_values:
        km_k = KMeans(
            n_clusters=k,
            n_init=10,
            max_iter=int(max_iter),
            random_state=int(random_state),
        )
        lbl_k = km_k.fit_predict(X_trans)
        sil_k = float(
            silhouette_score(X_trans, lbl_k, sample_size=sil_sample_size, random_state=int(random_state))
        )
        db_k = float(davies_bouldin_score(X_trans, lbl_k))
        k_eval_rows.append({
            "k": int(k),
            "inertia": round(float(km_k.inertia_), 2),
            "silhouette": round(sil_k, 4),
            "davies_bouldin": round(db_k, 4),
        })

    # 2.4 Fit final K-Means model with user-selected K and assign cluster labels
    km = KMeans(
        n_clusters=int(n_clusters),
        n_init=10,
        max_iter=int(max_iter),
        random_state=int(random_state),
    )
    labels = km.fit_predict(X_trans)
    chosen_sil = float(
        silhouette_score(X_trans, labels, sample_size=sil_sample_size, random_state=int(random_state))
    )
    chosen_db = float(davies_bouldin_score(X_trans, labels))

    # 2.5 Compute 2-D Principal Component Analysis (PCA) coordinates
    pca = PCA(n_components=2, random_state=int(random_state))
    coords = pca.fit_transform(X_trans)
    explained_var = [round(float(v * 100), 2) for v in pca.explained_variance_ratio_]

    # 2.6 Generate data-driven cluster profiles & post-clustering Revenue comparison
    num_features, _ = preprocessing.split_feature_types(features)
    profile_num_cols = list(dict.fromkeys(num_features + data_validation.DEFAULT_CLUSTERING_FEATURES))
    profile_num_cols = [c for c in profile_num_cols if c in clean_df.columns]

    global_means = clean_df[profile_num_cols].mean()
    global_stds = clean_df[profile_num_cols].std().replace(0, 1.0)

    # Revenue is used ONLY after clustering to inspect segment conversion rates
    revenue_bool = clean_df[data_validation.TARGET_COL].astype(bool)

    profiles = []
    segment_name_map = {}
    for cid in range(int(n_clusters)):
        mask = labels == cid
        sub = clean_df.loc[mask]
        size = int(mask.sum())
        share_pct = round(size / max(len(clean_df), 1) * 100, 2)

        c_means = {c: round(float(sub[c].mean()), 4) for c in profile_num_cols}
        c_z = {
            c: round(float((sub[c].mean() - global_means[c]) / global_stds[c]), 3)
            for c in profile_num_cols
        }

        seg_name, seg_desc = _generate_data_driven_segment_profile(cid, c_z, c_means)
        segment_name_map[cid] = seg_name

        # Post-hoc Revenue rate (interpretation only)
        rev_sub = revenue_bool.loc[mask]
        purchase_sessions = int(rev_sub.sum())
        purchase_rate_pct = round(float(rev_sub.mean() * 100), 2) if size > 0 else 0.0

        cat_modes = {}
        for cat_c in ("VisitorType", "Month", "Weekend"):
            if cat_c in sub.columns and len(sub) > 0:
                cat_modes[cat_c] = str(sub[cat_c].mode().iloc[0])

        profiles.append({
            "cluster_id": cid,
            "segment_name": seg_name,
            "size": size,
            "share_pct": share_pct,
            "purchase_sessions_posthoc": purchase_sessions,
            "purchase_rate_posthoc_pct": purchase_rate_pct,
            "dominant_visitor_type": cat_modes.get("VisitorType", "—"),
            "dominant_month": cat_modes.get("Month", "—"),
            "feature_means": c_means,
            "feature_z_scores": c_z,
            "interpretation": seg_desc,
        })

    # 2.7 Build full session assignments table for CSV export & sample for PCA scatter chart
    assignments_df = clean_df.copy()
    assignments_df["Cluster_ID"] = labels
    assignments_df["Session_Segment"] = [segment_name_map[int(c)] for c in labels]
    assignments_df["PCA_1"] = np.round(coords[:, 0], 4)
    assignments_df["PCA_2"] = np.round(coords[:, 1], 4)

    rng = np.random.RandomState(int(random_state))
    sample_idx = rng.choice(len(assignments_df), size=min(3500, len(assignments_df)), replace=False)
    pca_df = assignments_df.iloc[sample_idx][
        ["Cluster_ID", "Session_Segment", "PCA_1", "PCA_2", "ProductRelated", "BounceRates", "ExitRates", "PageValues", "VisitorType", "Revenue"]
    ].copy()

    result = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "n_clusters": int(n_clusters),
        "n_sessions": len(clean_df),
        "selected_features": features,
        "revenue_excluded_from_inputs": True,
        "scaler_name": scaler_name,
        "use_log1p": bool(use_log1p),
        "random_state": int(random_state),
        "max_iter": int(max_iter),
        "drop_duplicates": bool(drop_duplicates),
        "inertia": round(float(km.inertia_), 2),
        "silhouette_score": round(chosen_sil, 4),
        "davies_bouldin_index": round(chosen_db, 4),
        "pca_explained_variance_pct": explained_var,
        "k_evaluation": k_eval_rows,
        "profiles": profiles,
        "pca_sample": pca_df.to_dict(orient="records"),
        "assignments_df": assignments_df,
        "preprocessor": preprocessor,
        "kmeans": km,
        "pca": pca,
    }
    model_manager.save_clustering_bundle(result)
    return result
