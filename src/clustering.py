"""K-Means segmentation (customers and sessions): scaling, elbow/silhouette, PCA, data-driven profiles."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, RobustScaler, StandardScaler

from config import settings

# =============================================================================
# 1. CLUSTERING FEATURE DEFAULTS & PREPROCESSING PIPELINE
#    - Excludes visitorid / session_id from inputs.
#    - Automatically flags skewed non-negative features (|skew| > 1) for log1p
#      transformation before scaling (Standard / Robust / MinMax).
# =============================================================================
SCALERS = {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler}

CUSTOMER_DEFAULT_FEATURES = [
    "total_sessions", "total_views", "unique_items_viewed", "total_add_to_cart", "average_session_duration",
    "average_products_per_session", "historical_cart_abandonment_rate", "days_since_last_activity",
]
CUSTOMER_OPTIONAL_FEATURES = ["total_transactions", "purchase_frequency", "average_cart_additions_per_session",
                              "average_time_between_sessions_hours"]
SESSION_DEFAULT_FEATURES = [
    "total_events", "total_views", "unique_items_viewed", "add_to_cart_count", "session_duration_sec",
    "avg_time_between_events_sec", "cart_interaction_ratio", "item_revisit_ratio",
]
# Features that encode the purchase outcome: allowed only if the user insists (customer brief lists
# total_transactions/purchase_frequency as optional inputs) and then purchase-rate comparison is circular.
OUTCOME_FEATURES = {"total_transactions", "purchase_frequency", "transaction_count", "has_purchase"}


def _log1p(x):
    return np.log1p(np.clip(x, 0, None))


def make_transformer(features: list[str], log_features: list[str], scaler: str = "standard") -> Pipeline:
    """Median imputation -> optional log1p on skewed columns -> scaling. Fitted only on the fitting sample."""
    from sklearn.compose import ColumnTransformer
    log_cols = [f for f in features if f in log_features]
    lin_cols = [f for f in features if f not in log_features]
    parts = []
    if log_cols:
        parts.append(("log", Pipeline([("imp", SimpleImputer(strategy="median")),
                                       ("log", FunctionTransformer(_log1p, feature_names_out="one-to-one")),
                                       ("scale", SCALERS[scaler]())]), log_cols))
    if lin_cols:
        parts.append(("lin", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", SCALERS[scaler]())]), lin_cols))
    return ColumnTransformer(parts, remainder="drop")


def skewed_features(df: pd.DataFrame, features: list[str], threshold: float = 1.0) -> list[str]:
    """Suggest log1p for non-negative columns with |skew| > threshold (measured on the data)."""
    out = []
    for f in features:
        s = df[f].dropna()
        if len(s) > 2 and s.min() >= 0 and abs(s.skew()) > threshold:
            out.append(f)
    return out


@dataclass
class ClusterResult:
    labels: pd.Series                 # label for every row of the input frame
    transformer: Pipeline
    model: KMeans
    k: int
    inertia: float
    silhouette: float
    davies_bouldin: float
    pca_coords: pd.DataFrame          # sampled points: PC1, PC2, cluster
    pca_explained: list
    profile: pd.DataFrame             # per-cluster means + size + share + z-names
    z_profile: pd.DataFrame
    names: dict
    descriptions: dict
    features: list
    log_features: list
    scaler: str
    seed: int
    fit_rows: int


def _fit_sample(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    return df if len(df) <= max_rows else df.sample(n=max_rows, random_state=seed)


# =============================================================================
# 2. ELBOW METHOD, SILHOUETTE ANALYSIS & K-MEANS FITTING
#    - k_range_analysis() sweeps K=k_min..k_max to plot Inertia (Elbow) and
#      Silhouette scores on a seeded sample for fast responsiveness.
#    - fit_kmeans() fits on up to max_fit_rows, assigns ALL rows to clusters,
#      projects a sample to 2-D PCA, and builds data-driven cluster profiles.
# =============================================================================
def k_range_analysis(df: pd.DataFrame, features: list[str], log_features: list[str], scaler: str,
                     k_min: int, k_max: int, seed: int, max_rows: int = 60_000,
                     progress=None) -> pd.DataFrame:
    """Inertia (elbow) and silhouette for each K on a fixed random sample."""
    samp = _fit_sample(df, max_rows, seed)
    X = make_transformer(features, log_features, scaler).fit_transform(samp[features])
    sil_n = min(settings.SILHOUETTE_SAMPLE_SIZE, len(X))
    rng = np.random.RandomState(seed)
    rows = []
    ks = list(range(k_min, k_max + 1))
    for i, k in enumerate(ks, 1):
        km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(X)
        sil = silhouette_score(X, km.labels_, sample_size=sil_n, random_state=seed) if k > 1 else np.nan
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil,
                     "davies_bouldin": davies_bouldin_score(X, km.labels_) if k > 1 else np.nan})
        if progress:
            progress(i / len(ks), f"K={k} evaluated")
    return pd.DataFrame(rows)


def fit_kmeans(df: pd.DataFrame, features: list[str], k: int, seed: int = 42, log_features: list[str] | None = None,
               scaler: str = "standard", max_fit_rows: int = 200_000, outcome_col: str | None = "has_purchase"
               ) -> ClusterResult:
    """Fit K-Means on a (capped, seeded) sample; assign ALL rows; build profiles from actual data."""
    if not features:
        raise ValueError("Select at least one feature")
    if "visitorid" in features:
        raise ValueError("visitorid must not be a clustering input")
    log_features = log_features or []
    fit_df = _fit_sample(df, max_fit_rows, seed)
    tr = make_transformer(features, log_features, scaler).fit(fit_df[features])
    Xfit = tr.transform(fit_df[features])
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xfit)
    Xall = tr.transform(df[features])
    labels = pd.Series(km.predict(Xall), index=df.index, name="cluster")
    sil_n = min(settings.SILHOUETTE_SAMPLE_SIZE, len(Xfit))
    sil = silhouette_score(Xfit, km.labels_, sample_size=sil_n, random_state=seed) if k > 1 else float("nan")
    dbi = davies_bouldin_score(Xfit, km.labels_) if k > 1 else float("nan")

    # PCA on a plotting sample (assigned labels)
    n_plot = min(settings.PLOT_SAMPLE_SIZE, len(df))
    pos = np.random.RandomState(seed).choice(len(df), n_plot, replace=False)
    pca = PCA(n_components=2, random_state=seed).fit(Xall[pos])
    xy = pca.transform(Xall[pos])
    pca_coords = pd.DataFrame({"PC1": xy[:, 0], "PC2": xy[:, 1], "cluster": labels.to_numpy()[pos]})

    profile, z, names, desc = profile_clusters(df, labels, features, outcome_col)
    return ClusterResult(labels, tr, km, k, float(km.inertia_), float(sil), float(dbi), pca_coords,
                         pca.explained_variance_ratio_.tolist(), profile, z, names, desc, features, log_features,
                         scaler, seed, len(fit_df))


# =============================================================================
# 3. DATA-DRIVEN CLUSTER PROFILING & Z-SCORE NAMING
#    - Computes per-cluster means, sizes, percentage shares, and post-clustering
#      purchase rates (has_purchase is compared AFTER clustering, not as input).
#    - Generates honest descriptive names from each cluster's largest z-score
#      deviations vs the overall population (never hardcoded personas).
# =============================================================================
def profile_clusters(df: pd.DataFrame, labels: pd.Series, features: list[str], outcome_col: str | None):
    """Cluster means/medians, size and share, outcome rate (computed AFTER clustering), plus names
    derived from z-scores of cluster means vs the population (never hardcoded)."""
    d = df.assign(cluster=labels)
    g = d.groupby("cluster")
    prof = g[features].mean()
    prof.insert(0, "size", g.size())
    prof.insert(1, "share_pct", prof["size"] / prof["size"].sum() * 100)
    if outcome_col and outcome_col in d.columns:
        prof["purchase_rate_pct"] = g[outcome_col].mean() * 100
    mu, sd = df[features].mean(), df[features].std().replace(0, np.nan)
    z = ((g[features].mean() - mu) / sd).fillna(0.0)
    names, desc = {}, {}
    used = set()
    for c in prof.index:
        zs = z.loc[c].sort_values(key=lambda s: -s.abs())
        strong = [(f, v) for f, v in zs.items() if abs(v) >= 0.5][:2]
        if strong:
            name = " · ".join(f"{'High' if v > 0 else 'Low'} {f.replace('_', ' ')}" for f, v in strong)
            text = "; ".join(f"{f} is {abs(v):.1f} SD {'above' if v > 0 else 'below'} the overall mean" for f, v in strong)
        else:
            name, text = "Near-average behaviour", "no feature deviates by more than 0.5 SD from the overall mean"
        if name in used:
            name = f"{name} (#{c})"
        used.add(name)
        names[c], desc[c] = name, text
    return prof, z, names, desc


def assign_new(result_bundle: dict, df: pd.DataFrame) -> np.ndarray:
    """Assign new rows with a SAVED transformer + KMeans."""
    X = result_bundle["transformer"].transform(df[result_bundle["features"]])
    return result_bundle["model"].predict(X)


def result_to_bundle(r: ClusterResult) -> dict:
    return {"transformer": r.transformer, "model": r.model, "features": r.features, "log_features": r.log_features,
            "scaler": r.scaler, "seed": r.seed, "names": r.names}
