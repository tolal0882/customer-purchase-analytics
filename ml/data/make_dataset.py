"""
Step 1 – Data cleaning & session/customer aggregation:
    data/raw/events.csv  ➜  data/processed/clean.csv
                         ➜  data/processed/customer_profiles.csv
                         ➜  data/processed/insights_summary.json

Run from the project root:
    python -m ml.data.make_dataset
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from ml import config

# Plausible epoch-millisecond bounds (2000-01-01 to 2100-01-01)
TS_MIN_MS = 946_684_800_000
TS_MAX_MS = 4_102_444_800_000


# =============================================================================
# 1. LOAD RAW RETAILROCKET CLICKSTREAM DATA
# =============================================================================
def load_raw(path=config.RAW_DATA_PATH, nrows: int | None = None) -> pd.DataFrame:
    """Read raw events.csv without mutating the source file."""
    return pd.read_csv(path, nrows=nrows)


# =============================================================================
# 2. CLEAN RAW CLICKSTREAM EVENTS & ENGINEER 30-MINUTE SESSIONS
#    - Removes exact duplicate rows (460 in Retailrocket events.csv).
#    - Normalizes event names ('view', 'addtocart', 'transaction') & drops bad timestamps.
#    - Sorts chronologically by (visitorid, timestamp) and assigns session_id
#      whenever a visitor is inactive for more than 30 minutes.
# =============================================================================
def clean_events(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    """Clean raw clickstream rows and assign gap-based session_id and event_pos."""
    log: list[str] = []
    df = raw_df.copy()
    n_raw = len(df)

    # 1. Strip whitespace and normalize event strings
    df["event"] = df["event"].astype("string").str.strip().str.lower()

    # 2. Coerce numeric identifiers and timestamps
    for col in ("timestamp", "visitorid", "itemid"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Filter invalid timestamps, missing IDs, or unknown event types
    valid_mask = (
        df["timestamp"].between(TS_MIN_MS, TS_MAX_MS)
        & df["visitorid"].notna()
        & df["itemid"].notna()
        & df["event"].isin(config.VALID_EVENTS)
    )
    n_invalid = int((~valid_mask).sum())
    if n_invalid:
        df = df.loc[valid_mask]
    log.append(f"Removed {n_invalid} invalid timestamp/ID/event rows")

    # 4. Remove exact duplicate rows
    before_dedup = len(df)
    df = df.drop_duplicates()
    n_dups = before_dedup - len(df)
    log.append(f"Removed {n_dups} exact duplicate rows")

    # 5. Memory-efficient dtypes and chronological sort per visitor
    df["timestamp"] = df["timestamp"].astype("int64")
    df["visitorid"] = df["visitorid"].astype("int32")
    df["itemid"] = df["itemid"].astype("int32")
    df = df.sort_values(["visitorid", "timestamp"], kind="stable").reset_index(drop=True)

    # 6. Vectorized 30-minute inactivity session construction
    v = df["visitorid"].to_numpy()
    t = df["timestamp"].to_numpy()
    timeout_ms = config.SESSION_TIMEOUT_MINUTES * 60_000
    new_sess = np.empty(len(df), dtype=bool)
    new_sess[0] = True
    new_sess[1:] = (v[1:] != v[:-1]) | ((t[1:] - t[:-1]) > timeout_ms)

    sid = (np.cumsum(new_sess) - 1).astype("int32")
    start_idx = np.flatnonzero(new_sess)
    pos = (np.arange(len(df)) - start_idx[sid]).astype("int32")
    df["session_id"] = sid
    df["event_pos"] = pos
    log.append(f"Engineered {int(sid[-1]) + 1:,} sessions ({config.SESSION_TIMEOUT_MINUTES:g}-min timeout)")

    stats = {
        "raw_rows": int(n_raw),
        "clean_event_rows": int(len(df)),
        "duplicates_removed": int(n_dups),
        "invalid_rows_removed": int(n_invalid),
    }
    return df, log, stats


# =============================================================================
# 3. BUILD LEAKAGE-SAFE EARLY-SESSION PURCHASE PREDICTION DATASET (clean.csv)
#    - Observes the FIRST 3 non-transaction events of each eligible session.
#    - Target `will_purchase` = "Yes" iff a transaction occurs STRICTLY AFTER
#      the 3rd event in that same session ("No" otherwise).
#    - Sessions where a transaction already happened at/before event 3 are
#      excluded so post-purchase data never leaks into model inputs.
# =============================================================================
def build_modeling_snapshots(
    events: pd.DataFrame,
    max_rows: int | None = config.MAX_MODELING_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (snapshot_modeling_df, full_sessions_df)."""
    sid = events["session_id"].to_numpy()
    ts = events["timestamp"].to_numpy()
    pos = events["event_pos"].to_numpy()
    ev_type = events["event"].to_numpy()
    is_view = ev_type == "view"
    is_cart = ev_type == "addtocart"
    is_tx = ev_type == "transaction"
    n_sessions = int(sid[-1]) + 1

    # Full-session aggregates (used for historical prior-session features & customer profiles)
    full_base = pd.DataFrame({
        "session_id": sid,
        "visitorid": events["visitorid"].to_numpy(),
        "ts": ts,
        "is_view": is_view.astype(np.int32),
        "is_cart": is_cart.astype(np.int32),
        "is_tx": is_tx.astype(np.int32),
    })
    sessions = full_base.groupby("session_id", sort=True).agg(
        visitorid=("visitorid", "first"),
        total_events=("ts", "size"),
        start_ts=("ts", "min"),
        end_ts=("ts", "max"),
        total_views=("is_view", "sum"),
        add_to_cart_count=("is_cart", "sum"),
        transaction_count=("is_tx", "sum"),
    )
    uniq_items_all = (
        pd.DataFrame({"s": sid, "i": events["itemid"].to_numpy()})
        .drop_duplicates()
        .groupby("s")
        .size()
    )
    sessions["unique_items_viewed"] = uniq_items_all.reindex(sessions.index, fill_value=1).astype("int32")
    sessions["session_duration_sec"] = (sessions["end_ts"] - sessions["start_ts"]) / 1000.0

    # Prior-session historical features per visitor (strictly earlier sessions only via shift)
    s_ord = sessions[["visitorid", "start_ts", "transaction_count"]].sort_values(["visitorid", "start_ts"], kind="stable")
    g_vis = s_ord.groupby("visitorid", sort=False)
    s_ord["prior_sessions"] = g_vis.cumcount().astype("int32")
    s_ord["prior_purchases"] = (g_vis["transaction_count"].cumsum() - s_ord["transaction_count"]).astype("int32")
    sessions["prior_sessions"] = s_ord["prior_sessions"].reindex(sessions.index)
    sessions["prior_purchases"] = s_ord["prior_purchases"].reindex(sessions.index)

    # Identify the 3rd non-transaction event in each session (the snapshot cutoff)
    n_ev = config.SNAPSHOT_EVENTS
    cum_nontx = pd.Series((~is_tx).astype(np.int32)).groupby(sid).cumsum().to_numpy()
    hit = (~is_tx) & (cum_nontx == n_ev)
    snap_pos = np.full(n_sessions, -1, dtype="int32")
    snap_pos[sid[hit]] = pos[hit]

    in_prefix = (snap_pos[sid] >= 0) & (pos <= snap_pos[sid])
    after_snap = (snap_pos[sid] >= 0) & (pos > snap_pos[sid])

    prefix_tx = np.bincount(sid[in_prefix & is_tx], minlength=n_sessions)
    after_cnt = np.bincount(sid[after_snap], minlength=n_sessions)
    after_tx = np.bincount(sid[after_snap & is_tx], minlength=n_sessions)

    # Eligible sessions: reached n_ev non-tx events, 0 transactions in prefix, >=1 event after snapshot
    eligible = (snap_pos >= 0) & (prefix_tx == 0) & (after_cnt > 0)
    target_yes = after_tx > 0

    pref_ev = events.loc[in_prefix & eligible[sid]]
    p_sid = pref_ev["session_id"].to_numpy()
    p_ts = pref_ev["timestamp"].to_numpy()
    p_view = (pref_ev["event"].to_numpy() == "view").astype(np.int32)
    p_cart = (pref_ev["event"].to_numpy() == "addtocart").astype(np.int32)

    p_agg = pd.DataFrame({
        "session_id": p_sid,
        "ts": p_ts,
        "is_view": p_view,
        "is_cart": p_cart,
    }).groupby("session_id", sort=True).agg(
        session_start_ts=("ts", "min"),
        snap_ts=("ts", "max"),
        views_count=("is_view", "sum"),
        add_to_cart_count=("is_cart", "sum"),
        last_is_cart=("is_cart", "last"),
    )
    u_pref = (
        pd.DataFrame({"s": p_sid, "i": pref_ev["itemid"].to_numpy()})
        .drop_duplicates()
        .groupby("s")
        .size()
    )
    p_agg["unique_items_viewed"] = u_pref.reindex(p_agg.index, fill_value=1).astype("int32")
    p_agg["elapsed_seconds"] = ((p_agg["snap_ts"] - p_agg["session_start_ts"]) / 1000.0).round(1)
    p_agg["ends_with_cart"] = np.where(p_agg["last_is_cart"] == 1, "Yes", "No")

    dt = pd.to_datetime(p_agg["session_start_ts"], unit="ms")
    p_agg["start_hour"] = dt.dt.hour.astype("int32")
    p_agg["is_weekend"] = np.where(dt.dt.day_of_week >= 5, "Yes", "No")

    p_agg["visitorid"] = sessions.loc[p_agg.index, "visitorid"].astype("int32")
    p_agg["prior_sessions"] = sessions.loc[p_agg.index, "prior_sessions"].astype("int32")
    p_agg["prior_purchases"] = sessions.loc[p_agg.index, "prior_purchases"].astype("int32")
    p_agg[config.TARGET_COLUMN] = np.where(target_yes[p_agg.index.to_numpy()], config.POSITIVE_CLASS, config.NEGATIVE_CLASS)

    # Order chronologically by session_start_ts and remove duplicate feature+target rows if any
    cols = [
        "session_id",
        "visitorid",
        "session_start_ts",
        "views_count",
        "unique_items_viewed",
        "add_to_cart_count",
        "elapsed_seconds",
        "ends_with_cart",
        "start_hour",
        "is_weekend",
        "prior_sessions",
        "prior_purchases",
        config.TARGET_COLUMN,
    ]
    out = p_agg.reset_index()[cols].sort_values("session_start_ts", kind="stable").reset_index(drop=True)

    # Stratified sample (preserving chronological order) if dataset exceeds max_rows
    if max_rows and len(out) > max_rows:
        rng = np.random.RandomState(config.RANDOM_STATE)
        pos_idx = out.index[out[config.TARGET_COLUMN] == config.POSITIVE_CLASS].to_numpy()
        neg_idx = out.index[out[config.TARGET_COLUMN] == config.NEGATIVE_CLASS].to_numpy()
        n_pos = int(round(max_rows * (len(pos_idx) / len(out))))
        n_neg = max_rows - n_pos
        chosen = np.sort(np.concatenate([
            rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False),
            rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False),
        ]))
        out = out.iloc[chosen].reset_index(drop=True)

    return out, sessions


# =============================================================================
# 4. BUILD CUSTOMER BEHAVIORAL PROFILES FOR K-MEANS SEGMENTATION
# =============================================================================
def build_customer_profiles(
    events: pd.DataFrame,
    sessions: pd.DataFrame,
    max_customers: int | None = config.MAX_CUSTOMER_PROFILES,
) -> pd.DataFrame:
    """Aggregate visitor-level behavioral profiles for K-Means Customer Segmentation."""
    max_ts = int(events["timestamp"].max())
    s = sessions.assign(
        cart_sess=(sessions["add_to_cart_count"] > 0).astype(np.int32),
        abandoned=((sessions["add_to_cart_count"] > 0) & (sessions["transaction_count"] == 0)).astype(np.int32),
        buy_sess=(sessions["transaction_count"] > 0).astype(np.int32),
    )
    cust = s.groupby("visitorid", sort=True).agg(
        total_sessions=("start_ts", "size"),
        total_events=("total_events", "sum"),
        total_views=("total_views", "sum"),
        unique_items_viewed=("unique_items_viewed", "sum"),
        total_add_to_cart=("add_to_cart_count", "sum"),
        total_transactions=("transaction_count", "sum"),
        purchase_sessions=("buy_sess", "sum"),
        cart_sessions=("cart_sess", "sum"),
        abandoned_sessions=("abandoned", "sum"),
        avg_session_duration_sec=("session_duration_sec", "mean"),
        avg_items_per_session=("unique_items_viewed", "mean"),
        last_ts=("end_ts", "max"),
    )
    # Focus segmentation on engaged visitors (>= 2 sessions or >= 3 events)
    engaged = cust[(cust["total_sessions"] >= 2) | (cust["total_events"] >= 3)].copy()
    if engaged.empty:
        engaged = cust.copy()

    engaged["cart_abandonment_rate"] = np.where(
        engaged["cart_sessions"] > 0,
        (engaged["abandoned_sessions"] / engaged["cart_sessions"].clip(lower=1)).round(4),
        0.0,
    )
    engaged["recency_days"] = ((max_ts - engaged["last_ts"]) / 86_400_000.0).round(2)
    engaged["avg_session_duration_sec"] = engaged["avg_session_duration_sec"].round(2)
    engaged["avg_items_per_session"] = engaged["avg_items_per_session"].round(2)
    engaged["converted"] = np.where(engaged["total_transactions"] > 0, "Yes", "No")

    keep_cols = [
        "visitorid",
        *config.CLUSTERING_FEATURES,
        "total_transactions",
        "converted",
    ]
    out = engaged.reset_index()[keep_cols]
    if max_customers and len(out) > max_customers:
        out = out.sample(n=max_customers, random_state=config.RANDOM_STATE).sort_values("visitorid").reset_index(drop=True)
    return out


# =============================================================================
# 5. COMPUTE DATASET-WIDE BUSINESS INSIGHTS SUMMARY (FOR HOME & INSIGHTS PAGES)
# =============================================================================
def build_insights_summary(events: pd.DataFrame, sessions: pd.DataFrame, clean_df: pd.DataFrame, clean_stats: dict) -> dict:
    """Summarize real Retailrocket KPIs, conversion funnel, and temporal patterns."""
    ev_counts = events["event"].value_counts().to_dict()
    views = int(ev_counts.get("view", 0))
    carts = int(ev_counts.get("addtocart", 0))
    txs = int(ev_counts.get("transaction", 0))

    n_sess = int(len(sessions))
    sess_with_view = int((sessions["total_views"] > 0).sum())
    sess_with_cart = int((sessions["add_to_cart_count"] > 0).sum())
    sess_with_tx = int((sessions["transaction_count"] > 0).sum())
    cart_abandoned = int(((sessions["add_to_cart_count"] > 0) & (sessions["transaction_count"] == 0)).sum())

    dt = pd.to_datetime(events["timestamp"], unit="ms")
    hourly = dt.dt.hour.value_counts().sort_index()
    dow = dt.dt.day_of_week.value_counts().sort_index()
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # Conversion rate by early add-to-cart count in clean_df
    by_cart = (
        clean_df.assign(y=(clean_df[config.TARGET_COLUMN] == config.POSITIVE_CLASS).astype(int))
        .groupby("add_to_cart_count")["y"]
        .agg(sessions="size", conversion_rate=lambda s: round(float(s.mean() * 100), 2))
        .reset_index()
        .to_dict(orient="records")
    )

    return {
        **clean_stats,
        "unique_visitors": int(events["visitorid"].nunique()),
        "unique_items": int(events["itemid"].nunique()),
        "total_sessions": n_sess,
        "date_min": dt.min().strftime("%Y-%m-%d"),
        "date_max": dt.max().strftime("%Y-%m-%d"),
        "event_counts": {"view": views, "addtocart": carts, "transaction": txs},
        "session_funnel": [
            {"stage": "Sessions with Views", "count": sess_with_view, "pct": round(sess_with_view / max(n_sess, 1) * 100, 2)},
            {"stage": "Sessions with Add-to-Cart", "count": sess_with_cart, "pct": round(sess_with_cart / max(n_sess, 1) * 100, 2)},
            {"stage": "Sessions with Purchase", "count": sess_with_tx, "pct": round(sess_with_tx / max(n_sess, 1) * 100, 2)},
        ],
        "cart_abandonment_rate_pct": round(cart_abandoned / max(sess_with_cart, 1) * 100, 2),
        "session_conversion_rate_pct": round(sess_with_tx / max(n_sess, 1) * 100, 2),
        "modeling_rows": int(len(clean_df)),
        "modeling_positive_rate_pct": round(float((clean_df[config.TARGET_COLUMN] == config.POSITIVE_CLASS).mean() * 100), 2),
        "hourly_events": [{"hour": int(h), "events": int(hourly.get(h, 0))} for h in range(24)],
        "weekday_events": [{"day": day_names[d], "events": int(dow.get(d, 0))} for d in range(7)],
        "conversion_by_early_carts": by_cart,
    }


# =============================================================================
# 6. UNIFIED clean() ENTRYPOINT (COMPATIBLE WITH TEACHER'S TEMPLATE & TESTS)
# =============================================================================
def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Transform raw Retailrocket events (or validate an existing snapshot table) into clean.csv format."""
    df_in = df.copy()

    # Case A: Already a processed snapshot table
    if config.TARGET_COLUMN in df_in.columns and "views_count" in df_in.columns:
        out = df_in.drop_duplicates().dropna(subset=[config.TARGET_COLUMN]).reset_index(drop=True)
        out = out[out[config.TARGET_COLUMN].isin([config.POSITIVE_CLASS, config.NEGATIVE_CLASS])]
        return out.reset_index(drop=True)

    # Case B: Raw Retailrocket events.csv clickstream log
    events_clean, log, _ = clean_events(df_in)
    snap_df, _ = build_modeling_snapshots(events_clean, max_rows=config.MAX_MODELING_ROWS)
    snap_df = snap_df.drop_duplicates().reset_index(drop=True)
    log.append(f"Built {len(snap_df):,} leakage-safe early-session modeling rows (target '{config.TARGET_COLUMN}')")
    if verbose:
        print("\n".join("  - " + line for line in log))
        print(f"  Clean modeling shape: {snap_df.shape}")
    return snap_df


def main():
    """Run full data preparation: events.csv -> clean.csv, customer_profiles.csv, insights_summary.json."""
    raw = load_raw()
    print(f"Raw events shape: {raw.shape}")
    events_clean, log, stats = clean_events(raw)
    snap_df, sessions = build_modeling_snapshots(events_clean, max_rows=config.MAX_MODELING_ROWS)
    snap_df = snap_df.drop_duplicates().reset_index(drop=True)
    cust_df = build_customer_profiles(events_clean, sessions, max_customers=config.MAX_CUSTOMER_PROFILES)
    insights = build_insights_summary(events_clean, sessions, snap_df, stats)

    for line in log:
        print("  - " + line)
    print(f"  - Built modeling dataset: {snap_df.shape}")
    print(f"  - Built customer profiles: {cust_df.shape}")

    config.PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    snap_df.to_csv(config.PROCESSED_DATA_PATH, index=False)
    cust_df.to_csv(config.CUSTOMER_PROFILES_PATH, index=False)
    config.INSIGHTS_SUMMARY_PATH.write_text(json.dumps(insights, indent=2), encoding="utf-8")
    print(f"Saved ➜ {config.PROCESSED_DATA_PATH.relative_to(config.ROOT_DIR)}")
    print(f"Saved ➜ {config.CUSTOMER_PROFILES_PATH.relative_to(config.ROOT_DIR)}")
    print(f"Saved ➜ {config.INSIGHTS_SUMMARY_PATH.relative_to(config.ROOT_DIR)}")


if __name__ == "__main__":
    main()
