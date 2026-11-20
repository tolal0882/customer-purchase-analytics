"""
Step 1 – Data cleaning, session construction, and feature engineering:
    data/raw/events.csv  ➜  data/processed/clean.csv               (early-session purchase prediction)
                         ➜  data/processed/customer_profiles.csv   (customer-level K-Means profiles)
                         ➜  data/processed/session_profiles.csv    (session-level K-Means profiles)
                         ➜  data/processed/insights_summary.json   (EDA, quality audit, funnel, transitions)

Run from the project root:
    python -m ml.data.make_dataset
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from ml import config

TS_MIN_MS = 946_684_800_000
TS_MAX_MS = 4_102_444_800_000


# =============================================================================
# 1. LOAD RAW RETAILROCKET CLICKSTREAM DATA
# =============================================================================
def load_raw(path=config.RAW_DATA_PATH, nrows: int | None = None) -> pd.DataFrame:
    """Read raw events.csv without mutating the original file."""
    return pd.read_csv(path, nrows=nrows)


# =============================================================================
# 2. CLEAN RAW CLICKSTREAM EVENTS & ENGINEER SESSIONS (WITH FULL AUDIT LOG)
# =============================================================================
def clean_events(
    raw_df: pd.DataFrame,
    timeout_minutes: float = config.SESSION_TIMEOUT_MINUTES,
) -> tuple[pd.DataFrame, list[str], dict]:
    """Validate schema, clean duplicates/timestamps, and construct gap-based sessions."""
    log: list[str] = []
    steps_table: list[dict] = []
    df = raw_df.copy()
    n_raw = len(df)
    mem_raw_mb = round(float(df.memory_usage(deep=True).sum() / 1e6), 1)

    # Record raw missing values before cleaning
    raw_missing = {c: int(v) for c, v in df.isna().sum().items()}

    # 1. Normalize event strings
    df["event"] = df["event"].astype("string").str.strip().str.lower()

    # 2. Coerce numeric columns
    for col in ("timestamp", "visitorid", "itemid"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Check & remove invalid timestamps, missing IDs, or unknown event types
    bad_ts = int((~df["timestamp"].between(TS_MIN_MS, TS_MAX_MS)).sum())
    bad_ids = int((df["visitorid"].isna() | df["itemid"].isna()).sum())
    bad_ev = int((~df["event"].isin(config.VALID_EVENTS)).sum())
    valid_mask = (
        df["timestamp"].between(TS_MIN_MS, TS_MAX_MS)
        & df["visitorid"].notna()
        & df["itemid"].notna()
        & df["event"].isin(config.VALID_EVENTS)
    )
    n_invalid = int((~valid_mask).sum())
    steps_table.append({
        "step": "Validate schema, timestamps (2000–2100), IDs & event types",
        "rows_before": n_raw,
        "rows_after": n_raw - n_invalid,
        "rows_affected": n_invalid,
        "action": "removed" if n_invalid else "verified clean",
    })
    if n_invalid:
        df = df.loc[valid_mask]
    log.append(f"Removed {n_invalid} invalid timestamp/ID/event rows")

    # 4. Detect & remove exact duplicate rows
    before_dedup = len(df)
    df = df.drop_duplicates()
    n_dups = before_dedup - len(df)
    steps_table.append({
        "step": "Remove exact duplicate rows (keep first occurrence)",
        "rows_before": before_dedup,
        "rows_after": len(df),
        "rows_affected": n_dups,
        "action": "removed",
    })
    log.append(f"Removed {n_dups} exact duplicate rows")

    # 5. Optimize dtypes and sort chronologically per visitor
    df["timestamp"] = df["timestamp"].astype("int64")
    df["visitorid"] = df["visitorid"].astype("int32")
    df["itemid"] = df["itemid"].astype("int32")
    df = df.sort_values(["visitorid", "timestamp"], kind="stable").reset_index(drop=True)
    mem_clean_mb = round(float(df.memory_usage(deep=True).sum() / 1e6), 1)
    steps_table.append({
        "step": "Sort by (visitorid, timestamp) & downcast dtypes",
        "rows_before": len(df),
        "rows_after": len(df),
        "rows_affected": len(df),
        "action": f"sorted & optimized ({mem_raw_mb} MB -> {mem_clean_mb} MB)",
    })

    # 6. Vectorized session construction (configurable timeout_minutes)
    v = df["visitorid"].to_numpy()
    t = df["timestamp"].to_numpy()
    timeout_ms = float(timeout_minutes) * 60_000
    new_sess = np.empty(len(df), dtype=bool)
    new_sess[0] = True
    new_sess[1:] = (v[1:] != v[:-1]) | ((t[1:] - t[:-1]) > timeout_ms)

    sid = (np.cumsum(new_sess) - 1).astype("int32")
    start_idx = np.flatnonzero(new_sess)
    pos = (np.arange(len(df)) - start_idx[sid]).astype("int32")
    df["session_id"] = sid
    df["event_pos"] = pos
    n_sess = int(sid[-1]) + 1 if len(sid) else 0
    steps_table.append({
        "step": f"Engineer session_id ({timeout_minutes:g}-minute inactivity threshold)",
        "rows_before": len(df),
        "rows_after": len(df),
        "rows_affected": n_sess,
        "action": f"created {n_sess:,} sessions",
    })
    log.append(f"Engineered {n_sess:,} sessions ({timeout_minutes:g}-min timeout)")

    # Sensitivity of session count across different timeouts (15m, 30m, 60m)
    sensitivity = []
    for tm in (15, 30, 60):
        ns = int(1 + np.count_nonzero((v[1:] != v[:-1]) | ((t[1:] - t[:-1]) > tm * 60_000))) if len(v) else 0
        sensitivity.append({
            "timeout_minutes": tm,
            "sessions": ns,
            "avg_events_per_session": round(len(df) / max(ns, 1), 2),
        })

    stats = {
        "raw_rows": int(n_raw),
        "clean_event_rows": int(len(df)),
        "duplicates_removed": int(n_dups),
        "invalid_rows_removed": int(n_invalid),
        "invalid_timestamps": bad_ts,
        "invalid_ids": bad_ids,
        "invalid_events": bad_ev,
        "raw_missing_by_column": raw_missing,
        "memory_raw_mb": mem_raw_mb,
        "memory_clean_mb": mem_clean_mb,
        "session_timeout_minutes": float(timeout_minutes),
        "cleaning_steps": steps_table,
        "timeout_sensitivity": sensitivity,
    }
    return df, log, stats


# =============================================================================
# 3. BUILD LEAKAGE-SAFE EARLY-SESSION PURCHASE PREDICTION DATASET (clean.csv)
# =============================================================================
def build_modeling_snapshots(
    events: pd.DataFrame,
    max_rows: int | None = config.MAX_MODELING_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (snapshot_modeling_df, full_sessions_df, snapshot_eligibility_report)."""
    sid = events["session_id"].to_numpy()
    ts = events["timestamp"].to_numpy()
    pos = events["event_pos"].to_numpy()
    ev_type = events["event"].to_numpy()
    is_view = ev_type == "view"
    is_cart = ev_type == "addtocart"
    is_tx = ev_type == "transaction"
    n_sessions = int(sid[-1]) + 1

    # Full-session aggregates (retrospective: used for session profiles, customer profiles & prior-session history)
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
    sessions["session_duration_sec"] = ((sessions["end_ts"] - sessions["start_ts"]) / 1000.0).round(2)
    sessions["avg_time_between_events_sec"] = np.where(
        sessions["total_events"] > 1,
        (sessions["session_duration_sec"] / (sessions["total_events"] - 1).clip(lower=1)).round(2),
        0.0,
    )
    sessions["cart_interaction_ratio"] = (sessions["add_to_cart_count"] / sessions["total_events"].clip(lower=1)).round(4)
    sessions["item_revisit_ratio"] = np.where(
        sessions["total_views"] > 0,
        (1.0 - (sessions["unique_items_viewed"] / sessions["total_views"].clip(lower=1)).clip(0, 1)).round(4),
        0.0,
    )
    sessions["converted"] = np.where(sessions["transaction_count"] > 0, "Yes", "No")

    # Visitor history from STRICTLY EARLIER sessions only (shifted cumulative sums -> zero leakage)
    s_ord = sessions[["visitorid", "start_ts", "end_ts", "transaction_count"]].sort_values(
        ["visitorid", "start_ts"], kind="stable"
    )
    g_vis = s_ord.groupby("visitorid", sort=False)
    s_ord["prior_sessions"] = g_vis.cumcount().astype("int32")
    s_ord["prior_purchases"] = (g_vis["transaction_count"].cumsum() - s_ord["transaction_count"]).astype("int32")
    prev_end = g_vis["end_ts"].shift(1)
    s_ord["hours_since_prev_session"] = ((s_ord["start_ts"] - prev_end) / 3_600_000.0).fillna(0.0).clip(lower=0.0).round(2)
    sessions["prior_sessions"] = s_ord["prior_sessions"].reindex(sessions.index)
    sessions["prior_purchases"] = s_ord["prior_purchases"].reindex(sessions.index)
    sessions["hours_since_prev_session"] = s_ord["hours_since_prev_session"].reindex(sessions.index)

    # Identify the 3rd non-transaction event in each session (early-session prediction snapshot)
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

    ex_short = snap_pos < 0
    ex_already_bought = (~ex_short) & (prefix_tx > 0)
    ex_no_future = (~ex_short) & (~ex_already_bought) & (after_cnt == 0)
    eligible = ~(ex_short | ex_already_bought | ex_no_future)
    target_yes = after_tx > 0

    eligibility_report = {
        "snapshot_events": int(n_ev),
        "total_sessions": int(n_sessions),
        "excluded_fewer_than_n_events": int(ex_short.sum()),
        "excluded_transaction_at_or_before_snapshot": int(ex_already_bought.sum()),
        "excluded_no_post_snapshot_window": int(ex_no_future.sum()),
        "eligible_sessions_total": int(eligible.sum()),
        "eligible_positives": int((eligible & target_yes).sum()),
        "eligible_negatives": int((eligible & ~target_yes).sum()),
        "eligible_positive_rate_pct": round(float((eligible & target_yes).sum() / max(int(eligible.sum()), 1) * 100), 2),
    }

    pref_ev = events.loc[in_prefix & eligible[sid]]
    p_sid = pref_ev["session_id"].to_numpy()
    p_ts = pref_ev["timestamp"].to_numpy()
    p_pos = pref_ev["event_pos"].to_numpy()
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

    # First cart position (-1 if no cart event in the first 3 actions)
    cart_mask = p_cart == 1
    first_cart = pd.Series(p_pos[cart_mask]).groupby(p_sid[cart_mask]).min()
    p_agg["first_cart_position"] = first_cart.reindex(p_agg.index).fillna(-1).astype("int32")

    p_agg["elapsed_seconds"] = ((p_agg["snap_ts"] - p_agg["session_start_ts"]) / 1000.0).round(1)
    p_agg["ends_with_cart"] = np.where(p_agg["last_is_cart"] == 1, "Yes", "No")

    dt = pd.to_datetime(p_agg["session_start_ts"], unit="ms")
    p_agg["start_hour"] = dt.dt.hour.astype("int32")
    p_agg["start_dayofweek"] = dt.dt.day_of_week.astype("int32")
    p_agg["is_weekend"] = np.where(dt.dt.day_of_week >= 5, "Yes", "No")

    p_agg["visitorid"] = sessions.loc[p_agg.index, "visitorid"].astype("int32")
    p_agg["prior_sessions"] = sessions.loc[p_agg.index, "prior_sessions"].astype("int32")
    p_agg["prior_purchases"] = sessions.loc[p_agg.index, "prior_purchases"].astype("int32")
    p_agg[config.TARGET_COLUMN] = np.where(
        target_yes[p_agg.index.to_numpy()], config.POSITIVE_CLASS, config.NEGATIVE_CLASS
    )

    cols = [
        "session_id",
        "visitorid",
        "session_start_ts",
        "views_count",
        "unique_items_viewed",
        "add_to_cart_count",
        "first_cart_position",
        "elapsed_seconds",
        "ends_with_cart",
        "start_hour",
        "start_dayofweek",
        "is_weekend",
        "prior_sessions",
        "prior_purchases",
        config.TARGET_COLUMN,
    ]
    out = p_agg.reset_index()[cols].sort_values("session_start_ts", kind="stable").reset_index(drop=True)

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

    return out, sessions, eligibility_report


# =============================================================================
# 4. BUILD CUSTOMER-LEVEL & SESSION-LEVEL PROFILES FOR K-MEANS SEGMENTATION
# =============================================================================
def build_customer_profiles(
    events: pd.DataFrame,
    sessions: pd.DataFrame,
    max_customers: int | None = config.MAX_CUSTOMER_PROFILES,
) -> pd.DataFrame:
    """Aggregate visitor-level behavioral profiles for Customer K-Means Segmentation."""
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
        first_ts=("start_ts", "min"),
        last_ts=("end_ts", "max"),
    )
    engaged = cust[(cust["total_sessions"] >= 2) | (cust["total_events"] >= 3)].copy()
    if engaged.empty:
        engaged = cust.copy()

    engaged["purchase_frequency"] = (engaged["purchase_sessions"] / engaged["total_sessions"].clip(lower=1)).round(4)
    engaged["avg_cart_additions_per_session"] = (engaged["total_add_to_cart"] / engaged["total_sessions"].clip(lower=1)).round(3)
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
        *config.CUSTOMER_CLUSTERING_FEATURES,
        "purchase_frequency",
        "total_transactions",
        "converted",
    ]
    out = engaged.reset_index()[keep_cols]
    if max_customers and len(out) > max_customers:
        out = out.sample(n=max_customers, random_state=config.RANDOM_STATE).sort_values("visitorid").reset_index(drop=True)
    return out


def build_session_profiles(
    sessions: pd.DataFrame,
    max_sessions: int | None = config.MAX_SESSION_PROFILES,
) -> pd.DataFrame:
    """Extract multi-event session behavioral profiles for Session K-Means Segmentation."""
    multi = sessions[sessions["total_events"] >= 2].copy()
    if multi.empty:
        multi = sessions.copy()
    keep_cols = [
        "session_id",
        "visitorid",
        *config.SESSION_CLUSTERING_FEATURES,
        "transaction_count",
        "converted",
    ]
    out = multi.reset_index()[keep_cols]
    if max_sessions and len(out) > max_sessions:
        out = out.sample(n=max_sessions, random_state=config.RANDOM_STATE).sort_values("session_id").reset_index(drop=True)
    return out


# =============================================================================
# 5. COMPUTE COMPREHENSIVE EDA & BUSINESS INSIGHTS SUMMARY
# =============================================================================
def build_insights_summary(
    events: pd.DataFrame,
    sessions: pd.DataFrame,
    clean_df: pd.DataFrame,
    clean_stats: dict,
    eligibility_report: dict,
) -> dict:
    """Compute real KPIs, event transitions, top items, daily trends, and conversion funnel."""
    ev_counts = events["event"].value_counts().to_dict()
    views = int(ev_counts.get("view", 0))
    carts = int(ev_counts.get("addtocart", 0))
    txs = int(ev_counts.get("transaction", 0))

    n_sess = int(len(sessions))
    sess_with_view = int((sessions["total_views"] > 0).sum())
    sess_with_cart = int((sessions["add_to_cart_count"] > 0).sum())
    sess_with_tx = int((sessions["transaction_count"] > 0).sum())
    cart_abandoned = int(((sessions["add_to_cart_count"] > 0) & (sessions["transaction_count"] == 0)).sum())
    single_event_sess = int((sessions["total_events"] == 1).sum())
    multi_sess_visitors = int((sessions.groupby("visitorid").size() >= 2).sum())

    dt = pd.to_datetime(events["timestamp"], unit="ms")
    hourly = dt.dt.hour.value_counts().sort_index()
    dow = dt.dt.day_of_week.value_counts().sort_index()
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # Daily event trend (view, addtocart, transaction per day)
    daily_df = (
        pd.DataFrame({"date": dt.dt.strftime("%Y-%m-%d"), "event": events["event"].to_numpy()})
        .groupby(["date", "event"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for e in config.VALID_EVENTS:
        if e not in daily_df.columns:
            daily_df[e] = 0
    daily_trends = daily_df[["date", "view", "addtocart", "transaction"]].to_dict(orient="records")

    # Within-session event transition matrix (view -> addtocart -> transaction)
    sid_arr = events["session_id"].to_numpy()
    cur_ev = events["event"].astype(str).to_numpy()
    nxt_ev = np.roll(cur_ev, -1)
    same_sess = np.roll(sid_arr, -1) == sid_arr
    if len(same_sess):
        same_sess[-1] = False
    trans_ct = (
        pd.crosstab(pd.Series(cur_ev[same_sess], name="from"), pd.Series(nxt_ev[same_sess], name="to"))
        .reindex(index=list(config.VALID_EVENTS), columns=list(config.VALID_EVENTS), fill_value=0)
    )
    transitions = {
        r: {c: int(trans_ct.loc[r, c]) for c in config.VALID_EVENTS}
        for r in config.VALID_EVENTS
    }

    # Top 15 products by views, add-to-carts, and transactions
    item_ct = pd.crosstab(events["itemid"], events["event"])
    for e in config.VALID_EVENTS:
        if e not in item_ct.columns:
            item_ct[e] = 0
    item_ct["total"] = item_ct["view"] + item_ct["addtocart"] + item_ct["transaction"]
    top_items = (
        item_ct.sort_values("total", ascending=False)
        .head(15)
        .reset_index()
        .astype({"itemid": str})
        .to_dict(orient="records")
    )

    by_cart = (
        clean_df.assign(y=(clean_df[config.TARGET_COLUMN] == config.POSITIVE_CLASS).astype(int))
        .groupby("add_to_cart_count")["y"]
        .agg(sessions="size", conversion_rate=lambda s: round(float(s.mean() * 100), 2))
        .reset_index()
        .to_dict(orient="records")
    )

    n_categories = 0
    n_root_categories = 0
    if config.CATEGORY_TREE_PATH.exists():
        ct_df = pd.read_csv(config.CATEGORY_TREE_PATH)
        n_categories = int(ct_df["categoryid"].nunique())
        n_root_categories = int(ct_df["parentid"].isna().sum())

    raw_files_inventory = []
    for label, p in (
        ("events.csv", config.RAW_DATA_PATH),
        ("category_tree.csv", config.CATEGORY_TREE_PATH),
        ("item_properties_part1.csv", config.ITEM_PROPS_PART1_PATH),
        ("item_properties_part2.csv", config.ITEM_PROPS_PART2_PATH),
    ):
        raw_files_inventory.append({
            "file": label,
            "present": bool(p.exists()),
            "size_mb": round(p.stat().st_size / (1024 * 1024), 2) if p.exists() else 0.0,
        })

    return {
        **clean_stats,
        "snapshot_eligibility": eligibility_report,
        "unique_visitors": int(events["visitorid"].nunique()),
        "returning_visitors": multi_sess_visitors,
        "unique_items": int(events["itemid"].nunique()),
        "total_categories": n_categories,
        "root_categories": n_root_categories,
        "raw_files_inventory": raw_files_inventory,
        "total_sessions": n_sess,
        "single_event_sessions": single_event_sess,
        "single_event_sessions_pct": round(single_event_sess / max(n_sess, 1) * 100, 2),
        "avg_session_duration_sec": round(float(sessions["session_duration_sec"].mean()), 2),
        "avg_events_per_session": round(float(sessions["total_events"].mean()), 2),
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
        "daily_trends": daily_trends,
        "event_transitions": transitions,
        "top_items": top_items,
        "conversion_by_early_carts": by_cart,
    }


# =============================================================================
# 6. UNIFIED clean() ENTRYPOINT (COMPATIBLE WITH TEACHER'S TEMPLATE & TESTS)
# =============================================================================
def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Transform raw Retailrocket events (or validate an existing snapshot table) into clean.csv format."""
    df_in = df.copy()
    if config.TARGET_COLUMN in df_in.columns and "views_count" in df_in.columns:
        out = df_in.drop_duplicates().dropna(subset=[config.TARGET_COLUMN]).reset_index(drop=True)
        out = out[out[config.TARGET_COLUMN].isin([config.POSITIVE_CLASS, config.NEGATIVE_CLASS])]
        return out.reset_index(drop=True)

    events_clean, log, _ = clean_events(df_in)
    snap_df, _, _ = build_modeling_snapshots(events_clean, max_rows=config.MAX_MODELING_ROWS)
    snap_df = snap_df.drop_duplicates().reset_index(drop=True)
    log.append(f"Built {len(snap_df):,} leakage-safe early-session modeling rows (target '{config.TARGET_COLUMN}')")
    if verbose:
        print("\n".join("  - " + line for line in log))
        print(f"  Clean modeling shape: {snap_df.shape}")
    return snap_df


def main(timeout_minutes: float = config.SESSION_TIMEOUT_MINUTES):
    """Run full data preparation: events.csv -> clean.csv, customer_profiles.csv, session_profiles.csv, insights_summary.json."""
    raw = load_raw()
    print(f"Raw events shape: {raw.shape}")
    events_clean, log, stats = clean_events(raw, timeout_minutes=timeout_minutes)
    snap_df, sessions, elig = build_modeling_snapshots(events_clean, max_rows=config.MAX_MODELING_ROWS)
    snap_df = snap_df.drop_duplicates().reset_index(drop=True)
    cust_df = build_customer_profiles(events_clean, sessions, max_customers=config.MAX_CUSTOMER_PROFILES)
    sess_df = build_session_profiles(sessions, max_sessions=config.MAX_SESSION_PROFILES)
    insights = build_insights_summary(events_clean, sessions, snap_df, stats, elig)

    for line in log:
        print("  - " + line)
    print(f"  - Built modeling dataset: {snap_df.shape} (from {elig['eligible_sessions_total']:,} eligible sessions)")
    print(f"  - Built customer profiles: {cust_df.shape}")
    print(f"  - Built session profiles:  {sess_df.shape}")

    config.PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    snap_df.to_csv(config.PROCESSED_DATA_PATH, index=False)
    cust_df.to_csv(config.CUSTOMER_PROFILES_PATH, index=False)
    sess_df.to_csv(config.SESSION_PROFILES_PATH, index=False)
    config.INSIGHTS_SUMMARY_PATH.write_text(json.dumps(insights, indent=2), encoding="utf-8")
    print(f"Saved ➜ {config.PROCESSED_DATA_PATH.relative_to(config.ROOT_DIR)}")
    print(f"Saved ➜ {config.CUSTOMER_PROFILES_PATH.relative_to(config.ROOT_DIR)}")
    print(f"Saved ➜ {config.SESSION_PROFILES_PATH.relative_to(config.ROOT_DIR)}")
    print(f"Saved ➜ {config.INSIGHTS_SUMMARY_PATH.relative_to(config.ROOT_DIR)}")


if __name__ == "__main__":
    main()
