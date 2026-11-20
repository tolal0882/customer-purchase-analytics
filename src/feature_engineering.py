"""Session-, snapshot- and customer-level feature engineering.

LEAKAGE RULES (enforced in code and covered by tests/test_leakage.py):
  * Snapshot features are computed ONLY from events at/before the snapshot.
  * Sessions with a transaction at/before the snapshot are excluded (target undefined).
  * transaction_count / visitorid / itemid / transactionid are never model features.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import settings

# =============================================================================
# 1. FEATURE REGISTRY & LEAKAGE GUARD
#    - Lists all allowed pre-snapshot features (base, category, prior-session history).
#    - FORBIDDEN_FEATURES explicitly blocks IDs (visitorid, itemid, transactionid)
#      and post-purchase columns (transaction_count, target) from ever entering a model.
# =============================================================================
SNAPSHOT_BASE_FEATURES = [
    "total_events", "total_views", "unique_items_viewed", "add_to_cart_count",
    "session_duration_sec", "avg_time_between_events_sec", "cart_interaction_ratio",
    "item_revisit_ratio", "first_cart_position", "ends_with_cart", "start_hour", "start_dayofweek",
]
SNAPSHOT_CATEGORY_FEATURES = ["unique_categories_viewed"]
SNAPSHOT_HISTORY_FEATURES = ["hist_prior_sessions", "hist_prior_purchases", "hist_hours_since_prev_session"]
FORBIDDEN_FEATURES = {"visitorid", "itemid", "transactionid", "transaction_count", "target", "session_id",
                      "start_ts", "snapshot_ts", "purchase_sessions", "total_transactions"}
TARGET = "target"

FEATURE_DESCRIPTIONS = {
    "total_events": "Events observed up to the snapshot (constant when the snapshot is 'after N events').",
    "total_views": "Product views up to the snapshot.",
    "unique_items_viewed": "Distinct products viewed up to the snapshot.",
    "add_to_cart_count": "Add-to-cart events up to the snapshot.",
    "session_duration_sec": "Seconds between session start and the last pre-snapshot event.",
    "avg_time_between_events_sec": "Mean gap between consecutive pre-snapshot events (0 if only one event).",
    "cart_interaction_ratio": "Add-to-cart events divided by events so far.",
    "item_revisit_ratio": "Share of views that were repeat views of an already-viewed item.",
    "first_cart_position": "0-based position of the first add-to-cart (-1 if none yet).",
    "ends_with_cart": "1 if the last pre-snapshot event was an add-to-cart.",
    "start_hour": "Hour of day (UTC) at which the session started.",
    "start_dayofweek": "Day of week (0=Mon) at which the session started.",
    "unique_categories_viewed": "Distinct categories viewed (needs item_properties; as-of join, no look-ahead).",
    "hist_prior_sessions": "Visitor's earlier sessions (history window strictly before this session).",
    "hist_prior_purchases": "Visitor's transactions in EARLIER sessions only (default OFF: potentially strong, so opt-in).",
    "hist_hours_since_prev_session": "Hours since the visitor's previous session ended (NaN if first session).",
}


def assert_no_leakage_features(features: list[str]) -> None:
    bad = sorted(set(features) & FORBIDDEN_FEATURES)
    if bad:
        raise ValueError(f"Forbidden/leaky features selected: {bad}")


# =============================================================================
# 2. SESSION-LEVEL FEATURE AGGREGATION
#    - Shared aggregator used for BOTH full-session descriptive analytics AND
#      pre-snapshot ML features. Leakage is prevented by passing ONLY pre-snapshot
#      events into this function when building prediction datasets.
# =============================================================================
def aggregate_event_features(ev: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a set of events (must carry session_id/event_pos) into one row per session.

    The same code computes full-session descriptive features and pre-snapshot features;
    the caller decides WHICH events are passed in (that is what prevents leakage)."""
    is_view = (ev["event"] == "view").to_numpy()
    is_cart = (ev["event"] == "addtocart").to_numpy()
    is_tx = (ev["event"] == "transaction").to_numpy()
    base = pd.DataFrame({
        "session_id": ev["session_id"].to_numpy(),
        "ts": ev["timestamp"].to_numpy(),
        "is_view": is_view.astype(np.int32),
        "is_cart": is_cart.astype(np.int32),
        "is_tx": is_tx.astype(np.int32),
    })
    agg = base.groupby("session_id", sort=True).agg(
        total_events=("ts", "size"), start_ts=("ts", "min"), end_ts=("ts", "max"),
        total_views=("is_view", "sum"), add_to_cart_count=("is_cart", "sum"),
        transaction_count=("is_tx", "sum"), ends_with_cart=("is_cart", "last"))
    idx = agg.index

    sid = ev["session_id"].to_numpy()
    item = ev["itemid"].to_numpy()
    vmask = is_view
    uniq_views = pd.DataFrame({"s": sid[vmask], "i": item[vmask]}).drop_duplicates()
    agg["unique_items_viewed"] = uniq_views.groupby("s").size().reindex(idx, fill_value=0).astype("int32")
    uniq_all = pd.DataFrame({"s": sid, "i": item}).drop_duplicates().groupby("s").size()
    agg["unique_items_total"] = uniq_all.reindex(idx, fill_value=0).astype("int32")

    pos = ev["event_pos"].to_numpy()
    fc = pd.Series(pos[is_cart]).groupby(sid[is_cart]).min()
    agg["first_cart_position"] = fc.reindex(idx).fillna(-1).astype("int32")

    if "categoryid" in ev.columns:
        cat = ev["categoryid"]
        m = vmask & cat.notna().to_numpy()
        pairs = pd.DataFrame({"s": sid[m], "c": cat.to_numpy()[m]}).drop_duplicates()
        agg["unique_categories_viewed"] = pairs.groupby("s").size().reindex(idx, fill_value=0).astype("int32")

    n = agg["total_events"]
    dur = (agg["end_ts"] - agg["start_ts"]) / 1000.0
    agg["session_duration_sec"] = dur
    agg["avg_time_between_events_sec"] = np.where(n > 1, dur / (n - 1).clip(lower=1), 0.0)
    agg["cart_interaction_ratio"] = agg["add_to_cart_count"] / n
    agg["item_revisit_ratio"] = np.where(agg["total_views"] > 0,
                                         1.0 - agg["unique_items_viewed"] / agg["total_views"].clip(lower=1), 0.0)
    agg["ends_with_cart"] = agg["ends_with_cart"].astype("int8")
    return agg


def build_session_table(events: pd.DataFrame) -> pd.DataFrame:
    """Full-session table (RETROSPECTIVE / descriptive: includes transaction_count).
    Used for dashboard, EDA, session clustering and customer profiles. NOT for prediction."""
    agg = aggregate_event_features(events)
    vis = events.groupby("session_id", sort=True)["visitorid"].first()
    agg.insert(0, "visitorid", vis.reindex(agg.index).to_numpy())
    agg["has_purchase"] = (agg["transaction_count"] > 0).astype("int8")
    start = pd.to_datetime(agg["start_ts"], unit="ms")
    agg["start_hour"] = start.dt.hour.astype("int8")
    agg["start_dayofweek"] = start.dt.day_of_week.astype("int8")
    return agg


# =============================================================================
# 3. LEAKAGE-SAFE EARLY-SESSION PREDICTION SNAPSHOTS
#    - Splits each session into TWO windows at the snapshot cutoff:
#        [Prefix Window: <= snapshot] -> used ONLY to compute input features X
#        [Future Window: >  snapshot] -> used ONLY to compute label y (purchase = 1/0)
#    - Excludes sessions with < N prefix events, sessions that already purchased
#      at/before the snapshot, and sessions with no future events left.
#    - Optional visitor history uses shift-by-1 cumsum so only STRICTLY EARLIER
#      sessions contribute (never the current or future sessions).
# =============================================================================
@dataclass
class SnapshotResult:
    data: pd.DataFrame
    report: dict
    params: dict = field(default_factory=dict)


def add_history_features(snap: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """Visitor history from STRICTLY EARLIER sessions (shift-by-one cumulative sums)."""
    s = sessions[["visitorid", "start_ts", "end_ts", "transaction_count"]].sort_values(["visitorid", "start_ts"])
    g = s.groupby("visitorid", sort=False)
    s["hist_prior_sessions"] = g.cumcount()
    s["hist_prior_purchases"] = g["transaction_count"].cumsum() - s["transaction_count"]
    prev_end = g["end_ts"].shift(1)
    s["hist_hours_since_prev_session"] = (s["start_ts"] - prev_end) / 3_600_000.0
    cols = SNAPSHOT_HISTORY_FEATURES
    return snap.join(s[cols], how="left")


def build_prediction_snapshots(events: pd.DataFrame, sessions: pd.DataFrame, policy: str = "n_events",
                               n_events: int = settings.DEFAULT_SNAPSHOT_EVENTS,
                               cutoff_minutes: float = settings.DEFAULT_SNAPSHOT_CUTOFF_MIN,
                               include_history: bool = True) -> SnapshotResult:
    """Build leakage-safe early-session prediction samples.

    policy="n_events": snapshot = right after the session's n-th NON-transaction event.
    policy="time":     snapshot = session start + cutoff_minutes.

    Inputs = events at/before snapshot. Target = 1 iff a transaction occurs strictly AFTER the
    snapshot within the same session. Exclusions (reported with counts):
      1. not enough pre-snapshot activity (n_events policy: fewer than n events; time: no event)
      2. a transaction already happened at/before snapshot (outcome already known)
      3. no events after the snapshot (no valid prediction window)"""
    if policy not in ("n_events", "time"):
        raise ValueError("policy must be 'n_events' or 'time'")
    sid = events["session_id"].to_numpy()
    ts = events["timestamp"].to_numpy()
    pos = events["event_pos"].to_numpy()
    is_tx = (events["event"] == "transaction").to_numpy()
    n_sessions = int(sessions.index.max()) + 1
    sess_start = np.zeros(n_sessions, dtype="int64")
    sess_start[sessions.index.to_numpy()] = sessions["start_ts"].to_numpy()

    if policy == "n_events":
        cum_nontx = pd.Series((~is_tx).astype(np.int32)).groupby(sid).cumsum().to_numpy()
        hit = (~is_tx) & (cum_nontx == n_events)
        snap_pos = np.full(n_sessions, -1, dtype="int64")
        snap_pos[sid[hit]] = pos[hit]
        snap_ts = np.full(n_sessions, -1, dtype="int64")
        snap_ts[sid[hit]] = ts[hit]
        in_prefix = (snap_pos[sid] >= 0) & (pos <= snap_pos[sid])
        min_prefix = n_events
    else:
        snap_ts = sess_start + int(cutoff_minutes * 60_000)
        in_prefix = ts <= snap_ts[sid]
        min_prefix = 1

    prefix_n = np.bincount(sid[in_prefix], minlength=n_sessions)
    prefix_tx = np.bincount(sid[in_prefix & is_tx], minlength=n_sessions)
    after = ~in_prefix
    after_n = np.bincount(sid[after], minlength=n_sessions)
    after_tx = np.bincount(sid[after & is_tx], minlength=n_sessions)

    all_tx_sessions = np.bincount(sid[is_tx], minlength=n_sessions) > 0
    ex1 = prefix_n < min_prefix
    ex2 = ~ex1 & (prefix_tx > 0)
    ex3 = ~ex1 & ~ex2 & (after_n == 0)
    eligible = ~(ex1 | ex2 | ex3)
    target = (after_tx > 0)

    report = {
        "policy": policy, "n_events": n_events, "cutoff_minutes": cutoff_minutes,
        "total_sessions": n_sessions,
        "excluded_insufficient_prefix": int(ex1.sum()),
        "excluded_purchase_before_or_at_snapshot": int(ex2.sum()),
        "excluded_no_future_window": int(ex3.sum()),
        "eligible_samples": int(eligible.sum()),
        "positives": int((eligible & target).sum()),
        "negatives": int((eligible & ~target).sum()),
        "purchase_sessions_total": int(all_tx_sessions.sum()),
        "purchase_sessions_excluded_insufficient_prefix": int((ex1 & all_tx_sessions).sum()),
        "purchase_sessions_excluded_purchase_before_snapshot": int((ex2 & all_tx_sessions).sum()),
        "purchase_sessions_excluded_no_future_window": int((ex3 & all_tx_sessions).sum()),
    }
    report["positive_rate"] = report["positives"] / report["eligible_samples"] if report["eligible_samples"] else 0.0

    keep = in_prefix & eligible[sid]
    prefix_events = events.loc[keep]
    feats = aggregate_event_features(prefix_events)
    # start_ts from the aggregate is the first PREFIX event == session start (prefix begins at event 0).
    feats = feats.drop(columns=["transaction_count", "end_ts", "unique_items_total"], errors="ignore")
    feats["visitorid"] = prefix_events.groupby("session_id", sort=True)["visitorid"].first().reindex(feats.index).to_numpy()
    feats["snapshot_ts"] = snap_ts[feats.index.to_numpy()]
    start = pd.to_datetime(feats["start_ts"], unit="ms")
    feats["start_hour"] = start.dt.hour.astype("int8")
    feats["start_dayofweek"] = start.dt.day_of_week.astype("int8")
    feats[TARGET] = target[feats.index.to_numpy()].astype("int8")
    if include_history:
        feats = add_history_features(feats, sessions)
    feats = feats.sort_values("snapshot_ts", kind="stable")  # chronological order for splitting
    return SnapshotResult(feats, report, {"policy": policy, "n_events": n_events,
                                          "cutoff_minutes": cutoff_minutes, "include_history": include_history})


def available_snapshot_features(snap: pd.DataFrame) -> list[str]:
    all_f = SNAPSHOT_BASE_FEATURES + SNAPSHOT_CATEGORY_FEATURES + SNAPSHOT_HISTORY_FEATURES
    return [f for f in all_f if f in snap.columns]


# =============================================================================
# 4. CUSTOMER-LEVEL (VISITOR) BEHAVIOURAL PROFILES
#    - Groups all sessions by visitorid to produce lifetime shopping profiles
#      for Customer Segmentation (K-Means).
#    - Supports optional `as_of_ts` cutoff if historical-only profiles are needed.
#    - Note: Retailrocket has no removefromcart event, so cart abandonment is
#      measured as (cart sessions without purchase / cart sessions).
# =============================================================================
CUSTOMER_FEATURES = [
    "total_sessions", "total_views", "unique_items_viewed", "total_add_to_cart", "total_transactions",
    "purchase_frequency", "average_session_duration", "average_products_per_session",
    "average_cart_additions_per_session", "days_since_last_activity", "average_time_between_sessions_hours",
    "historical_cart_abandonment_rate",
]


def build_customer_features(events: pd.DataFrame, sessions: pd.DataFrame, as_of_ts: int | None = None) -> pd.DataFrame:
    """One row per visitorid.

    `as_of_ts` (epoch ms) restricts to sessions that ENDED at or before that time, giving a
    historical window suitable for predictive use. With as_of_ts=None the full history is used,
    which is fine for descriptive segmentation but must not be used to predict past outcomes.

    NOTE: the dataset has no remove-from-cart event, so total_cart_removals cannot be computed."""
    s = sessions
    ev = events
    if as_of_ts is not None:
        s = s[s["end_ts"] <= as_of_ts]
        ev = ev[ev["timestamp"] <= as_of_ts]
    reference = as_of_ts if as_of_ts is not None else int(ev["timestamp"].max())
    s = s.assign(cart_session=(s["add_to_cart_count"] > 0).astype("int32"),
                 abandoned=((s["add_to_cart_count"] > 0) & (s["transaction_count"] == 0)).astype("int32"),
                 purchase_session=(s["transaction_count"] > 0).astype("int32"))
    g = s.groupby("visitorid", sort=True)
    cust = g.agg(
        total_sessions=("start_ts", "size"), total_events=("total_events", "sum"),
        total_views=("total_views", "sum"), total_add_to_cart=("add_to_cart_count", "sum"),
        total_transactions=("transaction_count", "sum"), purchase_sessions=("purchase_session", "sum"),
        cart_sessions=("cart_session", "sum"), abandoned_sessions=("abandoned", "sum"),
        average_session_duration=("session_duration_sec", "mean"),
        average_products_per_session=("unique_items_viewed", "mean"),
        first_session_start=("start_ts", "min"), last_session_start=("start_ts", "max"),
        last_activity_ts=("end_ts", "max"))
    v = ev[ev["event"] == "view"]
    uniq = v.drop_duplicates(["visitorid", "itemid"]).groupby("visitorid").size()
    cust["unique_items_viewed"] = uniq.reindex(cust.index, fill_value=0).astype("int32")
    cust["purchase_frequency"] = cust["purchase_sessions"] / cust["total_sessions"]
    cust["average_cart_additions_per_session"] = cust["total_add_to_cart"] / cust["total_sessions"]
    cust["days_since_last_activity"] = (reference - cust["last_activity_ts"]) / 86_400_000.0
    gaps = np.where(cust["total_sessions"] > 1,
                    (cust["last_session_start"] - cust["first_session_start"]) / (cust["total_sessions"] - 1).clip(lower=1)
                    / 3_600_000.0, np.nan)
    cust["average_time_between_sessions_hours"] = gaps  # start-to-start; NaN for single-session visitors
    cust["historical_cart_abandonment_rate"] = np.where(
        cust["cart_sessions"] > 0, cust["abandoned_sessions"] / cust["cart_sessions"].clip(lower=1), 0.0)
    cust["has_purchase"] = (cust["total_transactions"] > 0).astype("int8")
    return cust
