"""Pure analytics functions for Dashboard / EDA. Every number is computed from the (filtered) events."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings


def filter_events(events: pd.DataFrame, start=None, end=None, event_types=None, visitor_ids=None,
                  category_ids=None) -> pd.DataFrame:
    """Apply dashboard filters with one boolean mask (no intermediate copies)."""
    mask = np.ones(len(events), dtype=bool)
    if start is not None:
        mask &= events["timestamp"].to_numpy() >= int(pd.Timestamp(start).timestamp() * 1000)
    if end is not None:  # inclusive end date
        mask &= events["timestamp"].to_numpy() < int((pd.Timestamp(end) + pd.Timedelta(days=1)).timestamp() * 1000)
    if event_types:
        mask &= events["event"].isin(event_types).to_numpy()
    if visitor_ids:
        mask &= events["visitorid"].isin(list(visitor_ids)).to_numpy()
    if category_ids and "categoryid" in events.columns:
        mask &= events["categoryid"].isin(list(category_ids)).to_numpy()
    return events.loc[mask]


def _dt(ts):
    return pd.to_datetime(ts, unit="ms")


def session_stats(ev: pd.DataFrame) -> pd.DataFrame:
    """Per-session stats restricted to the given events."""
    g = ev.groupby("session_id", sort=False)
    out = g.agg(events=("timestamp", "size"), start=("timestamp", "min"), end=("timestamp", "max"),
                visitorid=("visitorid", "first"))
    out["duration_sec"] = (out["end"] - out["start"]) / 1000.0
    cnt = pd.crosstab(ev["session_id"], ev["event"]).reindex(out.index, fill_value=0)
    for e in settings.VALID_EVENTS:
        out[e] = cnt[e] if e in cnt.columns else 0
    return out


def compute_kpis(ev: pd.DataFrame) -> dict:
    if ev.empty:
        return {k: 0 for k in ["total_events", "unique_visitors", "unique_sessions", "views", "addtocart", "transactions",
                               "purchase_sessions", "session_purchase_rate", "avg_session_duration_sec",
                               "cart_to_purchase_event_ratio"]}
    ss = session_stats(ev)
    counts = ev["event"].value_counts()
    views, carts, tx = int(counts.get("view", 0)), int(counts.get("addtocart", 0)), int(counts.get("transaction", 0))
    purchase_sessions = int((ss["transaction"] > 0).sum())
    return {
        "total_events": len(ev), "unique_visitors": int(ev["visitorid"].nunique()), "unique_sessions": len(ss),
        "views": views, "addtocart": carts, "transactions": tx, "purchase_sessions": purchase_sessions,
        "session_purchase_rate": purchase_sessions / len(ss) * 100 if len(ss) else 0.0,
        "avg_session_duration_sec": float(ss["duration_sec"].mean()),
        "cart_to_purchase_event_ratio": tx / carts * 100 if carts else 0.0,
    }


def event_distribution(ev: pd.DataFrame) -> pd.DataFrame:
    return ev["event"].value_counts().rename_axis("event").reset_index(name="count")


def time_series(ev: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Event counts per period and event type. freq: 'D' daily, 'W' weekly, 'h' hourly."""
    if ev.empty:
        return pd.DataFrame(columns=["period", "event", "count"])
    d = pd.DataFrame({"period": _dt(ev["timestamp"]).dt.to_period(freq).dt.start_time, "event": ev["event"].to_numpy()})
    return d.groupby(["period", "event"], observed=True).size().reset_index(name="count")


def funnel(ev: pd.DataFrame, level: str = "visitors") -> pd.DataFrame:
    """view -> addtocart -> transaction at event, session or visitor level (distinct entities having the event)."""
    rows = []
    for e in settings.VALID_EVENTS:
        sub = ev[ev["event"] == e]
        n = len(sub) if level == "events" else sub["session_id"].nunique() if level == "sessions" else sub["visitorid"].nunique()
        rows.append({"stage": e, "count": int(n)})
    f = pd.DataFrame(rows)
    f["pct_of_first"] = f["count"] / f["count"].iloc[0] * 100 if f["count"].iloc[0] else 0.0
    f["pct_of_previous"] = (f["count"] / f["count"].shift(1) * 100).fillna(100.0)
    return f


def top_items(ev: pd.DataFrame, n: int = 15, sort_by: str = "view") -> pd.DataFrame:
    ct = pd.crosstab(ev["itemid"], ev["event"])
    for e in settings.VALID_EVENTS:
        if e not in ct.columns:
            ct[e] = 0
    ct = ct[settings.VALID_EVENTS].sort_values(sort_by, ascending=False).head(n)
    ct["total"] = ct.sum(axis=1)
    return ct.reset_index()


def visitor_activity_distribution(ev: pd.DataFrame) -> pd.Series:
    return ev.groupby("visitorid").size()


def hist_counts(values: pd.Series, bins) -> pd.DataFrame:
    cut = pd.cut(values, bins=bins, right=True, include_lowest=True)
    out = cut.value_counts(sort=False).rename_axis("bin").reset_index(name="count")
    out["bin"] = out["bin"].astype(str)
    return out


def event_transitions(ev: pd.DataFrame) -> pd.DataFrame:
    """Within-session next-event transition counts (a session's last event has no successor)."""
    sid = ev["session_id"].to_numpy()
    cur = ev["event"].astype(str).to_numpy()
    nxt = np.roll(cur, -1)
    same = np.roll(sid, -1) == sid
    same[-1] = False
    t = pd.DataFrame({"from": cur[same], "to": nxt[same]})
    return pd.crosstab(t["from"], t["to"]).reindex(index=settings.VALID_EVENTS, columns=settings.VALID_EVENTS, fill_value=0)


def cart_abandonment(sessions: pd.DataFrame) -> dict:
    cart = sessions["add_to_cart_count"] > 0
    bought = sessions["transaction_count"] > 0
    n_cart = int(cart.sum())
    return {"sessions_with_cart": n_cart, "cart_and_purchase": int((cart & bought).sum()),
            "abandoned": int((cart & ~bought).sum()),
            "abandonment_rate": float((cart & ~bought).sum() / n_cart * 100) if n_cart else 0.0,
            "purchase_without_cart": int((~cart & bought).sum())}


def interactions_before_purchase(ev: pd.DataFrame) -> pd.DataFrame:
    """For each purchase session: number of events before its first transaction and views among them."""
    tx = ev[ev["event"] == "transaction"].groupby("session_id")["event_pos"].min().rename("first_tx_pos")
    if tx.empty:
        return pd.DataFrame(columns=["session_id", "events_before_purchase", "views_before_purchase"])
    e = ev[ev["session_id"].isin(tx.index)].join(tx, on="session_id")
    e = e[e["event_pos"] < e["first_tx_pos"]]
    out = pd.DataFrame({"events_before_purchase": e.groupby("session_id").size(),
                        "views_before_purchase": e[e["event"] == "view"].groupby("session_id").size()})
    out = out.reindex(tx.index, fill_value=0).reset_index()
    return out


def item_conversion(ev: pd.DataFrame, min_carts: int = 5, n: int = 20) -> pd.DataFrame:
    ct = pd.crosstab(ev["itemid"], ev["event"])
    for e in settings.VALID_EVENTS:
        if e not in ct.columns:
            ct[e] = 0
    ct = ct[ct["addtocart"] >= min_carts].copy()
    ct["transactions_per_100_carts"] = ct["transaction"] / ct["addtocart"] * 100
    return ct.sort_values("transactions_per_100_carts", ascending=False).head(n).reset_index()


def category_roots(tree: pd.DataFrame) -> pd.DataFrame:
    """Root ancestor and depth for each category (iterative parent walk, cycle-safe)."""
    parent = dict(zip(tree["categoryid"].astype(int), tree["parentid"].astype("float")))
    root, depth = {}, {}
    for c in parent:
        cur, d, seen = c, 0, set()
        while not np.isnan(parent.get(cur, np.nan)) and cur not in seen:
            seen.add(cur)
            cur = int(parent[cur])
            d += 1
            if cur not in parent:
                break
        root[c], depth[c] = cur, d
    return pd.DataFrame({"categoryid": list(root), "root": list(root.values()), "depth": list(depth.values())})


def category_performance(ev: pd.DataFrame, tree: pd.DataFrame | None, n: int = 20) -> pd.DataFrame:
    sub = ev[ev["categoryid"].notna()]
    ct = pd.crosstab(sub["categoryid"].astype(int), sub["event"])
    for e in settings.VALID_EVENTS:
        if e not in ct.columns:
            ct[e] = 0
    ct = ct[settings.VALID_EVENTS]
    ct["total"] = ct.sum(axis=1)
    ct["transactions_per_100_carts"] = np.where(ct["addtocart"] > 0, ct["transaction"] / ct["addtocart"].clip(lower=1) * 100, np.nan)
    ct = ct.sort_values("total", ascending=False).head(n).reset_index().rename(columns={"categoryid": "categoryid"})
    if tree is not None:
        ct = ct.merge(tree.rename(columns={"parentid": "parent"})[["categoryid", "parent"]], on="categoryid", how="left")
    return ct
