"""Logged, non-silent cleaning of the raw events table.

Every operation records how many rows it touched so the Preprocessing page can show
the effect of each step. Original raw files are never modified: cleaning works on an
in-memory copy and writes separate files into data/processed/."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import settings


@dataclass
class CleaningResult:
    events: pd.DataFrame                      # cleaned events (sorted by visitorid, timestamp)
    steps: pd.DataFrame                       # one row per operation with before/after counts
    removed: pd.DataFrame                     # audit trail of removed rows (capped)
    options: dict = field(default_factory=dict)


def _step(log: list, name: str, before: int, after: int, affected: int, action: str, note: str = "") -> None:
    log.append({"step": name, "rows_before": before, "rows_after": after,
                "rows_affected": affected, "action": action, "note": note})


def clean_events(raw: pd.DataFrame, remove_duplicates: bool = settings.DEFAULT_REMOVE_DUPLICATES,
                 drop_invalid: bool = True, removed_cap: int = 5000) -> CleaningResult:
    """Validate, coerce dtypes, optionally drop invalid/duplicate rows, and sort.

    Returns the cleaned frame plus a step log. Missing `transactionid` on non-transaction
    events is expected and is NOT treated as an error."""
    log: list[dict] = []
    removed_parts: list[pd.DataFrame] = []
    df = raw[settings.EVENTS_COLUMNS].copy()
    n0 = len(df)

    # 1) Type coercion (non-numeric values become NaN and are reported, never hidden).
    for c in ("timestamp", "visitorid", "itemid", "transactionid"):
        if not pd.api.types.is_numeric_dtype(df[c]):
            before_na = int(df[c].isna().sum())
            df[c] = pd.to_numeric(df[c], errors="coerce")
            _step(log, f"Coerce `{c}` to numeric", len(df), len(df), int(df[c].isna().sum()) - before_na,
                  "converted", "values that could not be parsed became missing")
    df["event"] = df["event"].astype("string").str.strip().str.lower()

    # 2) Invalid rows: bad timestamp / ids / unknown event names.
    bad_ts = df["timestamp"].isna() | (df["timestamp"] < settings.TS_MIN_MS) | (df["timestamp"] > settings.TS_MAX_MS)
    bad_ids = df["visitorid"].isna() | df["itemid"].isna()
    bad_event = ~df["event"].isin(settings.VALID_EVENTS)
    for name, mask in (("Invalid timestamps", bad_ts), ("Missing visitorid/itemid", bad_ids),
                       ("Unknown event type", bad_event)):
        _step(log, name, len(df), len(df), int(mask.sum()), "flagged", "counted before removal")
    invalid = bad_ts | bad_ids | bad_event
    if drop_invalid and invalid.any():
        removed_parts.append(df[invalid].head(removed_cap).assign(reason="invalid row"))
        n = len(df)
        df = df[~invalid]
        _step(log, "Remove invalid rows", n, len(df), int(invalid.sum()), "removed")
    elif not invalid.any():
        _step(log, "Remove invalid rows", len(df), len(df), 0, "none needed", "no invalid rows found")

    # 3) Transaction-id consistency (informational: we keep these rows and report).
    is_tx = df["event"] == "transaction"
    _step(log, "Transactions without transactionid", len(df), len(df), int((is_tx & df["transactionid"].isna()).sum()),
          "flagged", "kept; transactionid is not used as a feature")
    _step(log, "Non-transactions carrying a transactionid", len(df), len(df),
          int((~is_tx & df["transactionid"].notna()).sum()), "flagged", "kept")

    # 4) Exact duplicate rows.
    dup = df.duplicated(keep="first")
    _step(log, "Exact duplicate rows detected", len(df), len(df), int(dup.sum()), "flagged",
          "identical timestamp, visitor, event, item, transactionid")
    if remove_duplicates and dup.any():
        removed_parts.append(df[dup].head(removed_cap).assign(reason="duplicate"))
        n = len(df)
        df = df[~dup]
        _step(log, "Remove duplicate rows (keep first)", n, len(df), int(dup.sum()), "removed")
    elif not remove_duplicates:
        _step(log, "Remove duplicate rows (keep first)", len(df), len(df), 0, "skipped by user")

    # 5) Memory-efficient dtypes.
    mem_before = df.memory_usage(deep=True).sum() / 1e6
    df["timestamp"] = df["timestamp"].astype("int64")
    df["visitorid"] = df["visitorid"].astype("int32")
    df["itemid"] = df["itemid"].astype("int32")
    df["transactionid"] = df["transactionid"].round().astype("Int32")
    df["event"] = pd.Categorical(df["event"], categories=settings.VALID_EVENTS)
    mem_after = df.memory_usage(deep=True).sum() / 1e6
    _step(log, "Optimise dtypes", len(df), len(df), len(df), "converted",
          f"memory {mem_before:.0f} MB -> {mem_after:.0f} MB")

    # 6) Sort (stable, so same-timestamp events keep their file order).
    was_sorted = bool(df[["visitorid", "timestamp"]].equals(df[["visitorid", "timestamp"]].sort_values(
        ["visitorid", "timestamp"], kind="stable")))
    df = df.sort_values(["visitorid", "timestamp"], kind="stable").reset_index(drop=True)
    _step(log, "Sort by visitorid, timestamp", len(df), len(df), 0 if was_sorted else len(df), "sorted",
          "required for gap-based session construction")

    steps = pd.DataFrame(log)
    removed = pd.concat(removed_parts, ignore_index=True) if removed_parts else pd.DataFrame(
        columns=settings.EVENTS_COLUMNS + ["reason"])
    steps.attrs["rows_raw"] = n0
    steps.attrs["rows_clean"] = len(df)
    return CleaningResult(df, steps, removed, {"remove_duplicates": remove_duplicates, "drop_invalid": drop_invalid})


def attach_categories(events: pd.DataFrame, item_cat_hist: pd.DataFrame | None,
                      fallback_first: bool = False) -> pd.DataFrame:
    """Add `categoryid` (nullable Int32) to events using an **as-of join**: each event gets the
    item's category as known at (or before) the event time. This avoids look-ahead.

    fallback_first=True additionally assigns an item's *earliest known* category to events that
    happened before any record exists. That is acceptable for descriptive EDA but must NOT be
    used for prediction features (it looks forward in time)."""
    if item_cat_hist is None or item_cat_hist.empty:
        return events.assign(categoryid=pd.array([pd.NA] * len(events), dtype="Int32"))
    ev = events[["itemid", "timestamp"]].copy()
    ev["_row"] = np.arange(len(ev))
    ev = ev.sort_values("timestamp", kind="stable")
    hist = item_cat_hist.sort_values("timestamp", kind="stable")
    merged = pd.merge_asof(ev, hist, on="timestamp", by="itemid", direction="backward")
    cat = merged.sort_values("_row")["categoryid"].to_numpy()
    out = pd.Series(cat, dtype="Float64").astype("Int32")
    if fallback_first:
        first = hist.drop_duplicates("itemid", keep="first").set_index("itemid")["categoryid"]
        fill = events["itemid"].map(first)
        out = out.fillna(pd.Series(fill.to_numpy(), dtype="Float64").astype("Int32"))
    res = events.copy()
    res["categoryid"] = out.to_numpy()
    return res
