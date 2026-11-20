"""Gap-based session construction (fully vectorized with NumPy for speed).

Methodological note:
    Retailrocket does NOT supply a session_id field. We engineer sessions by
    sorting each visitor's events chronologically and starting a new session
    whenever the gap between consecutive events exceeds `timeout_minutes`
    (default: 30 minutes — an analytical assumption).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings


# =============================================================================
# 1. SORT-ORDER CHECK
#    - Verifies whether events are already ordered by (visitorid, timestamp)
#      so we can skip an unnecessary sort copy.
# =============================================================================
def is_sorted(events: pd.DataFrame) -> bool:
    """Return True if events are monotonically ordered by (visitorid, timestamp)."""
    v = events["visitorid"].to_numpy()
    t = events["timestamp"].to_numpy()
    if len(v) < 2:
        return True
    same = v[1:] == v[:-1]
    return bool(np.all((v[1:] > v[:-1]) | (same & (t[1:] >= t[:-1]))))


# =============================================================================
# 2. VECTORIZED SESSION ASSIGNMENT
#    - Marks row i as a new session if:
#        (a) i == 0 (first row), OR
#        (b) visitorid[i] != visitorid[i-1] (new visitor), OR
#        (c) timestamp[i] - timestamp[i-1] > timeout_minutes * 60_000 ms
#    - Cumulative sum of those flags produces globally unique session_ids (0..S-1).
#    - event_pos gives the 0-based order of each event inside its session.
# =============================================================================
def assign_sessions(events: pd.DataFrame, timeout_minutes: float = settings.DEFAULT_SESSION_TIMEOUT_MIN) -> pd.DataFrame:
    """Assign `session_id` and `event_pos` to every event row."""
    if timeout_minutes <= 0:
        raise ValueError("timeout_minutes must be positive")
    df = events if is_sorted(events) else events.sort_values(["visitorid", "timestamp"], kind="stable").reset_index(drop=True)
    n = len(df)
    if n == 0:
        return df.assign(session_id=np.array([], dtype="int32"), event_pos=np.array([], dtype="int32"))

    v = df["visitorid"].to_numpy()
    t = df["timestamp"].to_numpy()

    # Boolean boundary array: True wherever a new session begins
    new = np.empty(n, dtype=bool)
    new[0] = True
    new[1:] = (v[1:] != v[:-1]) | ((t[1:] - t[:-1]) > timeout_minutes * 60_000)

    # Convert session boundaries into 0-based integer session IDs and in-session positions
    sid = np.cumsum(new) - 1
    start_idx = np.flatnonzero(new)
    pos = np.arange(n) - start_idx[sid]

    out = df.copy()
    out["session_id"] = sid.astype("int32")
    out["event_pos"] = pos.astype("int32")
    return out


# =============================================================================
# 3. FAST SENSITIVITY COUNTER (USED ON DATA PREPROCESSING PAGE)
#    - Counts how many sessions a candidate timeout would produce without copying data.
# =============================================================================
def count_sessions(events: pd.DataFrame, timeout_minutes: float) -> int:
    """Count total sessions for a given timeout on an already-sorted events frame."""
    v = events["visitorid"].to_numpy()
    t = events["timestamp"].to_numpy()
    if len(v) == 0:
        return 0
    return int(1 + np.count_nonzero((v[1:] != v[:-1]) | ((t[1:] - t[:-1]) > timeout_minutes * 60_000)))
