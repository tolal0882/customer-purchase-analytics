"""Leakage tests for prediction snapshots."""
import numpy as np
import pandas as pd
import pytest

from src import feature_engineering as fe, session_engineering as se
from tests.conftest import make_events


def snaps(rows, **kw):
    ev = se.assign_sessions(make_events(rows), 30)
    s = fe.build_session_table(ev)
    return ev, s, fe.build_prediction_snapshots(ev, s, **kw)


BASE = [
    # session A (visitor 1): 3 views then add to cart then purchase -> positive
    (0, 1, "view", 1), (1, 1, "view", 2), (2, 1, "view", 3), (3, 1, "addtocart", 3), (4, 1, "transaction", 3, 9),
    # session B (visitor 2): 3 views, 2 more views -> negative
    (0, 2, "view", 1), (1, 2, "view", 2), (2, 2, "view", 3), (3, 2, "view", 4),
    # session C (visitor 3): purchase at event 2 (before snapshot) -> excluded
    (0, 3, "view", 1), (1, 3, "transaction", 1, 10), (2, 3, "view", 2), (3, 3, "view", 3), (4, 3, "view", 4),
    # session D (visitor 4): 2 events only -> excluded
    (0, 4, "view", 1), (1, 4, "view", 2),
    # session E (visitor 5): exactly 3 events, nothing after -> excluded (no window)
    (0, 5, "view", 1), (1, 5, "view", 2), (2, 5, "view", 3),
]


def test_eligibility_and_target():
    ev, s, r = snaps(BASE)
    rep = r.report
    assert rep["eligible_samples"] == 2 and rep["positives"] == 1 and rep["negatives"] == 1
    assert rep["excluded_insufficient_prefix"] == 1 and rep["excluded_purchase_before_or_at_snapshot"] == 1
    assert rep["excluded_no_future_window"] == 1
    vis = r.data.set_index("visitorid")["target"].to_dict()
    assert vis == {1: 1, 2: 0}


def test_features_ignore_post_snapshot_events():
    """Features of a session must be identical whether or not later events exist."""
    ev, s, r = snaps(BASE)
    fa = r.data[r.data.visitorid == 1].iloc[0]
    assert fa.total_events == 3 and fa.add_to_cart_count == 0 and fa.total_views == 3  # cart came AFTER snapshot
    assert fa.first_cart_position == -1 and fa.ends_with_cart == 0


def test_mutating_future_events_does_not_change_features():
    rows2 = [r for r in BASE]
    rows2[3] = (3, 1, "view", 99)           # change a post-snapshot event of session A
    rows2[4] = (40.0, 1, "view", 98)        # and delay the last event (still a different session!) -> adjust:
    rows2[4] = (4, 1, "view", 98)           # keep in-session; replace transaction by a view
    _, _, r1 = snaps(BASE)
    _, _, r2 = snaps(rows2)
    cols = [c for c in fe.SNAPSHOT_BASE_FEATURES if c in r1.data.columns]
    a = r1.data[r1.data.visitorid == 1][cols].reset_index(drop=True)
    b = r2.data[r2.data.visitorid == 1][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    assert r2.data[r2.data.visitorid == 1].iloc[0].target == 0  # label changes, features do not


def test_no_forbidden_columns_in_feature_list():
    _, _, r = snaps(BASE)
    feats = fe.available_snapshot_features(r.data)
    fe.assert_no_leakage_features(feats)
    assert not (set(feats) & fe.FORBIDDEN_FEATURES)
    with pytest.raises(ValueError):
        fe.assert_no_leakage_features(["total_events", "transaction_count"])


def test_time_policy():
    ev, s, r = snaps(BASE, policy="time", cutoff_minutes=2.0)
    rep = r.report
    # sessions A,B,E have events after minute 2 / C purchases at minute 1 (<= snapshot) -> excluded
    assert rep["excluded_purchase_before_or_at_snapshot"] == 1
    d = r.data.set_index("visitorid")
    assert d.loc[1, "target"] == 1 and d.loc[2, "target"] == 0
    assert (d["total_events"] <= 3).all()


def test_history_uses_only_earlier_sessions():
    rows = [(0, 1, "view", 1), (1, 1, "view", 2), (2, 1, "transaction", 2, 1),
            (100, 1, "view", 1), (101, 1, "view", 2), (102, 1, "view", 3), (103, 1, "view", 4),
            (200, 1, "view", 1), (201, 1, "view", 2), (202, 1, "view", 3), (203, 1, "addtocart", 3), (204, 1, "transaction", 3, 2)]
    ev, s, r = snaps(rows)
    d = r.data
    assert list(d["hist_prior_purchases"]) == [1, 1]   # second session sees only the first session's purchase
    assert list(d["hist_prior_sessions"]) == [1, 2]
    assert list(d["target"]) == [0, 1]


def test_chronological_order():
    _, _, r = snaps(BASE)
    assert r.data["snapshot_ts"].is_monotonic_increasing
