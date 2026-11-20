import numpy as np
import pytest
import pandas as pd

from src import feature_engineering as fe, preprocessing, session_engineering as se
from tests.conftest import make_events


def test_session_split_on_gap(toy_events):
    ev = se.assign_sessions(toy_events, 30)
    # visitor 1 -> 2 sessions, visitor 2 -> 1, visitor 3 -> 1
    assert ev["session_id"].nunique() == 4
    v1 = ev[ev.visitorid == 1]
    assert v1["session_id"].nunique() == 2
    assert list(v1[v1.session_id == v1.session_id.min()]["event_pos"]) == [0, 1, 2, 3, 4]


def test_gap_exactly_timeout_does_not_split():
    ev = make_events([(0, 1, "view", 1), (30, 1, "view", 2), (60.001, 1, "view", 3)])
    out = se.assign_sessions(ev, 30)
    assert out["session_id"].nunique() == 2  # 30 min gap stays; >30 splits


def test_sessions_never_cross_visitors(toy_events):
    ev = se.assign_sessions(toy_events, 30)
    assert (ev.groupby("session_id")["visitorid"].nunique() == 1).all()


def test_timeout_configurable(toy_events):
    a = se.assign_sessions(toy_events, 30)["session_id"].nunique()
    b = se.assign_sessions(toy_events, 1000)["session_id"].nunique()
    assert b < a


def test_unsorted_input_is_sorted():
    ev = make_events([(5, 1, "view", 1), (0, 1, "view", 2)])
    shuffled = ev.iloc[::-1].reset_index(drop=True)
    out = se.assign_sessions(shuffled, 30)
    assert out["timestamp"].is_monotonic_increasing


def test_cleaning_logs_duplicates_and_keeps_raw_untouched():
    raw = pd.DataFrame({"timestamp": [1_500_000_000_000] * 2 + [1_500_000_060_000], "visitorid": [1, 1, 1],
                        "event": ["view", "view", "VIEW "], "itemid": [1, 1, 2], "transactionid": [None] * 3})
    before = raw.copy()
    res = preprocessing.clean_events(raw)
    pd.testing.assert_frame_equal(raw, before)  # input not mutated
    assert len(res.events) == 2
    row = res.steps[res.steps.step.str.startswith("Remove duplicate")].iloc[0]
    assert row.rows_affected == 1 and len(res.removed) == 1
    kept = preprocessing.clean_events(raw, remove_duplicates=False)
    assert len(kept.events) == 3


def test_invalid_rows_reported():
    raw = pd.DataFrame({"timestamp": [1_500_000_000_000, -5], "visitorid": [1, 2], "event": ["view", "bogus"],
                        "itemid": [1, 2], "transactionid": [None, None]})
    res = preprocessing.clean_events(raw)
    assert len(res.events) == 1
    assert res.steps[res.steps.step == "Invalid timestamps"].iloc[0].rows_affected == 1
    assert res.steps[res.steps.step == "Unknown event type"].iloc[0].rows_affected == 1


def test_session_table_counts(toy_events):
    ev = se.assign_sessions(toy_events, 30)
    s = fe.build_session_table(ev)
    first = s[s.visitorid == 1].iloc[0]
    assert first.total_events == 5 and first.total_views == 3 and first.add_to_cart_count == 1
    assert first.transaction_count == 1 and first.unique_items_viewed == 3
    assert first.session_duration_sec == 240 and first.avg_time_between_events_sec == 60
    assert first.cart_interaction_ratio == pytest.approx(0.2)


def test_customer_features(toy_events):
    ev = se.assign_sessions(toy_events, 30)
    s = fe.build_session_table(ev)
    c = fe.build_customer_features(ev, s)
    v1 = c.loc[1]
    assert v1.total_sessions == 2 and v1.total_transactions == 1 and v1.purchase_frequency == 0.5
    assert 0 <= v1.historical_cart_abandonment_rate <= 1
    assert np.isnan(c.loc[3].average_time_between_sessions_hours)
