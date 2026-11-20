"""Integration tests on the REAL Retailrocket files (skipped if events.csv is absent)."""
import pytest

from src import data_loader, feature_engineering as fe, pipeline

pytestmark = pytest.mark.realdata
try:
    data_loader.events_path()
    HAVE = True
except FileNotFoundError:
    HAVE = False


@pytest.fixture(scope="module")
def core():
    return pipeline.build_core_tables()


@pytest.mark.skipif(not HAVE, reason="events.csv missing")
def test_real_counts(core):
    assert core.quality["rows"] == 2_756_101
    assert core.quality["event_counts"] == {"view": 2_664_312, "addtocart": 69_332, "transaction": 22_457}
    assert core.quality["duplicate_rows"] == 460
    assert len(core.events) == 2_756_101 - 460
    assert core.sessions.index.is_unique and core.events["session_id"].nunique() == len(core.sessions)
    assert (core.events.groupby("session_id")["visitorid"].nunique() == 1).all()


@pytest.mark.skipif(not HAVE, reason="events.csv missing")
def test_real_snapshots_leakage_safe(core):
    r = fe.build_prediction_snapshots(core.events, core.sessions)
    rep = r.report
    assert rep["eligible_samples"] == rep["positives"] + rep["negatives"] == len(r.data)
    assert r.data["target"].isin([0, 1]).all()
    assert (r.data["total_events"] == 3).all()
    assert not (set(fe.available_snapshot_features(r.data)) & fe.FORBIDDEN_FEATURES)
    assert r.data["snapshot_ts"].is_monotonic_increasing
    print(rep)
