import pandas as pd
import pytest

from src import feature_engineering as fe, preprocessing, session_engineering as se


def make_events(rows):
    """rows: (ts_minutes, visitor, event, item[, txid])"""
    recs = [{"timestamp": int(r[0] * 60_000) + 1_500_000_000_000, "visitorid": r[1], "event": r[2],
             "itemid": r[3], "transactionid": r[4] if len(r) > 4 else None} for r in rows]
    raw = pd.DataFrame(recs)
    return preprocessing.clean_events(raw).events


@pytest.fixture
def toy_events():
    return make_events([
        (0, 1, "view", 10), (1, 1, "view", 11), (2, 1, "addtocart", 11), (3, 1, "view", 12), (4, 1, "transaction", 11, 5),
        (100, 1, "view", 10),                       # >30 min gap -> new session for visitor 1
        (0, 2, "view", 20), (5, 2, "view", 21), (9, 2, "view", 22), (10, 2, "view", 23),
        (50, 3, "view", 30),
    ])
