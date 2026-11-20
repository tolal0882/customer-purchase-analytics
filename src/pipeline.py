"""Orchestrates raw -> cleaned events -> sessions -> session/customer tables, with an on-disk
cache in data/processed/ (parquet). Pure Python (no Streamlit); the UI wraps these calls in
st.cache_data. Raw files are only read."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from config import settings
from src import data_loader, data_validation, feature_engineering as fe, preprocessing, session_engineering
from src.utilities import atomic_write_json, file_signature, read_json, short_hash


@dataclass
class CoreTables:
    events: pd.DataFrame        # cleaned events + session_id + event_pos
    sessions: pd.DataFrame      # one row per session (retrospective features)
    customers: pd.DataFrame     # one row per visitor
    cleaning_steps: pd.DataFrame
    quality: dict
    meta: dict = field(default_factory=dict)


def dataset_version(timeout_minutes: float, remove_duplicates: bool) -> str:
    sig = file_signature(data_loader.events_path())
    return short_hash(sig, timeout_minutes, remove_duplicates)


def _paths(version: str):
    d = settings.PROCESSED_DIR
    return (d / f"events_{version}.parquet", d / f"sessions_{version}.parquet",
            d / f"customers_{version}.parquet", d / f"meta_{version}.json")


def build_core_tables(timeout_minutes: float = settings.DEFAULT_SESSION_TIMEOUT_MIN,
                      remove_duplicates: bool = settings.DEFAULT_REMOVE_DUPLICATES,
                      force: bool = False) -> CoreTables:
    """Build (or load from cache) all core tables for the given processing options."""
    settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    version = dataset_version(timeout_minutes, remove_duplicates)
    pe, ps, pc, pm = _paths(version)
    if not force and all(p.exists() for p in (pe, ps, pc, pm)):
        meta = read_json(pm)
        steps = pd.DataFrame(meta.pop("cleaning_steps"))
        quality = meta.pop("quality")
        return CoreTables(pd.read_parquet(pe), pd.read_parquet(ps), pd.read_parquet(pc), steps, quality, meta)

    t0 = time.time()
    raw = data_loader.read_events_raw()
    schema = data_validation.validate_events_schema(raw)
    if not schema.ok:
        raise ValueError("; ".join(schema.messages))
    quality = data_validation.data_quality_report(raw)
    cleaned = preprocessing.clean_events(raw, remove_duplicates=remove_duplicates)
    del raw
    events = session_engineering.assign_sessions(cleaned.events, timeout_minutes)
    sessions = fe.build_session_table(events)
    customers = fe.build_customer_features(events, sessions)
    meta = {
        "dataset_version": version, "session_timeout_minutes": timeout_minutes,
        "remove_duplicates": remove_duplicates,
        "rows_raw": quality["rows"], "rows_clean": len(events), "n_sessions": len(sessions),
        "n_visitors": len(customers), "build_seconds": round(time.time() - t0, 1),
        "built_at": pd.Timestamp.now().isoformat(),
    }
    events.to_parquet(pe, index=False)
    sessions.to_parquet(ps)
    customers.to_parquet(pc)
    atomic_write_json(pm, {**meta, "cleaning_steps": cleaned.steps.to_dict("records"), "quality": quality})
    return CoreTables(events, sessions, customers, cleaned.steps, quality, meta)
