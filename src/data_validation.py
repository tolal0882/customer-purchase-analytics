"""Schema validation and data-quality auditing (pure read-only checks).

Nothing in this module mutates or deletes rows; it only measures and reports issues.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from config import settings


# =============================================================================
# 1. SCHEMA VALIDATION
#    - Confirms all 5 required columns exist before any downstream processing.
# =============================================================================
@dataclass
class SchemaReport:
    ok: bool
    missing_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    dtypes: dict[str, str] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)


def validate_events_schema(df: pd.DataFrame) -> SchemaReport:
    """Verify that the raw events DataFrame contains the required Retailrocket columns."""
    cols = list(df.columns)
    missing = [c for c in settings.EVENTS_COLUMNS if c not in cols]
    extra = [c for c in cols if c not in settings.EVENTS_COLUMNS]
    msgs = []
    if missing:
        msgs.append(f"Missing required columns: {missing}. Expected {settings.EVENTS_COLUMNS}.")
    if extra:
        msgs.append(f"Unexpected extra columns (ignored): {extra}")
    return SchemaReport(not missing, missing, extra, {c: str(t) for c, t in df.dtypes.items()}, msgs)


# =============================================================================
# 2. FULL DATA-QUALITY AUDIT REPORT
#    - Measures missing counts, timestamp validity/range, event categories,
#      duplicate rows, and transactionid consistency.
# =============================================================================
def data_quality_report(df: pd.DataFrame) -> dict:
    """Compute real data-quality metrics on an events DataFrame."""
    n = len(df)
    rep: dict = {"rows": n, "columns": list(df.columns)}
    rep["missing_by_column"] = {c: int(v) for c, v in df.isna().sum().items()}

    # Check timestamp validity against plausible epoch-ms range (2000–2100)
    ts = pd.to_numeric(df["timestamp"], errors="coerce")
    bad_ts = ts.isna() | (ts < settings.TS_MIN_MS) | (ts > settings.TS_MAX_MS)
    rep["invalid_timestamps"] = int(bad_ts.sum())
    valid_ts = ts[~bad_ts]
    if len(valid_ts):
        rep["timestamp_min"] = pd.to_datetime(valid_ts.min(), unit="ms").isoformat()
        rep["timestamp_max"] = pd.to_datetime(valid_ts.max(), unit="ms").isoformat()
        rep["span_days"] = float((valid_ts.max() - valid_ts.min()) / 86_400_000)
    rep["sorted_by_timestamp"] = bool(ts.is_monotonic_increasing)

    # Check event labels against allowed categories: view, addtocart, transaction
    ev = df["event"].astype("string").str.strip().str.lower()
    rep["event_counts"] = {str(k): int(v) for k, v in ev.value_counts(dropna=False).items()}
    rep["invalid_events"] = int((~ev.isin(settings.VALID_EVENTS)).sum())

    # Check visitorid / itemid validity and cardinality
    for c in ("visitorid", "itemid"):
        num = pd.to_numeric(df[c], errors="coerce")
        rep[f"invalid_{c}"] = int(num.isna().sum())
        rep[f"unique_{c}"] = int(num.nunique())

    # Count exact duplicate rows and transactionid consistency
    rep["duplicate_rows"] = int(df.duplicated().sum())
    is_tx = ev == "transaction"
    rep["transactions_without_transactionid"] = int((is_tx & df["transactionid"].isna()).sum())
    rep["non_transactions_with_transactionid"] = int((~is_tx & df["transactionid"].notna()).sum())
    return rep


# =============================================================================
# 3. COLUMN-LEVEL SUMMARY (USED BY DATASET EXPLORER)
# =============================================================================
def describe_dataframe_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Return dtype, missing count, missing percentage, and unique count per column."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(3),
        "unique": df.nunique(),
    })
