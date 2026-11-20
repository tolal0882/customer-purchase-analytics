"""Small shared helpers (no Streamlit imports so they stay unit-testable)."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ms_to_datetime(ts_ms) -> pd.Series | pd.DatetimeIndex:
    """Epoch milliseconds -> naive UTC datetimes. All timestamps in the app use UTC."""
    return pd.to_datetime(ts_ms, unit="ms", utc=False)


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GB"


def memory_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / 1e6)


def file_signature(path: Path) -> str:
    """Cheap, stable fingerprint of a file (size + mtime) used for dataset versioning."""
    st = Path(path).stat()
    return f"{st.st_size}-{int(st.st_mtime)}"


def short_hash(*parts: Any) -> str:
    h = hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()
    return h[:10]


def atomic_write_json(path: Path, obj: Any) -> None:
    """Write JSON atomically so a crash never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=2, default=json_default)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return str(o)


def read_json(path: Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


def pct(a: float, b: float) -> float:
    return float(a) / float(b) * 100.0 if b else 0.0


def sample_df(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Random sample capped at n rows (for plotting/silhouette on large data)."""
    return df if len(df) <= n else df.sample(n=n, random_state=seed)
