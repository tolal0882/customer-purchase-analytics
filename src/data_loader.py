"""Dataset discovery and memory-efficient loading.

Rule: Original raw CSV files are strictly READ-ONLY and never modified.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import settings


# =============================================================================
# 1. FILE METADATA & SETUP INSTRUCTIONS
# =============================================================================
@dataclass(frozen=True)
class FileInfo:
    """Describes one expected dataset file on disk (existence and byte size)."""
    key: str
    filename: str
    path: Path
    exists: bool
    size_bytes: int


class DatasetNotFoundError(FileNotFoundError):
    """Raised with clear setup instructions when events.csv is missing."""


SETUP_INSTRUCTION = (
    "Download the Retailrocket dataset from "
    "https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset and place "
    "events.csv into `data/raw/`. Alternatively upload events.csv on "
    "the Data Preprocessing page."
)


# =============================================================================
# 2. FILE DISCOVERY
# =============================================================================
def discover_files(raw_dir: Path | None = None) -> dict[str, FileInfo]:
    """Check whether events.csv is available locally."""
    raw_dir = Path(raw_dir) if raw_dir else settings.resolve_raw_dir()
    out = {}
    for key, fname in settings.DATASET_FILES.items():
        p = raw_dir / fname
        out[key] = FileInfo(key, fname, p, p.exists(), p.stat().st_size if p.exists() else 0)
    return out


def events_path(raw_dir: Path | None = None) -> Path:
    """Return the path to events.csv, or raise DatasetNotFoundError if absent."""
    info = discover_files(raw_dir)["events"]
    if not info.exists:
        raise DatasetNotFoundError(f"events.csv not found at {info.path}. {SETUP_INSTRUCTION}")
    return info.path


# =============================================================================
# 3. CORE CSV LOADER (events.csv)
# =============================================================================
def read_events_raw(path: Path | str | None = None, nrows: int | None = None) -> pd.DataFrame:
    """Read events.csv exactly as stored (no silent cleaning or type changes).

    Validation, dtype optimization, and cleaning happen in separate logged steps
    (see src/data_validation.py and src/preprocessing.py).
    """
    p = Path(path) if path else events_path()
    return pd.read_csv(p, nrows=nrows)
