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
    "events.csv (required) and optionally category_tree.csv, item_properties_part1.csv "
    "and item_properties_part2.csv into `data/raw/`. Alternatively upload events.csv on "
    "the Data Preprocessing page."
)


# =============================================================================
# 2. FILE DISCOVERY
# =============================================================================
def discover_files(raw_dir: Path | None = None) -> dict[str, FileInfo]:
    """Check which of the 4 Retailrocket CSV files are available locally."""
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
# 3. CORE CSV LOADERS (events.csv & category_tree.csv)
# =============================================================================
def read_events_raw(path: Path | str | None = None, nrows: int | None = None) -> pd.DataFrame:
    """Read events.csv exactly as stored (no silent cleaning or type changes).

    Validation, dtype optimization, and cleaning happen in separate logged steps
    (see src/data_validation.py and src/preprocessing.py).
    """
    p = Path(path) if path else events_path()
    return pd.read_csv(p, nrows=nrows)


def read_category_tree(path: Path | str | None = None) -> pd.DataFrame | None:
    """Read optional category_tree.csv using compact int32/Int32 dtypes."""
    p = Path(path) if path else discover_files()["category_tree"].path
    if not Path(p).exists():
        return None
    df = pd.read_csv(p)
    df["categoryid"] = df["categoryid"].astype("int32")
    df["parentid"] = df["parentid"].astype("Int32")  # Nullable int: root categories have NaN parent
    return df


# =============================================================================
# 4. CHUNKED ENRICHMENT LOADER (item_properties_part1/2.csv ~ 900 MB total)
#    - Reads 1M rows at a time so peak RAM stays small on a 16 GB laptop.
#    - Keeps ONLY property == 'categoryid' rows (~788k rows instead of ~20M).
#    - Caches the filtered result as Parquet in data/processed/ for fast reloads.
# =============================================================================
def build_item_category_history(raw_dir: Path | None = None, chunksize: int = 1_000_000) -> pd.DataFrame | None:
    """Stream item_properties CSVs in chunks and extract [itemid, timestamp, categoryid]."""
    files = discover_files(raw_dir)
    parts = [files[k] for k in ("item_properties_part1", "item_properties_part2") if files[k].exists]
    if not parts:
        return None
    pieces = []
    for info in parts:
        for chunk in pd.read_csv(info.path, chunksize=chunksize, dtype={"property": "string", "value": "string"}):
            sub = chunk.loc[chunk["property"] == "categoryid", ["itemid", "timestamp", "value"]]
            if len(sub):
                pieces.append(sub)
    if not pieces:
        return None
    df = pd.concat(pieces, ignore_index=True)
    df["categoryid"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["categoryid"]).drop(columns="value")
    df["categoryid"] = df["categoryid"].astype("int32")
    df["itemid"] = df["itemid"].astype("int32")
    df["timestamp"] = df["timestamp"].astype("int64")
    # Sort by timestamp so merge_asof() can perform leakage-free as-of category joins
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


def load_item_category_history(use_cache: bool = True) -> pd.DataFrame | None:
    """Load cached item-category history from Parquet, or build and cache it once."""
    cache = settings.PROCESSED_DIR / "item_category_history.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    df = build_item_category_history()
    if df is not None:
        settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    return df
