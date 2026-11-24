"""
Dataset Validation & Feature Group Definitions for the UCI Online Shoppers
Purchasing Intention Dataset (Sakar & Kastro, 2018).

This module defines:
  1. Expected 18-column schema (17 input features + 1 binary target `Revenue`)
  2. The 5 official CRISP-DM feature groups (Section 5.2)
  3. Continuous numerical features vs. nominal/categorical features
  4. Default and optional clustering features (strictly excluding `Revenue`)
  5. Complete dataset validation function (`validate_dataset`)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# =============================================================================
# 1. TARGET & SCHEMA DEFINITIONS (17 INPUT FEATURES + 1 TARGET = 18 COLUMNS)
# =============================================================================
TARGET_COL = "Revenue"

EXPECTED_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
    "Revenue",
]

# All 17 predictor columns (excluding the target column Revenue)
INPUT_FEATURES = [c for c in EXPECTED_COLUMNS if c != TARGET_COL]

# =============================================================================
# 2. FEATURE GROUPS (SECTION 5.2)
#    Organizes the 17 input features into 5 domain-meaningful behavioral groups.
# =============================================================================
FEATURE_GROUPS = {
    "Browsing Activity": ["Administrative", "Informational", "ProductRelated"],
    "Browsing Duration": ["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration"],
    "Website Interaction": ["BounceRates", "ExitRates", "PageValues", "SpecialDay"],
    "Visitor & Calendar": ["Month", "VisitorType", "Weekend"],
    "Technical & Geographic": ["OperatingSystems", "Browser", "Region", "TrafficType"],
}

# =============================================================================
# 3. NUMERICAL VS. CATEGORICAL FEATURE SEPARATION
#    - CONTINUOUS_FEATURES are scaled with StandardScaler/RobustScaler/MinMaxScaler.
#    - NOMINAL_CATEGORICAL_FEATURES (including integer-coded OS, Browser, Region,
#      TrafficType) are encoded with OneHotEncoder(handle_unknown="ignore").
# =============================================================================
CONTINUOUS_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

NOMINAL_CATEGORICAL_FEATURES = [
    "Month",
    "VisitorType",
    "Weekend",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
]

# =============================================================================
# 4. CLUSTERING FEATURE SETS (SECTION 7)
#    CRITICAL RULE: 'Revenue' is never included in default or optional clustering
#    features so K-Means discovers purely behavioral session segments.
# =============================================================================
DEFAULT_CLUSTERING_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "SpecialDay",
]

OPTIONAL_CLUSTERING_FEATURES = [
    "PageValues",
    "Month",
    "VisitorType",
    "Weekend",
]

# Chronological calendar order for the 10 months present in the UCI dataset
MONTH_ORDER = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Human-readable data dictionary descriptions for every column
FEATURE_DESCRIPTIONS = {
    "Administrative": "Number of administrative pages visited by the visitor in that session.",
    "Administrative_Duration": "Total time (seconds) spent on administrative pages during the session.",
    "Informational": "Number of informational pages visited in the session.",
    "Informational_Duration": "Total time (seconds) spent on informational pages.",
    "ProductRelated": "Number of product-related pages visited in the session.",
    "ProductRelated_Duration": "Total time (seconds) spent on product-related pages.",
    "BounceRates": "Average Google Analytics bounce rate of the pages visited in the session (0.0 to 0.2).",
    "ExitRates": "Average Google Analytics exit rate of the pages visited in the session (0.0 to 0.2).",
    "PageValues": "Average Google Analytics value of a page visited prior to an e-commerce transaction.",
    "SpecialDay": "Closeness of the site visiting date to a specific special day (e.g., Mother's Day, Valentine's Day) in [0, 1].",
    "Month": "Month of the year of the session (Feb, Mar, May, June, Jul, Aug, Sep, Oct, Nov, Dec).",
    "OperatingSystems": "Integer-coded operating system of the visitor (1 to 8).",
    "Browser": "Integer-coded web browser of the visitor (1 to 13).",
    "Region": "Integer-coded geographic region from which the session was started (1 to 9).",
    "TrafficType": "Integer-coded traffic source by which the visitor arrived at the site (1 to 20).",
    "VisitorType": "Visitor classification: Returning_Visitor, New_Visitor, or Other.",
    "Weekend": "Boolean indicating whether the date of the visit is on a weekend (True / False).",
    "Revenue": "Target variable: True (1) if the session finalized with a transaction, False (0) otherwise.",
}


# =============================================================================
# 5. DATASET VALIDATION & QUALITY AUDIT FUNCTION (SECTION 4)
#    Inspects shape, column presence, data types, missing values, exact duplicate
#    records, target class distribution, and domain range constraints.
# =============================================================================
def validate_dataset(df: pd.DataFrame) -> dict:
    """Run complete schema, dtype, missing, duplicate, target, and range checks."""
    # 5.1 Check column names against expected UCI schema
    present_cols = list(df.columns)
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in present_cols]
    extra_cols = [c for c in present_cols if c not in EXPECTED_COLUMNS]
    is_valid_schema = len(missing_cols) == 0

    # 5.2 Inspect dimensions, exact duplicate rows, and total missing cells
    n_rows, n_cols = df.shape
    duplicate_mask = df.duplicated(keep="first")
    n_duplicates = int(duplicate_mask.sum())
    n_missing_total = int(df.isna().sum().sum())

    # 5.3 Domain & invalid-value checks (non-negative counts/durations, [0, 1] rates)
    invalid_checks = []
    for col in CONTINUOUS_FEATURES:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            neg_cnt = int((df[col] < 0).sum())
            invalid_checks.append({
                "Column": col,
                "Check": "Non-negative values (>= 0)",
                "Invalid Rows": neg_cnt,
                "Status": "Pass" if neg_cnt == 0 else "Warning",
            })

    for rate_col in ("BounceRates", "ExitRates"):
        if rate_col in df.columns and pd.api.types.is_numeric_dtype(df[rate_col]):
            out_cnt = int(((df[rate_col] < 0) | (df[rate_col] > 1.0)).sum())
            invalid_checks.append({
                "Column": rate_col,
                "Check": "Valid probability rate in [0, 1]",
                "Invalid Rows": out_cnt,
                "Status": "Pass" if out_cnt == 0 else "Warning",
            })

    if "SpecialDay" in df.columns and pd.api.types.is_numeric_dtype(df["SpecialDay"]):
        sp_cnt = int(((df["SpecialDay"] < 0) | (df["SpecialDay"] > 1.0)).sum())
        invalid_checks.append({
            "Column": "SpecialDay",
            "Check": "Closeness index in [0.0, 1.0]",
            "Invalid Rows": sp_cnt,
            "Status": "Pass" if sp_cnt == 0 else "Warning",
        })

    # 5.4 Build per-column data dictionary and quality summary table
    col_summary = []
    group_lookup = {col: grp for grp, cols in FEATURE_GROUPS.items() for col in cols}
    group_lookup[TARGET_COL] = "Target (Supervised)"

    for col in present_cols:
        s = df[col]
        col_summary.append({
            "Column": col,
            "Feature Group": group_lookup.get(col, "Other"),
            "Data Type": str(s.dtype),
            "Role": "Target" if col == TARGET_COL else ("Continuous / Numerical" if col in CONTINUOUS_FEATURES else "Categorical / Nominal"),
            "Missing": int(s.isna().sum()),
            "Missing (%)": round(float(s.isna().mean() * 100), 2),
            "Unique Values": int(s.nunique(dropna=True)),
            "Description": FEATURE_DESCRIPTIONS.get(col, ""),
        })

    # 5.5 Compute Revenue target class distribution and imbalance ratio
    target_dist = {}
    if TARGET_COL in df.columns:
        rev_bool = df[TARGET_COL].astype(bool)
        pos_cnt = int(rev_bool.sum())
        neg_cnt = int(n_rows - pos_cnt)
        target_dist = {
            "purchase_count": pos_cnt,
            "no_purchase_count": neg_cnt,
            "purchase_rate_pct": round(pos_cnt / max(n_rows, 1) * 100, 2),
            "no_purchase_rate_pct": round(neg_cnt / max(n_rows, 1) * 100, 2),
            "imbalance_ratio": round(neg_cnt / max(pos_cnt, 1), 2),
        }

    return {
        "is_valid_schema": is_valid_schema,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_input_features": len([c for c in present_cols if c != TARGET_COL]),
        "missing_columns": missing_cols,
        "extra_columns": extra_cols,
        "n_missing_total": n_missing_total,
        "n_duplicates": n_duplicates,
        "duplicate_pct": round(n_duplicates / max(n_rows, 1) * 100, 2),
        "column_summary": col_summary,
        "invalid_checks": invalid_checks,
        "target_distribution": target_dist,
    }
