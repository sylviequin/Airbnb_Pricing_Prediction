"""Encoding and imputation steps applied after feature engineering.

Ported from the notebook's encoding/imputation cells (99, 106-113). All
"fit" values (medians, one-hot columns) are computed from train and then
applied to test, never the other way round.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from airbnb_pricing.config import BOOLEAN_COLUMNS, MEDIAN_IMPUTE_COLUMNS, RESPONSE_TIME_ORDER


def one_hot_encode(
    train_df: pd.DataFrame, test_df: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encode `columns`, fit on train categories, test columns aligned
    to match (unseen categories in test become all-zero rows).
    """
    train_dummies = pd.get_dummies(train_df[columns], dtype=int)
    test_dummies = pd.get_dummies(test_df[columns], dtype=int)
    test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)
    return train_dummies, test_dummies


def encode_response_time(
    df: pd.DataFrame, mapping: dict[str, int] = RESPONSE_TIME_ORDER
) -> pd.DataFrame:
    """Ordinal-encode host_response_time (faster response -> lower value)."""
    df = df.copy()
    df["host_response_time"] = df["host_response_time"].fillna("unknown").map(mapping)
    return df


def encode_boolean_columns(df: pd.DataFrame, columns: list[str] = BOOLEAN_COLUMNS) -> pd.DataFrame:
    """Map Airbnb's 't'/'f' string booleans to 1/0, filling missing as 'f'."""
    df = df.copy()
    for col in columns:
        df[col] = df[col].fillna("f").map({"t": 1, "f": 0})
    return df


def compute_imputation_medians(
    train_df: pd.DataFrame, columns: list[str] = MEDIAN_IMPUTE_COLUMNS
) -> pd.Series:
    """Median of each column, computed from train only."""
    return train_df[columns].median()


def apply_median_imputation(
    df: pd.DataFrame, medians: pd.Series, days_since_last_review_fill: float
) -> pd.DataFrame:
    """Fill missing values using train-derived medians (and 0 for reviews_per_month,
    which means "no reviews yet" rather than "unknown").

    days_since_last_review is filled with `days_since_last_review_fill` (the train
    max) rather than a median, since a missing value here means the listing has
    never been reviewed — using the median would imply an average-recency review
    that doesn't exist.
    """
    df = df.copy()
    df["days_since_last_review"] = df["days_since_last_review"].fillna(days_since_last_review_fill)
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
    for col in medians.index:
        if col in df.columns:
            df[col] = df[col].fillna(medians[col])
    return df


def clean_numeric_edge_cases(df: pd.DataFrame, fill_values: pd.Series) -> pd.DataFrame:
    """Replace any residual +/-inf (e.g. from a ratio divided by zero that slipped
    through) with NaN, then fill with the given column medians.
    """
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(fill_values)
    return df
