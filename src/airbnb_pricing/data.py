"""Loading and basic cleaning of the raw Airbnb listings data.

Ported from the notebook's "Load data" and price-cleaning cells (cells 11, 83).
"""

from pathlib import Path

import pandas as pd

from airbnb_pricing.config import COLUMNS_TO_DROP, RAW_TRAIN_PATH


def load_raw_data(path: Path = RAW_TRAIN_PATH) -> pd.DataFrame:
    """Load a raw listings CSV as-is (no cleaning)."""
    return pd.read_csv(path)


def clean_price_column(df: pd.DataFrame, column: str = "price") -> pd.DataFrame:
    """Strip `$`/commas from the price column and cast to float.

    The raw data stores price like "$97.00" or "$1,250.50" — train.csv has this
    column, test.csv does not (it's the prediction target).
    """
    df = df.copy()
    if column in df.columns:
        df[column] = df[column].astype(str).str.replace(r"[$,]", "", regex=True).astype(float)
    return df


def clean_percentage_columns(
    df: pd.DataFrame, columns: tuple[str, ...] = ("host_response_rate", "host_acceptance_rate")
) -> pd.DataFrame:
    """Strip `%` from rate columns (e.g. "89%") and cast to float."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .replace("nan", pd.NA)
                .astype(float)
            )
    return df


def drop_unused_columns(df: pd.DataFrame, columns: list[str] = COLUMNS_TO_DROP) -> pd.DataFrame:
    """Drop identifier / free-text / redundant columns that carry no modeling signal.

    See config.COLUMNS_TO_DROP for the rationale behind each column.
    """
    return df.drop(columns=[c for c in columns if c in df.columns])
