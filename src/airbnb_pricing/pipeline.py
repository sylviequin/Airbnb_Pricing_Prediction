"""End-to-end feature pipeline: raw listings CSV -> model-ready X/y.

This replaces the notebook's approach of mutating global `train_data` /
`test_data` frames across ~15 cells in a fixed run order. `FeaturePipeline`
is a single object that:

  - `fit_transform(raw_train_df)` fits every train-only step (geo clusters,
    rare-category cutoffs, imputation medians, the neighbourhood target
    encoder, one-hot columns) and returns (X_train, y_train)
  - `transform(raw_df)` re-applies those already-fitted steps to new data
    (test.csv, or a single new listing at inference time) — nothing is
    re-fit, so there's no risk of the test set leaking into training
    statistics.

The whole fitted object can be pickled with joblib alongside the trained
model, so predict.py doesn't need to re-derive medians/clusters/encodings
from scratch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from airbnb_pricing.config import (
    AMENITY_MAP,
    BASE_NUM_FEATURES,
    BOOLEAN_COLUMNS,
    CAT_FEATURES,
    CAT_NOMINAL,
    DROP_AFTER_ENGINEERING,
    ENG_NUM_FEATURES,
    NAME_KEYWORDS,
    TARGET_COLUMN,
    TOP_N_CATEGORIES,
)
from airbnb_pricing.data import clean_percentage_columns, clean_price_column, drop_unused_columns
from airbnb_pricing.features import (
    GeoClusterer,
    NeighbourhoodTargetEncoder,
    add_engineered_features,
    extract_amenity_flags,
    extract_name_keyword_flags,
)
from airbnb_pricing.preprocessing import (
    apply_median_imputation,
    compute_imputation_medians,
    encode_boolean_columns,
    encode_response_time,
)

_AMENITY_FEATURES = list(AMENITY_MAP.keys())
_NAME_FEATURES = list(NAME_KEYWORDS.keys())
_SELECT_COLUMNS = (
    BASE_NUM_FEATURES
    + ENG_NUM_FEATURES
    + CAT_FEATURES
    + BOOLEAN_COLUMNS
    + _AMENITY_FEATURES
    + _NAME_FEATURES
)


class FeaturePipeline:
    """Fit on train.csv, transform train and/or test.csv into model-ready frames."""

    def __init__(self) -> None:
        self.geo_clusterer = GeoClusterer()
        self.target_encoder = NeighbourhoodTargetEncoder()
        self.category_top_values: dict[str, pd.Index] = {}
        self.one_hot_columns: pd.Index | None = None
        self.imputation_medians: pd.Series | None = None
        self.days_since_last_review_fill: float | None = None
        self.all_features: list[str] | None = None
        self.numeric_fill_medians: pd.Series | None = None

    @staticmethod
    def _clean_and_engineer(raw_df: pd.DataFrame, has_target: bool) -> pd.DataFrame:
        df = clean_percentage_columns(raw_df)
        df = drop_unused_columns(df)
        if has_target:
            df = clean_price_column(df, TARGET_COLUMN)
        df = add_engineered_features(df)
        df = extract_amenity_flags(df)
        df = extract_name_keyword_flags(df)
        return df

    def fit_transform(self, raw_train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        df = self._clean_and_engineer(raw_train_df, has_target=True)
        self.geo_clusterer.fit(df)
        df = self.geo_clusterer.transform(df)
        df = df.drop(columns=[c for c in DROP_AFTER_ENGINEERING if c in df.columns])

        select_cols = [c for c in _SELECT_COLUMNS if c in df.columns]
        selection = df[select_cols + [TARGET_COLUMN]].copy()

        # Full-precision neighbourhood (before rare-category collapsing) for target encoding
        neighbourhood_full = selection["neighbourhood_cleansed"].copy()

        self.days_since_last_review_fill = selection["days_since_last_review"].max()
        self.imputation_medians = compute_imputation_medians(selection)
        selection = apply_median_imputation(
            selection, self.imputation_medians, self.days_since_last_review_fill
        )
        selection = encode_response_time(selection)
        selection = encode_boolean_columns(selection)

        for col in CAT_NOMINAL:
            top_values = selection[col].value_counts().nlargest(TOP_N_CATEGORIES).index
            self.category_top_values[col] = top_values
            selection[col] = selection[col].where(selection[col].isin(top_values), other="other")

        cat_dummies = pd.get_dummies(selection[CAT_NOMINAL], dtype=int)
        self.one_hot_columns = cat_dummies.columns

        y_train = np.log1p(selection[TARGET_COLUMN])

        self.all_features = [c for c in selection.columns if c not in CAT_NOMINAL + [TARGET_COLUMN]]
        selection["neighbourhood_te"] = self.target_encoder.fit_transform(
            neighbourhood_full, y_train
        )
        self.all_features.append("neighbourhood_te")

        X = pd.concat(
            [
                selection[self.all_features].reset_index(drop=True),
                cat_dummies.reset_index(drop=True),
            ],
            axis=1,
        )
        X = X.replace([np.inf, -np.inf], np.nan)
        self.numeric_fill_medians = X.median(numeric_only=True)
        X = X.fillna(self.numeric_fill_medians)

        return X, y_train

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        if self.all_features is None:
            raise RuntimeError("Call fit_transform on the training data before transform().")

        df = self._clean_and_engineer(raw_df, has_target=False)
        df = self.geo_clusterer.transform(df)
        df = df.drop(columns=[c for c in DROP_AFTER_ENGINEERING if c in df.columns])

        select_cols = [c for c in _SELECT_COLUMNS if c in df.columns]
        selection = df[select_cols].copy()

        neighbourhood_full = selection["neighbourhood_cleansed"].copy()

        selection = apply_median_imputation(
            selection, self.imputation_medians, self.days_since_last_review_fill
        )
        selection = encode_response_time(selection)
        selection = encode_boolean_columns(selection)

        for col in CAT_NOMINAL:
            top_values = self.category_top_values[col]
            selection[col] = selection[col].where(selection[col].isin(top_values), other="other")

        cat_dummies = pd.get_dummies(selection[CAT_NOMINAL], dtype=int)
        cat_dummies = cat_dummies.reindex(columns=self.one_hot_columns, fill_value=0)

        selection["neighbourhood_te"] = self.target_encoder.transform(neighbourhood_full)

        X = pd.concat(
            [
                selection.reindex(columns=self.all_features, fill_value=0).reset_index(drop=True),
                cat_dummies.reset_index(drop=True),
            ],
            axis=1,
        )
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(self.numeric_fill_medians)
        return X
