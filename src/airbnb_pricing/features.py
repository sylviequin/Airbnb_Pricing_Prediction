"""Feature engineering: turning cleaned raw columns into model inputs.

Ported from the notebook's feature-engineering cells (88, 91). The notebook did
this by mutating global `train_data`/`test_data` frames in a fixed cell order;
here each transformation is a small, named, independently testable function.

Anything "fit" on data (the geo cluster model, the rare-category cutoffs, the
neighbourhood target encoder) is a class with `.fit(train_df)` /
`.transform(df)`, so it's fit on train only and re-applied to test — the same
leakage-avoidance the notebook was careful about, just made explicit instead of
relying on cell execution order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from airbnb_pricing.config import (
    AMENITY_MAP,
    CBD_LAT,
    CBD_LON,
    N_GEO_CLUSTERS,
    NAME_KEYWORDS,
    RANDOM_STATE,
    SNAPSHOT_DATE,
    TOP_N_CATEGORIES,
)


def haversine_cbd(lat: pd.Series, lon: pd.Series) -> pd.Series:
    """Great-circle distance (km) from each listing to the Brisbane CBD."""
    r = 6371
    lat1, lon1 = np.radians(lat), np.radians(lon)
    lat2, lon2 = np.radians(CBD_LAT), np.radians(CBD_LON)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return r * 2 * np.arcsin(np.sqrt(a))


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """numerator / denominator, with denominator == 0 treated as missing (not 1)."""
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the per-row derived features that don't require fitting on train.

    amenities_count, host_experience_days, days_since_last_review,
    is_commercial_host, beds_per_person, bath_to_bedroom_ratio,
    accommodates_per_bedroom, availability_rate, dist_to_cbd, is_inactive_host.
    """
    df = df.copy()

    amenities_raw = df["amenities"].fillna("[]")
    df["amenities_count"] = np.where(amenities_raw == "[]", 0, amenities_raw.str.count(",") + 1)

    host_since = pd.to_datetime(df["host_since"], errors="coerce")
    df["host_experience_days"] = (SNAPSHOT_DATE - host_since).dt.days.clip(lower=0)

    last_review = pd.to_datetime(df["last_review"], errors="coerce")
    df["days_since_last_review"] = (SNAPSHOT_DATE - last_review).dt.days

    df["is_commercial_host"] = (df["host_listings_count"] > 1).astype(int)
    df["beds_per_person"] = safe_ratio(df["beds"], df["accommodates"])
    df["bath_to_bedroom_ratio"] = safe_ratio(df["bathrooms"], df["bedrooms"])
    df["accommodates_per_bedroom"] = safe_ratio(df["accommodates"], df["bedrooms"])
    df["availability_rate"] = df["availability_365"] / 365
    df["dist_to_cbd"] = haversine_cbd(df["latitude"], df["longitude"])
    df["is_inactive_host"] = (df["estimated_occupancy_l365d"] == 0).astype(int)

    return df


def extract_amenity_flags(
    df: pd.DataFrame, amenity_map: dict[str, str] = AMENITY_MAP
) -> pd.DataFrame:
    """One binary column per amenity keyword, matched against the raw `amenities` text."""
    df = df.copy()
    amenities_str = df["amenities"].fillna("[]").str.lower()
    for col, pattern in amenity_map.items():
        df[col] = amenities_str.str.contains(pattern, regex=True).astype(int)
    return df


def extract_name_keyword_flags(
    df: pd.DataFrame, keyword_map: dict[str, str] = NAME_KEYWORDS
) -> pd.DataFrame:
    """One binary column per marketing keyword, matched against the raw `name` text."""
    df = df.copy()
    name_str = df["name"].fillna("").str.lower()
    for col, pattern in keyword_map.items():
        df[col] = name_str.str.contains(pattern, regex=True).astype(int)
    return df


class GeoClusterer:
    """K-Means spatial clustering on (latitude, longitude), fit on train only.

    Produces a `geo_cluster` label column (e.g. "zone_0") so downstream
    encoding can treat location as a small categorical instead of two raw
    continuous coordinates.
    """

    def __init__(self, n_clusters: int = N_GEO_CLUSTERS, random_state: int = RANDOM_STATE):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)

    def fit(self, train_df: pd.DataFrame) -> "GeoClusterer":
        coords = train_df[["latitude", "longitude"]]
        self.kmeans.fit(self.scaler.fit_transform(coords))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        coords = df[["latitude", "longitude"]]
        labels = self.kmeans.predict(self.scaler.transform(coords))
        df["geo_cluster"] = ["zone_" + str(label) for label in labels]
        return df


def collapse_rare_categories(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: list[str],
    top_n: int = TOP_N_CATEGORIES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each column, keep the top-N most frequent categories (by train counts)
    and collapse everything else to 'other'. Returns new (train_df, test_df).
    """
    train_df, test_df = train_df.copy(), test_df.copy()
    for col in columns:
        top_categories = train_df[col].value_counts().nlargest(top_n).index
        train_df[col] = train_df[col].where(train_df[col].isin(top_categories), other="other")
        test_df[col] = test_df[col].where(test_df[col].isin(top_categories), other="other")
    return train_df, test_df


class NeighbourhoodTargetEncoder:
    """5-fold target-mean encoding for a high-cardinality categorical column.

    Fit computes out-of-fold means on train (so each row's encoded value never
    saw its own price), and a full-train mean per category for encoding test.
    Unseen categories fall back to the global mean.
    """

    def __init__(self, n_splits: int = 5, random_state: int = RANDOM_STATE):
        self.n_splits = n_splits
        self.random_state = random_state
        self.full_train_means_: pd.Series | None = None
        self.global_mean_: float | None = None

    def fit_transform(self, categories: pd.Series, target: pd.Series) -> np.ndarray:
        """Return out-of-fold encoded values for the training set, and fit the
        full-train lookup table used later by `.transform()` on test/new data.
        """
        self.global_mean_ = target.mean()
        kf = KFold(self.n_splits, shuffle=True, random_state=self.random_state)

        encoded = np.zeros(len(categories))
        for train_idx, val_idx in kf.split(categories):
            fold_means = target.iloc[train_idx].groupby(categories.iloc[train_idx]).mean()
            encoded[val_idx] = (
                categories.iloc[val_idx].map(fold_means).fillna(self.global_mean_).values
            )

        self.full_train_means_ = target.groupby(categories).mean()
        return encoded

    def transform(self, categories: pd.Series) -> np.ndarray:
        if self.full_train_means_ is None:
            raise RuntimeError("Call fit_transform on the training data first.")
        return categories.map(self.full_train_means_).fillna(self.global_mean_).values
