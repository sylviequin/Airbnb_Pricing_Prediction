"""Shared test fixtures.

`make_raw_listings` builds a small synthetic dataframe with every raw column
the pipeline touches, so tests don't depend on the real (gitignored) CSVs —
CI has no access to data/train.csv, only to this fixture.
"""

import numpy as np
import pandas as pd
import pytest

_NEIGHBOURHOODS = [
    "Brisbane City",
    "South Brisbane",
    "Brisbane City",
    "New Farm",
    "Brisbane City",
    "Sunnybank",
]


def make_raw_listings(n: int = 12, with_price: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = {
        "ID": range(n),
        "name": [f"Cosy {i} bed unit near CBD" for i in range(n)],
        "description": ["some text"] * n,
        "neighborhood_overview": ["overview"] * n,
        "host_id": range(1000, 1000 + n),
        "host_name": [f"Host {i}" for i in range(n)],
        "host_since": pd.date_range("2018-01-01", periods=n, freq="200D").astype(str),
        "host_location": ["Brisbane"] * n,
        "host_about": ["about"] * n,
        "host_response_time": (["within an hour", "within a day", None] * n)[:n],
        "host_response_rate": ([f"{v}%" for v in rng.integers(50, 100, n)]),
        "host_acceptance_rate": ([f"{v}%" for v in rng.integers(50, 100, n)]),
        "host_is_superhost": (["t", "f"] * n)[:n],
        "host_neighbourhood": ["Brisbane City"] * n,
        "host_listings_count": rng.integers(1, 20, n).astype(float),
        "host_verifications": ["['email']"] * n,
        "host_identity_verified": (["t", "f"] * n)[:n],
        "neighbourhood_cleansed": (_NEIGHBOURHOODS * 3)[:n],
        "latitude": -27.47 + rng.normal(0, 0.05, n),
        "longitude": 153.02 + rng.normal(0, 0.05, n),
        "property_type": (["Entire rental unit", "Entire home", "Private room in home"] * n)[:n],
        "room_type": (["Entire home/apt", "Private room"] * n)[:n],
        "accommodates": rng.integers(1, 8, n).astype(float),
        "bathrooms": rng.integers(1, 3, n).astype(float),
        "bathrooms_text": ["1 bath"] * n,
        "bedrooms": rng.integers(1, 4, n).astype(float),
        "beds": rng.integers(1, 4, n).astype(float),
        "amenities": ['["Wifi", "Kitchen", "Free parking on premises"]'] * n,
        "minimum_nights": rng.integers(1, 5, n).astype(float),
        "maximum_nights": rng.integers(30, 365, n).astype(float),
        "availability_30": rng.integers(0, 30, n).astype(float),
        "availability_365": rng.integers(0, 365, n).astype(float),
        "number_of_reviews": rng.integers(0, 50, n).astype(float),
        "number_of_reviews_ltm": rng.integers(0, 20, n).astype(float),
        "number_of_reviews_l30d": rng.integers(0, 5, n).astype(float),
        "estimated_occupancy_l365d": rng.integers(0, 300, n).astype(float),
        "estimated_revenue_l365d": rng.integers(0, 30000, n).astype(float),
        "last_review": pd.date_range("2024-01-01", periods=n, freq="30D").astype(str),
        "review_scores_rating": rng.uniform(3.5, 5.0, n),
        "review_scores_accuracy": rng.uniform(3.5, 5.0, n),
        "review_scores_cleanliness": rng.uniform(3.5, 5.0, n),
        "review_scores_checkin": rng.uniform(3.5, 5.0, n),
        "review_scores_communication": rng.uniform(3.5, 5.0, n),
        "review_scores_location": rng.uniform(3.5, 5.0, n),
        "review_scores_value": rng.uniform(3.5, 5.0, n),
        "instant_bookable": (["t", "f"] * n)[:n],
        "calculated_host_listings_count": rng.integers(1, 10, n).astype(float),
        "reviews_per_month": rng.uniform(0, 5, n),
    }
    df = pd.DataFrame(data)
    if with_price:
        df["price"] = [f"${p:.2f}" for p in rng.uniform(50, 500, n)]
    return df


@pytest.fixture
def raw_train_df():
    return make_raw_listings(n=12, with_price=True)


@pytest.fixture
def raw_test_df():
    return make_raw_listings(n=6, with_price=False)
