import numpy as np
import pandas as pd

from airbnb_pricing.config import CBD_LAT, CBD_LON
from airbnb_pricing.features import (
    extract_amenity_flags,
    extract_name_keyword_flags,
    haversine_cbd,
    safe_ratio,
)


def test_haversine_cbd_is_zero_at_the_cbd_itself():
    dist = haversine_cbd(pd.Series([CBD_LAT]), pd.Series([CBD_LON]))
    assert dist.iloc[0] == 0


def test_safe_ratio_treats_zero_denominator_as_missing_not_one():
    result = safe_ratio(pd.Series([4, 6]), pd.Series([2, 0]))
    assert result.iloc[0] == 2
    assert np.isnan(result.iloc[1])


def test_extract_amenity_flags_matches_case_insensitively():
    df = pd.DataFrame({"amenities": ['["Wifi", "Free parking on premises"]', None]})
    result = extract_amenity_flags(df)
    assert result["has_wifi"].tolist() == [1, 0]
    assert result["has_free_parking"].tolist() == [1, 0]


def test_extract_name_keyword_flags_finds_keywords_in_title():
    df = pd.DataFrame({"name": ["Luxury CBD Penthouse with Ocean View", "Cosy studio"]})
    result = extract_name_keyword_flags(df)
    assert result["name_luxury"].tolist() == [1, 0]
    assert result["name_studio"].tolist() == [0, 1]
    assert result["name_cbd"].tolist() == [1, 0]
