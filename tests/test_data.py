"""Example unit test — this is the pattern to follow for the rest of src/.

A professional repo doesn't need 100% coverage, but a handful of tests on
the trickiest logic (price parsing, feature engineering edge cases) proves
the code works and catches regressions when you tweak the pipeline later.
"""

import pandas as pd

from airbnb_pricing.data import clean_price_column


def test_clean_price_column_strips_currency_formatting():
    df = pd.DataFrame({"price": ["$97.00", "$1,250.50"]})

    result = clean_price_column(df)

    assert result["price"].tolist() == [97.00, 1250.50]
