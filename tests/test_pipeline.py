"""End-to-end pipeline test on synthetic data (see conftest.py).

This is the test that would have caught wiring bugs between data.py,
features.py, preprocessing.py, and pipeline.py — the kind of mistake that's
easy to make when splitting one big notebook into separate modules.
"""

import numpy as np

from airbnb_pricing.pipeline import FeaturePipeline


def test_fit_transform_produces_matching_columns_for_train_and_test(raw_train_df, raw_test_df):
    pipeline = FeaturePipeline()

    X_train, y_train = pipeline.fit_transform(raw_train_df)
    X_test = pipeline.transform(raw_test_df)

    assert list(X_train.columns) == list(X_test.columns)
    assert len(X_train) == len(raw_train_df)
    assert len(X_test) == len(raw_test_df)
    assert not X_train.isna().any().any()
    assert not X_test.isna().any().any()
    # y_train is log1p(price), so it should be strictly positive here
    assert (y_train > 0).all()


def test_transform_does_not_require_price_column(raw_train_df, raw_test_df):
    pipeline = FeaturePipeline()
    pipeline.fit_transform(raw_train_df)

    assert "price" not in raw_test_df.columns
    X_test = pipeline.transform(raw_test_df)
    assert np.isfinite(X_test.to_numpy()).all()
