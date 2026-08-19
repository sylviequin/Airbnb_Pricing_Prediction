"""Score new listings with a saved model bundle (pipeline + model, from train.py).

Run:
    python -m airbnb_pricing.predict --bundle models/xgboost_bundle.joblib \\
        --input data/test.csv --output reports/predictions.csv
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from airbnb_pricing.data import load_raw_data


def predict(bundle_path: str, input_path: str) -> pd.Series:
    bundle = joblib.load(bundle_path)
    pipeline, model = bundle["pipeline"], bundle["model"]

    raw_df = load_raw_data(input_path)
    X = pipeline.transform(raw_df)
    log_price_pred = model.predict(X)
    return pd.Series(np.expm1(log_price_pred), name="predicted_price")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle", required=True, help="Path to a *_bundle.joblib saved by train.py"
    )
    parser.add_argument("--input", required=True, help="CSV of raw listings to score")
    parser.add_argument("--output", required=True, help="Where to write predictions CSV")
    args = parser.parse_args()

    preds = predict(args.bundle, args.input)
    preds.to_csv(args.output, index=False)
    print(f"Wrote {len(preds)} predictions to {args.output}")
