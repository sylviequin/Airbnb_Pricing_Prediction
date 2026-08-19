# Airbnb Pricing Prediction

Predicting nightly listing price for Brisbane, Australia Airbnb listings from host, property, availability, and review features. Originally built for a Kaggle-linked university assignment (BUSA3020); this repo is the code refactored into a runnable package instead of a single notebook.

## Overview

- **Task:** regression — predict `price` (AUD) per listing, trained on `log1p(price)`
- **Data:** Brisbane Airbnb listings scrape, 3,735 training rows / 1,601 test rows, 65 raw columns. See [`data/README.md`](data/README.md) for how to get it — not committed to this repo.
- **Models compared:** Random Forest, XGBoost, CatBoost (tuned via `RandomizedSearchCV`), with OLS/Lasso as linear baselines in the original notebook.
- **Pipeline:** `src/airbnb_pricing/` — feature engineering (amenity/keyword extraction, geo-clustering, distance-to-CBD, target encoding) turns 65 raw columns into 91 model-ready features. Verified end-to-end against the real data: `python -m airbnb_pricing.train --quick` runs the full pipeline in under 3 minutes.

## A note on the evaluation metric

The original notebook's model-comparison tables report "MAPE" computed directly on `log1p(price)` predictions — not on actual dollar price, even though the Kaggle competition scores on real-price MAPE. Both numbers are meaningfully different (in a quick verification run: ~4% log-price MAPE vs. ~21% real price-scale MAPE for the same model), so `train.py` now reports both explicitly:

- **CV MAPE (log-price)** — used during hyperparameter search, matches what the notebook reported
- **price-scale MAPE / R²** — computed via out-of-fold predictions inverse-transformed with `expm1`, matches the competition's actual definition

Run the full (non-`--quick`) training to get final numbers — the table below has TODOs where those go.

## Repo structure

```
├── src/airbnb_pricing/     the pipeline as an installable package (see below)
├── tests/                  pytest suite — unit tests + a synthetic end-to-end pipeline test
├── notebook/                the original analysis notebook (EDA writeup + narrative)
├── data/                    data source notes (raw CSVs are not committed — see data/README.md)
├── reports/figures/        saved charts from model evaluation
├── models/                  trained model bundles (gitignored — regenerate with train.py)
└── requirements*.txt
```

### `src/airbnb_pricing/`

| Module | What it does |
|---|---|
| `config.py` | Paths, feature lists, hyperparameter constants — one source of truth instead of redefined per notebook cell |
| `data.py` | Load raw CSVs, clean `price`/percentage columns, drop unused columns |
| `features.py` | Amenity + name-keyword extraction, CBD distance, geo-clustering (`GeoClusterer`), rare-category collapsing, K-fold neighbourhood target encoding (`NeighbourhoodTargetEncoder`) |
| `preprocessing.py` | One-hot/ordinal/boolean encoding, train-only median imputation |
| `pipeline.py` | `FeaturePipeline` — fits every train-only step once, `.transform()` re-applies it to test/new data without re-fitting (no leakage) |
| `train.py` | Tunes RF/XGBoost/CatBoost via `RandomizedSearchCV`, picks the best by CV MAPE, saves a `{pipeline, model}` bundle + feature importance plot + Kaggle submission CSV |
| `evaluate.py` | MAPE/R² helpers, feature importance plotting |
| `predict.py` | Loads a saved bundle, scores new raw listings |

## Getting started

```bash
git clone <your-repo-url>
cd airbnb-pricing-prediction
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
```

Download the data per [`data/README.md`](data/README.md) into `data/train.csv` / `data/test.csv`, then:

```bash
pytest                                    # run the test suite (no data files needed)
python -m airbnb_pricing.train --quick    # fast smoke test (~3 min)
python -m airbnb_pricing.train            # full run — tunes all 3 models (~15-20 min)
python -m airbnb_pricing.predict --bundle models/xgboost_bundle.joblib \
    --input data/test.csv --output reports/predictions.csv
```

The original exploratory analysis (missing-value breakdown, univariate/bivariate plots, feature-selection writeup) is in `notebook/airbnb_pricing_prediction.ipynb`.

## Results

TODO — fill in from a full (non-`--quick`) `train.py` run:

| Model | CV MAPE (log-price) | Price-scale MAPE (OOF) | Price-scale R² (OOF) |
|---|---|---|---|
| Random Forest | TODO | TODO | TODO |
| XGBoost | TODO | TODO | TODO |
| CatBoost | TODO | TODO | TODO |

Top features by importance for the winning model: see `reports/figures/`.

## Team / acknowledgments

This started as a group assignment for BUSA3020 (Kaggle team `BUSA3020_Dataholic`). The original EDA, feature selection, and modeling approach in `notebook/airbnb_pricing_prediction.ipynb` were developed together with:

- Dieu Linh Ngo
- Phuong Linh Ngo
- Quynh Huong Nguyen

The `src/airbnb_pricing/` package is a solo refactor of that shared work into a runnable pipeline.

## License

MIT — see [LICENSE](LICENSE).
