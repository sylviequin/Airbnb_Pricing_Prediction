# Repository design — Airbnb Pricing Prediction

Status: the folder rename is done, the notebook logic has been split into a real `src/` package, and it's verified against the actual data. Below is the final structure and what's still manual.

## Final structure

```
airbnb-pricing-prediction/
├── README.md
├── LICENSE                     MIT
├── .gitignore                  excludes data/*.csv, catboost_info/, models/, caches, OS junk
├── requirements.txt            runtime deps (verified against actual imports)
├── requirements-dev.txt        pytest, ruff, black, pre-commit, nbstripout
├── pyproject.toml              package metadata + ruff/black/pytest config
├── .pre-commit-config.yaml     auto-lint + strip notebook outputs before commit
├── .github/workflows/ci.yml    lint + test on every push
├── data/
│   └── README.md                data source notes — raw CSVs are .gitignored, not committed
├── notebook/
│   └── airbnb_pricing_prediction.ipynb    original EDA + narrative (BUSA3020 group assignment)
├── reports/
│   └── figures/                 saved evaluation plots (populated by train.py)
├── models/                      gitignored — trained model bundles, regenerate with train.py
├── src/airbnb_pricing/          the pipeline as an installable package — see README for module map
└── tests/                       pytest suite, including a synthetic end-to-end pipeline test
```

## What changed from the original notebook

The notebook did feature engineering and encoding by mutating global `train_data`/`test_data`
frames across ~15 cells, in a specific run order, with train-only fitting (medians, K-Means
clusters, rare-category cutoffs, target encoding) done correctly but implicitly — get the cell
order wrong on a re-run and it's easy to leak test statistics into train, or vice versa.

`src/airbnb_pricing/pipeline.py`'s `FeaturePipeline` makes that explicit: `.fit_transform(train_df)`
fits every train-only step once and returns `(X_train, y_train)`; `.transform(test_df)` (or any new
raw data at inference time) re-applies those already-fitted steps without ever re-fitting. The whole
fitted object serializes with the model via `joblib`, so `predict.py` doesn't need the training data
to score new listings.

Verified end-to-end against the real `train.csv`/`test.csv`: `FeaturePipeline` reproduces the
notebook's own row/column counts exactly (3,735 train rows -> 91 model-ready features), and
`python -m airbnb_pricing.train --quick` runs the full load → engineer → tune → save pipeline in
under 3 minutes with no errors.

## What's still manual

- **Run a full (non-`--quick`) `train.py`** and fill in the results table in README.md — the
  `--quick` run used a tiny hyperparameter search (3 iterations, 2-fold CV) just to prove the
  pipeline is wired correctly, not to produce a real result.
- **Decide on team credit.** This started as a group assignment (BUSA3020, team
  BUSA3020_Dataholic) — README.md has a placeholder for whether/how to credit teammates before
  this goes public.
- **Push to GitHub** — `git init`, `gh repo create`, first commit. Ask if you want a walkthrough.

## Why the raw data stays out of git

The CSVs total ~11 MB and match InsideAirbnb's column schema for Brisbane. Per your earlier
call, `data/` is gitignored and `data/README.md` documents the source instead, so the repo stays
code-only and doesn't risk redistributing someone else's scrape without checking its terms.
