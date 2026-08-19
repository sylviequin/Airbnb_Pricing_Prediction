"""Train and compare pricing models, save the best one.

Ported from the notebook's model-comparison cells (159-165): baseline CV,
then RandomizedSearchCV tuning for Random Forest, XGBoost, and CatBoost, then
feature importance for the winner.

One thing worth flagging that the notebook didn't: the model is trained on
log1p(price), and the notebook's own "CV MAPE" figures (cell 136, 160) are
MAPE computed directly on the log-price predictions — not on actual dollar
price, even though the competition's stated metric (and the Kaggle
submission step) is MAPE on real price. Both are reported below so it's
explicit which is which; `price_scale_mape` is the one that matches the
competition's definition.

Run:
    python -m airbnb_pricing.train                  # full run (~10-20 min, all 3 models tuned)
    python -m airbnb_pricing.train --quick           # fast smoke-test (small search, for CI/dev)
    python -m airbnb_pricing.train --model catboost  # tune+save a single model instead of all three
"""

from __future__ import annotations

import argparse
import logging

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict

from airbnb_pricing.config import (
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    RAW_TEST_PATH,
    RAW_TRAIN_PATH,
)
from airbnb_pricing.data import load_raw_data
from airbnb_pricing.evaluate import evaluate_predictions, plot_feature_importance
from airbnb_pricing.pipeline import FeaturePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RF_PARAM_DIST = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [10, 20, 30, None],
    "max_features": ["sqrt", "log2", 0.5, 0.8],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4, 8],
}

XGB_PARAM_DIST = {
    "n_estimators": [300, 500, 1000, 2000],
    "max_depth": [3, 4, 6, 8],
    "eta": [0.005, 0.01, 0.05, 0.1],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 10],
    "reg_alpha": [0, 0.01, 0.1, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
}

CATBOOST_PARAM_DIST = {
    "depth": [4, 6, 8, 10],
    "l2_leaf_reg": [1, 3, 5, 7, 9],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "iterations": [200, 300, 500, 1000],
}


def _build_search(name: str, n_iter: int, cv: int) -> RandomizedSearchCV:
    if name == "random_forest":
        estimator = RandomForestRegressor(random_state=RANDOM_STATE, criterion="absolute_error")
        param_dist = RF_PARAM_DIST
    elif name == "xgboost":
        from xgboost import XGBRegressor

        estimator = XGBRegressor(
            objective="reg:absoluteerror",
            eval_metric="mape",
            random_state=RANDOM_STATE,
            verbosity=0,
            tree_method="hist",
        )
        param_dist = XGB_PARAM_DIST
    elif name == "catboost":
        from catboost import CatBoostRegressor

        estimator = CatBoostRegressor(loss_function="MAPE", random_state=RANDOM_STATE, verbose=0)
        param_dist = CATBOOST_PARAM_DIST
    else:
        raise ValueError(f"Unknown model: {name}")

    return RandomizedSearchCV(
        estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_absolute_percentage_error",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def train_and_compare(X_train, y_train, model_names: list[str], n_iter: int, cv: int):
    """RandomizedSearchCV each model, return {name: fitted_search}."""
    searches = {}
    for name in model_names:
        logger.info("Tuning %s (n_iter=%d, cv=%d)...", name, n_iter, cv)
        search = _build_search(name, n_iter=n_iter, cv=cv)
        search.fit(X_train, y_train)
        searches[name] = search
        logger.info(
            "  %s best CV MAPE (log-price): %.4f  params: %s",
            name,
            -search.best_score_,
            search.best_params_,
        )
    return searches


def main(model_names: list[str], quick: bool) -> None:
    n_iter = 3 if quick else 20
    cv = 2 if quick else 5

    logger.info("Loading and transforming data...")
    raw_train = load_raw_data(RAW_TRAIN_PATH)
    raw_test = load_raw_data(RAW_TEST_PATH)

    pipeline = FeaturePipeline()
    X_train, y_train = pipeline.fit_transform(raw_train)
    X_test = pipeline.transform(raw_test)
    logger.info("X_train: %s, X_test: %s", X_train.shape, X_test.shape)

    searches = train_and_compare(X_train, y_train, model_names, n_iter=n_iter, cv=cv)

    best_name = min(searches, key=lambda n: -searches[n].best_score_)
    best_search = searches[best_name]
    best_model = best_search.best_estimator_
    logger.info("Best model: %s", best_name)

    # Out-of-fold predictions -> honest price-scale MAPE/R² (matches the
    # competition's actual metric, unlike the log-price MAPE used for search above)
    oof_log_pred = cross_val_predict(best_model, X_train, y_train, cv=cv, n_jobs=-1)
    price_scale_metrics = evaluate_predictions(np.expm1(y_train), np.expm1(oof_log_pred))
    logger.info(
        "%s out-of-fold price-scale metrics: MAPE=%.4f  R2=%.4f",
        best_name,
        price_scale_metrics["mape"],
        price_scale_metrics["r2"],
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle_path = MODELS_DIR / f"{best_name}_bundle.joblib"
    joblib.dump({"pipeline": pipeline, "model": best_model, "model_name": best_name}, bundle_path)
    logger.info("Saved pipeline + model bundle to %s", bundle_path)

    if hasattr(best_model, "feature_importances_"):
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig = plot_feature_importance(
            X_train.columns,
            best_model.feature_importances_,
            title=f"{best_name} — Feature Importance",
        )
        fig_path = FIGURES_DIR / f"{best_name}_feature_importance.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        logger.info("Saved feature importance plot to %s", fig_path)

    # Kaggle-style submission using the winning model
    y_test_pred = np.expm1(best_model.predict(X_test))
    submission = raw_test[["ID"]].copy()
    submission["price"] = np.round(y_test_pred, 2)
    submission_path = MODELS_DIR / f"submission_{best_name}.csv"
    submission.to_csv(submission_path, index=False)
    logger.info("Saved submission to %s", submission_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        choices=["random_forest", "xgboost", "catboost"],
        help="Restrict to one model (repeatable). Default: compare all three.",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Small search + 2-fold CV, for a fast smoke test."
    )
    args = parser.parse_args()
    main(args.model or ["random_forest", "xgboost", "catboost"], quick=args.quick)
