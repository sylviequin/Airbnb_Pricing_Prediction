"""Shared evaluation metrics and plots.

The competition (and the notebook) scores on MAPE computed in the original
price scale — these helpers assume predictions/targets passed in are already
back-transformed with `np.expm1` if the model was trained on log1p(price).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, r2_score


def evaluate_predictions(y_true, y_pred) -> dict[str, float]:
    """Return MAPE and R² — the two metrics the notebook tracked throughout."""
    return {
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def plot_feature_importance(
    feature_names, importances, top_n: int = 20, title: str = "Feature Importance"
):
    """Horizontal bar chart of the top-N most important features. Returns the Figure
    so the caller decides whether to show it, save it, or both.
    """
    series = pd.Series(importances, index=feature_names).nlargest(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    series.plot(kind="barh", color="steelblue", ax=ax)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig
