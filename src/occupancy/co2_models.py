"""The four regression methods taught in MMA3001 Week 5, applied to CO2 prediction.

Week 5's notebooks (5.1-5.4) each cover one regression technique --
Linear Regression, Decision Tree Regression, Support Vector Regression and
Neural Network Regression -- and 5.5 teaches comparing them with MAE/RMSE/R².
This module builds all four as directly comparable pipelines for the CO2
regression task in :mod:`occupancy.co2_regression`, so the "which method is
most suitable, and why" comparison (occupancy.evaluation, the "Alternative
solutions" report requirement) uses methods actually taught in the unit --
unlike the occupancy classifier, which uses classification methods (Logistic
Regression, Random Forest Classifier) that Week 5 never covers.

All four share one preprocessing step (`StandardScaler`). Scaling is
essential for SVR and the neural network (both sensitive to feature scale)
and harmless for the other two: ordinary least squares is scale-invariant
in its predictions, and a decision tree's splits are invariant to any
monotonic per-feature transform. Using one shared preprocessing step
keeps the four models genuinely comparable -- differences in performance
come from the model, not from inconsistent inputs.
"""

from __future__ import annotations

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def build_linear_regression_pipeline(**kwargs) -> Pipeline:
    """Week 5.1: ordinary least-squares linear regression."""
    return Pipeline(
        [("scale", StandardScaler()), ("regress", LinearRegression(**kwargs))]
    )


def build_decision_tree_regression_pipeline(**kwargs) -> Pipeline:
    """Week 5.2: a single decision tree regressor.

    ``random_state=42`` is fixed by default for reproducibility; pass
    ``random_state=...`` to override.
    """
    defaults = dict(max_depth=8, random_state=42)
    defaults.update(kwargs)
    return Pipeline(
        [("scale", StandardScaler()), ("regress", DecisionTreeRegressor(**defaults))]
    )


def build_svr_pipeline(**kwargs) -> Pipeline:
    """Week 5.3: Support Vector Regression (RBF kernel, scikit-learn default)."""
    defaults = dict(C=10.0, epsilon=0.5)
    defaults.update(kwargs)
    return Pipeline([("scale", StandardScaler()), ("regress", SVR(**defaults))])


def build_neural_network_regression_pipeline(**kwargs) -> Pipeline:
    """Week 5.4: a small multi-layer-perceptron regressor.

    ``random_state=42`` and a modest ``max_iter`` are fixed by default so
    runs are reproducible and terminate promptly on this dataset's size;
    override either via kwargs.
    """
    defaults = dict(
        hidden_layer_sizes=(32, 16),
        max_iter=500,
        random_state=42,
        early_stopping=True,
    )
    defaults.update(kwargs)
    return Pipeline(
        [("scale", StandardScaler()), ("regress", MLPRegressor(**defaults))]
    )
