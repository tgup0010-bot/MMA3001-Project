"""Unit tests for occupancy.model pipelines (logistic regression + random forest)."""

import numpy as np
import pandas as pd
import pytest

from occupancy.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from occupancy.model import build_logistic_pipeline, build_random_forest_pipeline

PIPELINE_BUILDERS = [build_logistic_pipeline, build_random_forest_pipeline]


def _toy_dataset(n: int = 40) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {name: rng.uniform(-1, 1, size=n) for name in NUMERIC_FEATURES}
    )
    for name in CATEGORICAL_FEATURES:
        X[name] = rng.choice(["room-a", "room-b"], size=n)
    # Target correlated with occupied_now so the classifier has *something*
    # real to learn, but noisy enough not to be trivially separable.
    y = (X["occupied_now"] + rng.normal(0, 0.3, size=n) > 0).astype(int).to_numpy()
    return X, y


@pytest.mark.parametrize("build_pipeline", PIPELINE_BUILDERS)
def test_pipeline_fits_and_predicts_valid_probabilities(build_pipeline):
    X, y = _toy_dataset()
    pipeline = build_pipeline()
    pipeline.fit(X, y)

    proba = pipeline.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)

    preds = pipeline.predict(X)
    assert preds.shape == (len(X),)
    assert set(np.unique(preds)).issubset({0, 1})


@pytest.mark.parametrize("build_pipeline", PIPELINE_BUILDERS)
def test_pipeline_handles_unseen_room_category_at_predict_time(build_pipeline):
    X, y = _toy_dataset()
    pipeline = build_pipeline()
    pipeline.fit(X, y)

    X_new = X.iloc[[0]].copy()
    X_new["floorspaceid"] = "room-never-seen-in-training"
    # Should not raise (OneHotEncoder(handle_unknown="ignore")), and should
    # still return a valid probability.
    proba = pipeline.predict_proba(X_new)
    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0, 1] <= 1.0


def test_random_forest_pipeline_is_reproducible_with_fixed_random_state():
    """Two forests built with the same random_state must agree exactly."""
    X, y = _toy_dataset()
    pipeline_a = build_random_forest_pipeline(random_state=7, n_estimators=10)
    pipeline_b = build_random_forest_pipeline(random_state=7, n_estimators=10)
    pipeline_a.fit(X, y)
    pipeline_b.fit(X, y)

    np.testing.assert_array_equal(
        pipeline_a.predict_proba(X), pipeline_b.predict_proba(X)
    )
