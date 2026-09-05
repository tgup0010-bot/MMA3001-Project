"""The ML model: a logistic regression classifier over engineered features.

Logistic regression is chosen as the first ML model (rather than jumping
straight to a more complex model) because it gives interpretable
coefficients, trains in milliseconds on this dataset's size, and provides
a meaningful step up from the Markov baseline to measure against -- if it
can't beat the baseline, that is itself an important, honestly-reported
result (see the report's justification-of-approach section).
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from occupancy.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_logistic_pipeline(**logistic_regression_kwargs) -> Pipeline:
    """Construct the preprocessing + classifier pipeline.

    Args:
        **logistic_regression_kwargs: Passed through to
            `sklearn.linear_model.LogisticRegression` (e.g. ``C=0.5``),
            letting callers tune regularisation strength during the
            sensitivity-analysis step without editing this function.

    Returns:
        An unfitted `sklearn.pipeline.Pipeline` with:

        1. one-hot encoding of ``occupancy.features.CATEGORICAL_FEATURES``
           (currently just the room identifier), with unknown rooms at
           prediction time encoded as all-zero rather than raising; and
        2. a `sklearn.linear_model.LogisticRegression` over that plus the
           passthrough numeric features
           (``occupancy.features.NUMERIC_FEATURES``).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "room",
                OneHotEncoder(handle_unknown="ignore"),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="passthrough",  # numeric features pass through unscaled;
        # logistic regression's L2 penalty is somewhat scale-sensitive, but
        # all numeric features here are already on comparable [0,1]/[-1,1]
        # scales (occupancy fractions, sin/cos), so an explicit scaler was
        # judged unnecessary complexity for this feature set.
    )
    classifier = LogisticRegression(max_iter=1000, **logistic_regression_kwargs)
    return Pipeline([("preprocess", preprocessor), ("classify", classifier)])
