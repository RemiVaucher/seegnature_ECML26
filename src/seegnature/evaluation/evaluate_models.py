from typing import Mapping, Optional

import pandas as pd
from sklearn.base import BaseEstimator


from .evaluate_model import evaluate_model


def evaluate_models(
    models: Mapping[str, BaseEstimator],
    X,
    y,
    random_state: Optional[int] = None,
    **cross_validate_kwargs,
) -> pd.DataFrame:
    df = pd.concat(
        {
            name: evaluate_model(
                model, X, y, random_state=random_state, **cross_validate_kwargs
            )
            for name, model in models.items()
        },
        names=["model"],
    )
    return df
