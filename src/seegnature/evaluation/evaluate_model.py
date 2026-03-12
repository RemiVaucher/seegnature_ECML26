from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate


def evaluate_model(
    model: BaseEstimator,
    X,
    y,
    random_state: Optional[int] = None,
    **cross_validate_kwargs,
) -> pd.DataFrame:
    kwargs = {
        "scoring": ["accuracy", "f1", "roc_auc", "d2_brier_score"],
        "cv": 5,
        "n_jobs": 5,
        "verbose": 1,
        "return_train_score": True,
    }
    kwargs.update(cross_validate_kwargs)
    scores: dict[str, np.ndarray] = cross_validate(estimator=model, X=X, y=y, **kwargs)
    test_df = pd.DataFrame(
        {"score": scores["test_score"]}
        if "test_score" in scores
        else {
            key: scores[f"test_{key}"]
            for key in kwargs["scoring"]
            if f"test_{key}" in scores
        }
    )
    train_df = None
    if kwargs["return_train_score"]:
        train_df = pd.DataFrame(
            {"score": scores["test_score"]}
            if "train_score" in scores
            else {
                key: scores[f"train_{key}"]
                for key in kwargs["scoring"]
                if f"train_{key}" in scores
            }
        )
    df: pd.DataFrame = (
        test_df
        if train_df is None
        else pd.concat({"train": train_df, "test": test_df}, axis=1)
    )
    if random_state is not None:
        df["seed"] = np.full((len(df)), random_state)
    df.index.name = "fold"
    return df
