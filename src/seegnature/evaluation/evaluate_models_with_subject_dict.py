from typing import Mapping, Optional

from sklearn.base import BaseEstimator
import pandas as pd

from .evaluate_model_with_subject_dict import evaluate_model_with_subject_dict


def evaluate_models_with_subject_dict(
    models: Mapping[str, BaseEstimator],
    subject_dict: Mapping[str, tuple],
    random_state: Optional[int] = None,
    **cross_validate_kwargs,
):
    df = pd.concat(
        {
            name: evaluate_model_with_subject_dict(
                model, subject_dict, random_state=random_state, **cross_validate_kwargs
            )
            for name, model in models.items()
        },
        names=["model"],
    )
    return df
