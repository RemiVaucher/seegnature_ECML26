from typing import Mapping, Optional

from .evaluate_model import evaluate_model
import pandas as pd
from sklearn.base import BaseEstimator


def evaluate_model_with_subject_dict(
    model: BaseEstimator,
    subject_dict: Mapping[str, tuple],
    random_state: Optional[int] = None,
    **cross_validate_kwargs,
) -> pd.DataFrame:
    df = pd.concat(
        {
            subject: evaluate_model(
                model, X, y, random_state=random_state, **cross_validate_kwargs
            )
            for subject, (X, y) in subject_dict.items()
        },
        names=["subject"],
    )
    return df
