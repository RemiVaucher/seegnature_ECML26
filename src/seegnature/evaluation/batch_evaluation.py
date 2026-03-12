from pathlib import Path
from typing import Optional

from sklearn.base import BaseEstimator

from .evaluate_model import evaluate_model


def batch_evaluation(
    models_dict: dict[str, BaseEstimator],
    X,
    y,
    result_dir: Path | str,
    random_state: Optional[int] = None,
    **evaluate_kwargs,
):
    if isinstance(result_dir, str):
        result_dir = Path(result_dir)

    for key, val in models_dict.items():
        df = evaluate_model(val, X, y, random_state=random_state, **evaluate_kwargs)
        df.to_parquet(result_dir / f"{key}.parquet", index=True)
