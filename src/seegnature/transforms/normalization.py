import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class NormalizationTransform(TransformerMixin, BaseEstimator):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        return (X - X.mean(axis=2)[..., None]) / np.abs(X).max(axis=2)[..., None]
