from pysiglib import signature
from sklearn.base import BaseEstimator, TransformerMixin


class SignatureTransform(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        degree: int,
        time_aug: bool = False,
        lead_lag_aug: bool = False,
    ) -> None:
        super().__init__()
        self.degree = degree
        self.time_aug = time_aug
        self.lead_lag_aug = lead_lag_aug

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if X.ndim == 2:
            X = X[None, :]
        if X.ndim != 3:
            raise ValueError(
                "SignatureTransform takes values following the format (dimensions, length) or (batch, dimensions, length)"
            )
        X = X.swapaxes(-1, -2)
        return signature(
            X,
            degree=self.degree,
            time_aug=self.time_aug,
            lead_lag=self.lead_lag_aug,
        )
