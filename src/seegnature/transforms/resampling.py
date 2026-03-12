import math

from scipy.signal import resample
from sklearn.base import BaseEstimator, TransformerMixin


class ResamplingTransform(TransformerMixin, BaseEstimator):
    def __init__(self, ratio: float, axis: int = 2) -> None:
        super().__init__()
        self.ratio = ratio
        self.axis = axis

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        return resample(X, math.ceil(self.ratio * X.shape[self.axis]), axis=self.axis)
