import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from ..transforms import NormalizationTransform, SignatureTransform


class SigLR(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        degree,
        time_aug=False,
        lead_lag_aug=False,
        l1_ratio=0.0,
        normalization=True,
        random_state=None,
    ) -> None:
        super().__init__()
        self.degree = degree
        self.time_aug = time_aug
        self.lead_lag_aug = lead_lag_aug
        self.l1_ratio = l1_ratio
        self.normalization = normalization
        self.random_state = random_state

    def fit(self, X, y):
        check_classification_targets(y)

        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        steps = [
            SignatureTransform(
                degree=self.degree,
                time_aug=self.time_aug,
                lead_lag_aug=self.lead_lag_aug,
            ),
            LogisticRegression(
                l1_ratio=self.l1_ratio,
                solver="lbfgs" if self.l1_ratio == 0.0 else "saga",
                random_state=self.random_state,
            ),
        ]

        if self.normalization:
            steps.insert(0, NormalizationTransform())

        self.pipeline_ = make_pipeline(*steps)

        self.pipeline_.fit(X, y_encoded)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.label_encoder_.inverse_transform(self.pipeline_.predict(X))

    def predict_proba(self, X):
        check_is_fitted(self)
        return self.pipeline_.predict_proba(X)
