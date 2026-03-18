import numpy as np
from mne.decoding import CSP
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted


class CSPLDA(ClassifierMixin, BaseEstimator):
    def __init__(self, csp_nb_components: int = 10) -> None:
        super().__init__()
        self.csp_nb_components = csp_nb_components

    def fit(self, X, y):
        check_classification_targets(y)

        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        self.pipeline_ = make_pipeline(
            CSP(self.csp_nb_components), LinearDiscriminantAnalysis()
        )

        self.pipeline_.fit(X, y_encoded)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.label_encoder_.inverse_transform(self.pipeline_.predict(X))

    def predict_proba(self, X):
        check_is_fitted(self)
        return self.pipeline_.predict_proba(X)
