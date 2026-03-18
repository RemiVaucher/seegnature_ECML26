from typing import Optional

import numpy as np
from mne.decoding import CSP
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted


class CSPSVM(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        csp_nb_components: int = 10,
        svm_c: float = 0.1,
        svm_probability=True,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.csp_nb_components = csp_nb_components
        self.random_state = random_state
        self.svm_c = svm_c
        self.svm_probability = svm_probability

    def fit(self, X, y):
        check_classification_targets(y)

        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        self.pipeline_ = make_pipeline(
            CSP(self.csp_nb_components),
            SVC(
                C=self.svm_c,
                probability=self.svm_probability,
                random_state=self.random_state,
            ),
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
