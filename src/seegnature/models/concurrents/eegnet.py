from typing import Optional

import numpy as np
import torch
from braindecode import EEGClassifier
from braindecode.models import EEGNet
from braindecode.util import set_random_seeds
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from skorch.dataset import ValidSplit

from ...transforms import ResamplingTransform


class EEGNetClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        lr: float = 0.000625,
        max_epochs=500,
        batch_size=64,
        data_freq: int = 128,
        random_state: Optional[int] = None,
        verbose: int = 1,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.data_freq = data_freq
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        check_classification_targets(y)

        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        set_random_seeds(seed=self.random_state, cuda=torch.cuda.is_available())

        device = "cuda" if torch.cuda.is_available() else "cpu"

        steps = [
            EEGClassifier(
                module=EEGNet,
                optimizer=torch.optim.Adam,
                lr=self.lr,
                max_epochs=self.max_epochs,
                batch_size=self.batch_size,
                train_split=ValidSplit(0.2, random_state=self.random_state),
                device=device,
                verbose=self.verbose,
            )
        ]

        freq_ratio = 128.0 / self.data_freq
        if freq_ratio != 1.0:
            steps.insert(0, ResamplingTransform(freq_ratio))

        self.pipeline_ = make_pipeline(*steps)

        self.pipeline_.fit(X, y)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.label_encoder_.inverse_transform(self.pipeline_.predict(X))

    def predict_proba(self, X):
        check_is_fitted(self)
        return self.pipeline_.predict_proba(X)
