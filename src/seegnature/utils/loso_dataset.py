from typing import List, Mapping

import numpy as np


def loso_dataset(
    subject_dict: Mapping[str, tuple[np.ndarray, np.ndarray]],
    shuffle=False,
    random_state=None,
) -> tuple[np.ndarray, np.ndarray, List[tuple[np.ndarray, np.ndarray]]]:
    last_index = 0
    datasets = []
    labels = []
    splits = []
    for __, (X, y) in subject_dict.items():
        datasets.append(X)
        labels.append(y)
        nb_lines = X.shape[0]
        splits.append(np.arange(nb_lines) + last_index)
        last_index += nb_lines
    whole_indexing = np.arange(last_index)
    X, y, splits = (
        np.concatenate(datasets),
        np.concatenate(labels),
        [(np.delete(whole_indexing, indexes), indexes) for indexes in splits],
    )
    if shuffle:
        np.random.seed(random_state)
        rng = np.random.default_rng()
        for train, test in splits:
            rng.shuffle(train)
            rng.shuffle(test)
    return X, y, splits  # type: ignore
