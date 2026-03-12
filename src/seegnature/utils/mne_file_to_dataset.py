from pathlib import Path

import numpy as np

import mne


def mne_file_to_dataset(file: Path | str) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(file, str):
        file = Path(file)

    epochs = mne.read_epochs(file)

    return epochs.get_data(), epochs.metadata["label"].to_numpy()
