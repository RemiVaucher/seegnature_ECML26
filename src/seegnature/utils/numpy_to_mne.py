from pathlib import Path

import mne
import numpy as np
import pandas as pd


def numpy_to_mne(subject_dir: Path | str, sfreq: float = 128) -> mne.Epochs:
    if isinstance(subject_dir, str):
        subject_dir = Path(subject_dir)

    names = np.load(
        subject_dir / f"{subject_dir.name}-channel_names.npy", allow_pickle=True
    )
    pos = np.load(
        subject_dir / f"{subject_dir.name}-channel_coords.npy", allow_pickle=True
    )
    epochs = np.load(
        subject_dir / f"{subject_dir.name}_array-epochs.npy", allow_pickle=True
    )
    labels = np.load(subject_dir / f"{subject_dir.name}-labels.npy", allow_pickle=True)

    channel_pos = {key: pos for key, pos in zip(names, pos)}
    montage = mne.channels.make_dig_montage(ch_pos=channel_pos, coord_frame="head")

    info = mne.create_info(
        ch_names=names.tolist(),
        sfreq=sfreq,
        ch_types="eeg",
    )
    info.set_montage(montage)

    metadata = pd.DataFrame({"label": labels})

    return mne.EpochsArray(epochs, info=info, metadata=metadata)
