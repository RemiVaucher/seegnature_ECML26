from pathlib import Path
from typing import List

from .mne_file_to_dataset import mne_file_to_dataset


def subject_dict_from_files(files: List[Path | str]):
    file_paths: List[Path] = [Path(file) for file in files]
    result = {}
    for file in file_paths:
        X, y = mne_file_to_dataset(file)
        result["-".join(file.stem.split("-")[:-1])] = (X, y)
    return result
