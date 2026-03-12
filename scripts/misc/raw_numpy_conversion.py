import glob
from pathlib import Path

import mne

from seegnature.utils import numpy_to_mne

# ========== VARIABLES ==========

RAW_NUMPY_DIR = "data/raw_numpy"
MNE_DIR = "data/mne"
SAMPLING_FREQ = 128

# ========== EXECUTION ==========

print(f"Looking through \033[36m{RAW_NUMPY_DIR}\033[0m for subject folders")

raw_numpy_dir_path = Path(RAW_NUMPY_DIR)
mne_dir_path = Path(MNE_DIR)

subject_folders = glob.glob(str(raw_numpy_dir_path / "*"))

for subject in subject_folders:
    subject = Path(subject)
    if not subject.is_dir():
        continue
    hypothetical_result = mne_dir_path / f"{subject.name}-epo.fif"
    print(
        f"Found subject folder \033[36m{subject.name}\033[0m, attempting conversion into \033[36m{hypothetical_result}\033[0m..."
    )
    try:
        epochs: mne.Epochs = numpy_to_mne(subject, SAMPLING_FREQ)
        epochs.save(hypothetical_result)
        print("---> \033[32mSucceeded\033[0m")
    except Exception as e:
        print(f"---> \033[31mFailed\033[0m ({e})")
