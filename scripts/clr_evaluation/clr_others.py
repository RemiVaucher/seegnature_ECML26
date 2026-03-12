import glob
from pathlib import Path

from sklearn.base import BaseEstimator

from seegnature.evaluation import evaluate_models_with_subject_dict
from seegnature.models import SigLR
from seegnature.models.concurrents import (
    ATCNetClassifier,
    EEGConformerClassifier,
    EEGNetClassifier,
)
from seegnature.utils import subject_dict_from_files

SEED = 42
RESULT_FILE = "results/clr/model_comparison.parquet"

# ========== MODELS ==========
DEGREE = 2

MODELS_DICT: dict[str, BaseEstimator] = {
    "clr_siglr": SigLR(DEGREE, time_aug=True, lead_lag_aug=True, random_state=SEED),
    "clr_eegnet": EEGNetClassifier(data_freq=128, random_state=SEED),
    "clr_eeg_conformer": EEGConformerClassifier(data_freq=128, random_state=SEED),
    "clr_atcnet": ATCNetClassifier(data_freq=128, random_state=SEED),
}

# ========== DATA ==========
FILES_GLOB = "data/mne/*"

files = glob.glob(FILES_GLOB)
if not files:
    print("No file found, exiting")

# The slice on X is important to select the moment of mental imagery.
# The operation on y is meant to allow for binary classification
subject_dict = {
    subject: (X[..., 61:124], (y == "encoding_memoranda").astype(int))
    for subject, (X, y) in subject_dict_from_files(files).items()  # type: ignore
}

# ========== EVALUATION ============

df = evaluate_models_with_subject_dict(MODELS_DICT, subject_dict, SEED)

# ========== SAVING ==========

Path(RESULT_FILE).parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(RESULT_FILE)
