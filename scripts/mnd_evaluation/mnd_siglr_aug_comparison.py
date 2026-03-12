import glob
from pathlib import Path

from sklearn.base import BaseEstimator

from seegnature.evaluation import evaluate_models_with_subject_dict
from seegnature.models import SigLR
from seegnature.utils import subject_dict_from_files

SEED = 42
RESULT_FILE = "results/mnd/aug_comparison.parquet"

# ========== MODELS ==========
DEGREE = 2

MODELS_DICT: dict[str, BaseEstimator] = {
    "mnd_siglr_no_aug": SigLR(DEGREE, random_state=SEED),
    "mnd_siglr_time_aug": SigLR(DEGREE, time_aug=True, random_state=SEED),
    "mnd_siglr_lead_lag_aug": SigLR(DEGREE, lead_lag_aug=True, random_state=SEED),
    "mnd_siglr_both_aug": SigLR(
        DEGREE, time_aug=True, lead_lag_aug=True, random_state=SEED
    ),
}

# ========== DATA ==========
FILES_GLOB = "data/mne/*"

files = glob.glob(FILES_GLOB)
if not files:
    print("No file found, exiting")

# The slice on X is important to select the moment of mental imagery.
# The operation on y is meant to allow for binary classification
subject_dict = {
    subject: (X[..., -375:], (y == "encoding_memoranda").astype(int))
    for subject, (X, y) in subject_dict_from_files(files).items()  # type: ignore
}

# ========== EVALUATION ============

df = evaluate_models_with_subject_dict(MODELS_DICT, subject_dict, SEED)

# ========== SAVING ==========

Path(RESULT_FILE).parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(RESULT_FILE)
