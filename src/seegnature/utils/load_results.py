import glob
from pathlib import Path

import pandas as pd


def load_results(result_dir: Path | str) -> pd.DataFrame:
    if isinstance(result_dir, str):
        result_dir = Path(result_dir)

    files = [Path(path) for path in glob.glob(str(result_dir / "*.parquet"))]
    return pd.concat({file.stem: pd.read_parquet(file) for file in files})
