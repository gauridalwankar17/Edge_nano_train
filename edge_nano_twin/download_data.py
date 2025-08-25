import os
import shutil
from pathlib import Path


def download_kaggle_dataset(dataset: str = "nphantawee/pump-sensor-data", dest_dir: str = "data") -> Path:
    """Download a Kaggle dataset into dest_dir using kaggle API if available.

    Requires Kaggle credentials to be configured via `~/.kaggle/kaggle.json` or
    env vars `KAGGLE_USERNAME` and `KAGGLE_KEY`.
    """
    from kaggle import api  # type: ignore

    dest = Path(dest_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    # If already downloaded (any csv present), skip.
    if any(p.suffix.lower() == ".csv" for p in dest.glob("**/*.csv")):
        return dest

    # Download and unzip
    api.dataset_download_files(dataset, path=str(dest), unzip=True, quiet=False)

    # Some datasets create nested directory - flatten top-level if safe
    subdirs = [p for p in dest.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        inner = subdirs[0]
        # Move files up one level
        for p in inner.iterdir():
            target = dest / p.name
            if target.exists():
                continue
            shutil.move(str(p), str(target))
        try:
            inner.rmdir()
        except OSError:
            pass

    return dest

