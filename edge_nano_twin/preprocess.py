from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


LABEL_ALIASES = {
    "NORMAL": "NORMAL",
    "OK": "NORMAL",
    "HEALTHY": "NORMAL",
    "RECOVERING": "WARNING",
    "ALERT": "WARNING",
    "WARNING": "WARNING",
    "BROKEN": "FAILURE",
    "FAIL": "FAILURE",
    "FAILURE": "FAILURE",
}


def _find_label_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["machine_status", "status", "label", "target", "y"]
    for c in candidates:
        if c in df.columns:
            return c
    # Try best-effort guess: any object column with small number of uniques
    for c in df.columns:
        if df[c].dtype == "object" and df[c].nunique(dropna=True) <= 10:
            return c
    return None


def _normalize_label(value) -> Optional[str]:
    if value is None:
        return None
    key = str(value).strip().upper()
    return LABEL_ALIASES.get(key, None)


def load_dataframe(data_dir: str | Path) -> Tuple[pd.DataFrame, List[str]]:
    """Load the main CSV from the dataset and return (df, feature_columns).

    - Attempts to find a CSV with sensor-like columns
    - Maps labels to {NORMAL, WARNING, FAILURE}
    - Sorts by timestamp if present
    """
    data_dir = Path(data_dir)
    csv_files = list(data_dir.glob("**/*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    # Prefer files with 'sensor' or 'pump' in the name
    csv_files.sort(key=lambda p: ("sensor" not in p.name.lower(), "pump" not in p.name.lower(), p.name))
    df = pd.read_csv(csv_files[0])

    # Timestamp sort if present
    time_cols = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
    if time_cols:
        try:
            df[time_cols[0]] = pd.to_datetime(df[time_cols[0]], errors="coerce")
            df = df.sort_values(time_cols[0]).reset_index(drop=True)
        except Exception:
            pass

    label_col = _find_label_column(df)
    if label_col is None:
        raise ValueError("Could not identify a label column in the dataset.")

    df["label_raw"] = df[label_col]
    df["label"] = df[label_col].map(_normalize_label)
    df = df.dropna(subset=["label"]).reset_index(drop=True)

    # feature columns are numeric (float/int) excluding label columns
    feature_cols = [
        c
        for c in df.columns
        if c not in {label_col, "label", "label_raw"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(feature_cols) == 0:
        raise ValueError("No numeric feature columns found in dataset.")

    return df[[*feature_cols, "label"]], feature_cols


def make_windows(
    data: np.ndarray,
    labels: Sequence[str],
    window_size: int = 256,
    step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding windows and majority label per window.

    data: (N, F) array
    labels: length-N list of class strings
    returns X: (M, window_size, F), y: (M,) class indices
    """
    num_samples, num_features = data.shape
    if num_samples < window_size:
        raise ValueError("Not enough samples to form a single window.")

    label_map = {"NORMAL": 0, "WARNING": 1, "FAILURE": 2}

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    for start in range(0, num_samples - window_size + 1, step):
        end = start + window_size
        window = data[start:end]
        window_labels = labels[start:end]
        # majority label in the window
        vals, counts = np.unique(window_labels, return_counts=True)
        majority = vals[np.argmax(counts)]
        X_list.append(window)
        y_list.append(label_map[majority])

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def zscore_per_window(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply z-score per window, per feature.

    X: (M, T, F)
    """
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std = np.maximum(std, eps)
    return (X - mean) / std


def balance_classes(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Oversample minority classes to match the maximum class count."""
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_balanced: List[np.ndarray] = []
    y_balanced: List[np.ndarray] = []
    for cls in classes:
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            continue
        reps = int(np.ceil(max_count / len(idx)))
        idx_sampled = np.tile(idx, reps)[:max_count]
        rng.shuffle(idx_sampled)
        X_balanced.append(X[idx_sampled])
        y_balanced.append(y[idx_sampled])

    Xb = np.concatenate(X_balanced, axis=0)
    yb = np.concatenate(y_balanced, axis=0)

    # Shuffle
    perm = rng.permutation(len(yb))
    return Xb[perm], yb[perm]


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 3
        assert y.ndim == 1
        self.X = X.astype(np.float32)  # (N, T, F)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int):
        window = self.X[idx]  # (T, F)
        # Rearrange to (C=F, T)
        window_ch_first = np.transpose(window, (1, 0))
        x = torch.from_numpy(window_ch_first)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

