"""
Data loading and splitting utilities.

Supported datasets: JAG, borehole, hst_H
See DATASET_CONFIG for dataset specifications (paths, dimensions, column names).
"""

from typing import Tuple
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.spatial import cKDTree  # type: ignore
from scipy.stats import qmc

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Paths relative to this file's location
_MODULE_DIR = Path(__file__).parent
_DATA_DIR = _MODULE_DIR.parent / "data"

# Dataset configuration
DATASET_CONFIG = {
    "JAG": {
        "path": _DATA_DIR / "JAG_10k.csv",
        "n_inputs": 5,
        "n_outputs": 1,
        "columns": ["x1", "x2", "x3", "x4", "x5", "y"],
    },
    "borehole": {
        "path": _DATA_DIR / "borehole_10k.csv",
        "n_inputs": 8,
        "n_outputs": 1,
        "columns": ["rw", "r", "Tu", "Hu", "Tl", "Hl", "L", "Kw", "y"],
    },
    "hst_H": {
        "path": _DATA_DIR / "hst_H_10k.csv",
        "n_inputs": 8,
        "n_outputs": 1,
        "columns": [
            "Umag",
            "Ts",
            "Ta",
            "alphan",
            "sigmat",
            "theta",
            "phi",
            "panang",
            "Cd",
        ],
    },
}


def load_data(
    dataset: str = "JAG",
    n_samples: int = 10000,
    random: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load a subset of a dataset from CSV.

    Assumes:
        - CSV has exactly n_inputs + n_outputs columns
        - No header, or any header will be ignored and replaced

    Args:
        dataset: Dataset name (see DATASET_CONFIG for supported options).
        n_samples: Number of rows to load.
        random: If True, select rows randomly; else select first n_samples rows.
        seed: Random seed for reproducibility (used if random is True).

    Returns:
        pd.DataFrame with input features and output column for the selected dataset.
    """
    if dataset not in DATASET_CONFIG:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. Supported: {list(DATASET_CONFIG.keys())}"
        )

    cfg = DATASET_CONFIG[dataset]
    csv_path = cfg["path"]

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    df = pd.read_csv(csv_path)  # type: ignore
    df.columns = cfg["columns"]

    # Check and warn if n_samples is too large
    if n_samples > len(df):
        warnings.warn(
            "n_samples is greater than the number of rows in the dataset "
            f"({len(df)}). Using the full dataset instead."
        )
        n_samples = len(df)

    # Select rows
    if random:
        print(f"Selecting {n_samples} random samples from the {dataset} dataset.\n")
        df = df.sample(n=n_samples, random_state=seed)
    else:
        print(f"Selecting the first {n_samples} samples from the {dataset} dataset.\n")
        df = df.iloc[:n_samples]

    return df


def split_data(
    df: pd.DataFrame,
    LHD: bool = False,
    n_train: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets using either Latin Hypercube Design
    (LHD) or random split.

    Args:
        df: Input DataFrame where the last column is the output.
        LHD: If True, use Latin Hypercube Design for selecting training
            samples; if False, use random split.
        n_train: Number of training samples to select.
        seed: Random seed for reproducibility.

    Returns:
        x_train: Training features array.
        x_test: Testing features array.
        y_train: Training labels array (1D).
        y_test: Testing labels array (1D).

    Raises:
        ValueError: If n_train is greater than the total number of samples in df.
    """
    # Split the data into features (x) and labels (y)
    data = df.to_numpy()
    x = data[:, :-1]
    y = data[:, -1]
    n_total, k = x.shape

    # Ensure n_train is not greater than total_samples
    if n_train > n_total:
        raise ValueError(
            f"n_train cannot be greater than the total number of samples ({n_total})."
        )

    if LHD:
        print(
            "Using n_train closest points to Latin Hypercube Design for "
            "training points.\n"
        )
        # Latin Hypercube Sampling for n_train points in k dimensions
        LHD_gen = qmc.LatinHypercube(d=k, seed=seed)  # type: ignore
        x_lhd = LHD_gen.random(n=n_train)

        # Scale LHD points to the range of x
        x_min = x.min(axis=0)
        x_range = x.ptp(axis=0)  # ptp = peak-to-peak = max - min
        x_lhd = x_lhd * x_range + x_min

        # Build KDTree for nearest neighbor search
        tree = cKDTree(x)

        def query_unique(tree_obj, small_data):
            used_indices = set()
            unique_indices = []
            unique_distances = []

            for point in small_data:
                distances, indices = tree_obj.query(point, k=50)
                for dist, idx in zip(distances, indices):
                    if idx not in used_indices:
                        used_indices.add(idx)
                        unique_indices.append(idx)
                        unique_distances.append(dist)
                        break
            return np.array(unique_distances), np.array(unique_indices)

        # Query for unique nearest neighbors
        _, index = query_unique(tree, x_lhd)

        x_train = x[index]
        y_train = y[index].reshape(-1)
        mask = np.ones(n_total, dtype=bool)
        mask[index] = False
        x_test = x[mask]
        y_test = y[mask].reshape(-1)
    else:
        # Standard random split with exact n_train samples
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            train_size=n_train,
            test_size=None,
            random_state=seed,
        )
        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)

    print(f"x_train shape: {x_train.shape}")
    print(f"x_test shape:  {x_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape:  {y_test.shape}\n")

    return x_train, x_test, y_train, y_test


def load_and_split(
    dataset: str = "JAG",
    n_samples: int = 10000,
    random_rows: bool = True,
    seed: int = 42,
    LHD: bool = False,
    n_train: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function: load dataset, then split into train and test.

    Args:
        dataset: Dataset name (see DATASET_CONFIG for supported options).
        n_samples: Number of samples to load from CSV.
        random_rows: Randomly choose rows or take first n_samples.
        seed: Random seed used for row sampling and splitting.
        LHD: Use LHD based train selection if True.
        n_train: Number of training samples.

    Returns:
        x_train, x_test, y_train, y_test
    """
    df = load_data(
        dataset=dataset,
        n_samples=n_samples,
        random=random_rows,
        seed=seed,
    )

    return split_data(df, LHD=LHD, n_train=n_train, seed=seed)


def normalize_data(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features and targets using StandardScaler (zero mean, unit variance).

    Uses one scaler for all X features and one scaler for y targets.
    Fits scalers on training data and applies to both train and test sets.

    Args:
        x_train: Training features array of shape (n_train, n_features).
        x_test: Testing features array of shape (n_test, n_features).
        y_train: Training labels array of shape (n_train,).
        y_test: Testing labels array of shape (n_test,).

    Returns:
        x_train_norm: Normalized training features.
        x_test_norm: Normalized testing features.
        y_train_norm: Normalized training labels.
        y_test_norm: Normalized testing labels.
    """
    # One scaler for all X features
    x_scaler = StandardScaler()
    x_train_norm = x_scaler.fit_transform(x_train)
    x_test_norm = x_scaler.transform(x_test)

    # One scaler for y target
    y_scaler = StandardScaler()
    y_train_reshaped = y_train.reshape(-1, 1)
    y_test_reshaped = y_test.reshape(-1, 1)
    y_train_norm = y_scaler.fit_transform(y_train_reshaped).reshape(-1)
    y_test_norm = y_scaler.transform(y_test_reshaped).reshape(-1)

    return x_train_norm, x_test_norm, y_train_norm, y_test_norm
