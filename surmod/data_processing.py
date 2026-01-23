"""
General data loading and splitting utilities for JAG and borehole datasets.

JAG:
    - 5 inputs, 1 output
    - default path: "../../data/JAG_10k.csv"

Borehole:
    - 8 inputs, 1 output
    - default path: "../../data/borehole_10k.csv"
"""

from typing import Optional, Tuple
import warnings
import os

import numpy as np
import pandas as pd

from scipy.spatial import cKDTree  # type: ignore
from scipy.stats import qmc

from sklearn.model_selection import train_test_split


# Dataset configuration
DATASET_CONFIG = {
    "JAG": {
        "path": "../../data/JAG_10k.csv",
        "n_inputs": 5,
        "n_outputs": 1,
    },
    "borehole": {
        "path": "../../data/borehole_10k.csv",
        "n_inputs": 8,
        "n_outputs": 1,
    },
}


# Loading data

def load_data(
    dataset: str = "JAG",
    path_to_csv: Optional[str] = None,
    n_samples: int = 10000,
    random: bool = True,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Load a subset of a dataset from CSV.

    Assumes:
        - CSV has exactly n_inputs + n_outputs columns
        - No header, or any header will be ignored and replaced

    Args:
        dataset: Which dataset to load, "JAG" or "borehole".
        path_to_csv: Optional explicit path; if None, use default from config.
        n_samples: Number of rows to load.
        random: If True, select rows randomly; else select first n_samples rows.
        seed: Random seed for reproducibility (used if random is True).

    Returns:
        pd.DataFrame:
            For JAG:
                columns: [x0, x1, x2, x3, x4, y]
            For borehole:
                columns: [rw, r, Tu, Hu, Tl, Hl, L, Kw, y]
    """
    if dataset not in DATASET_CONFIG:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. "
            f"Supported: {list(DATASET_CONFIG.keys())}"
        )

    cfg = DATASET_CONFIG[dataset]

    if path_to_csv is None:
        path_to_csv = cfg["path"]

    if not os.path.isfile(path_to_csv): # type: ignore
        raise FileNotFoundError(f"CSV file not found at: {path_to_csv}")

    df = pd.read_csv(path_to_csv) # type: ignore

    if dataset == "JAG":
        df.columns = ["x0", "x1", "x2", "x3", "x4", "y"]
    elif dataset == "borehole":
        df.columns = ["rw", "r", "Tu", "Hu", "Tl", "Hl", "L", "Kw", "y"]

    # Check and warn if n_samples is too large
    if n_samples > len(df):
        warnings.warn(
            "n_samples is greater than the number of rows in the dataset "
            f"({len(df)}). Using the full 10k dataset instead."
        )
        n_samples = len(df)

    # Select rows
    if random:
        print(
            f"Selecting {n_samples} samples at random from the {dataset} dataset (seed={seed}).\n"
        )
        df = df.sample(n=n_samples, random_state=seed)
    else:
        print(f"Selecting the first {n_samples} samples from the {dataset} dataset.\n")
        df = df.iloc[:n_samples]

    return df


# Splitting data

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
        y_train: Training labels array (column vector).
        y_test: Testing labels array (column vector).

    Raises:
        ValueError: If n_train is greater than the total number of samples in df.
    """
    # Split the data into features (x) and labels (y)
    x = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    n_total, k = x.shape

    # Ensure n_train is not greater than total_samples
    if n_train > n_total:
        raise ValueError(
            f"n_train cannot be greater than the total number of samples "
            f"({n_total})."
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
        for i in range(k):
            x_lhd[:, i] = x_lhd[:, i] * (np.max(x[:, i]) - np.min(x[:, i])) + np.min(
                x[:, i]
            )

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

        x_train = x[index, :]
        y_train = y[index].reshape(-1, 1)
        mask = np.ones(n_total, dtype=bool)
        mask[index] = False
        x_test = x[mask, :]
        y_test = y[mask].reshape(-1, 1)
    else:
        # Standard random split with exact n_train samples
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            train_size=n_train,
            test_size=None,
            random_state=seed,
        )
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

    print(f"x_train shape: {x_train.shape}")
    print(f"x_test shape:  {x_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape:  {y_test.shape}\n")

    return x_train, x_test, y_train, y_test


# Convenience wrapper

def load_and_split(
    dataset: str = "JAG",
    path_to_csv: Optional[str] = None,
    n_samples: int = 10000,
    random_rows: bool = True,
    seed: int = 42,
    LHD: bool = False,
    n_train: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function: load dataset, then split into train and test.

    Args:
        dataset: "JAG" or "borehole".
        path_to_csv: Optional explicit path, overrides default.
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
        path_to_csv=path_to_csv,
        n_samples=n_samples,
        random=random_rows,
        seed=seed,
    )

    return split_data(df, LHD=LHD, n_train=n_train, seed=seed)