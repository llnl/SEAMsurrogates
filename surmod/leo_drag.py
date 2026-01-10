"""
Functions for JAG ICF data.
"""

from typing import Optional
import warnings

import numpy as np
import pandas as pd

from scipy.spatial import cKDTree  # type: ignore
from scipy.stats import qmc

from sklearn.model_selection import train_test_split


def load_data(
    path_to_csv: str = "../../data/hst_10k_sample_O2.csv",
    n_samples: int = 10000,
    random: bool = True,
    seed: Optional[int] = None,
    scale_data: bool = False,
) -> pd.DataFrame:
    """
    Load a subset of the dataset from a CSV file.

    Args:
        path_to_csv (str): Path to the CSV file.
        n_samples (int): Number of rows to load. Defaults to dataset size.
        random (bool): If True, select rows randomly. Otherwise, select the
            first n_samples rows.
        seed (int or None): Random seed for reproducibility (used if random is
            True). Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with columns ["x0", "x1", "x2", "x3", "x4", "y"].
    """
    df = pd.read_csv(path_to_csv)
    #df.columns = ["x0", "x1", "x2", "x3", "x4", "y"]

    # Check and warn if n_samples is too large
    if n_samples > len(df):
        warnings.warn(
            "n_samples is greater than the number of rows in the dataset "
            f"({len(df)}). Using the full dataset instead."
        )
        n_samples = len(df)

    # Select rows
    if random:
        print(
            f"Selecting {n_samples} samples at random from the JAG_10k dataset (seed={seed}).\n"
        )
        df = df.sample(n=n_samples, random_state=seed)
    else:
        print(f"Selecting the first {n_samples} samples from the JAG_10k dataset.\n")
        df = df.iloc[:n_samples]

    return df


def split_data(df: pd.DataFrame, LHD: bool = False, n_train: int = 100, seed: int = 42):
    """
    Split data into train and test sets using either Latin Hypercube Design
    (LHD) or random split.

    Args:
        df (pd.DataFrame): Input DataFrame where the last column is the label.
        LHD (bool): If True, use Latin Hypercube Design for selecting training
            samples. If False, use random split. Defaults to False.
        n_train (int): Number of training samples to select. Defaults to 100.
        seed (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            x_train: Training features array.
            x_test: Testing features array.
            y_train: Training labels array (reshaped to column vector).
            y_test: Testing labels array (reshaped to column vector).

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
            "Using n_train closest points to Latin Hypercube Design for"
            " training points.\n"
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

        def query_unique(tree, small_data):
            used_indices = set()
            unique_indices = []
            unique_distances = []

            for point in small_data:
                distances, indices = tree.query(point, k=50)
                for dist, idx in zip(distances, indices):
                    if idx not in used_indices:
                        used_indices.add(idx)
                        unique_indices.append(idx)
                        unique_distances.append(dist)
                        break
            return np.array(unique_distances), np.array(unique_indices)

        # Query for unique nearest neighbors
        distances, index = query_unique(tree, x_lhd)

        x_train = x[index, :]
        y_train = y[index].reshape(-1, 1)
        mask = np.ones(n_total, dtype=bool)
        mask[index] = False
        x_test = x[mask, :]
        y_test = y[mask].reshape(-1, 1)
    else:
        # Standard random split
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
    print(f"x_test shape: {x_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape} \n")

    return x_train, x_test, y_train, y_test
