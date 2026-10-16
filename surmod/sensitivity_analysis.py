"""
Sensitivity analysis for surrogate models.
"""

from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from surmod.test_functions import (
    get_input_spec,
)


def load_test_settings(
    test_function: str,
) -> tuple[int, Callable[[np.ndarray, float, float, float], np.ndarray]]:
    """
    Load the test function and its input dimension for simulating data.

    Args:
        test_function: Name of the test function to load.
            Must be one of 'parabola', 'otlcircuit', 'wingweight', or 'piston'.

    Returns:
        A tuple containing the input dimension and the function used to
        simulate data.

    Raises:
        ValueError: If the provided test_function is not recognized.
    """
    out_dim, test_function, _ = get_input_spec(test_function)
    return out_dim, test_function


def simulate_data(
    test_function: str,
    n_train: int,
    n_test: int,
    b1: float,
    b2: float,
    b12: float,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate training and testing data from a selected test function.

    Args:
        test_function: Name of the test function to use.
            Must be one of 'parabola', 'otlcircuit', 'wingweight', or 'piston'.
        n_train: Number of training samples to generate.
        n_test: Number of testing samples to generate.
        b1: First coefficient parameter for the test function.
        b2: Second coefficient parameter for the test function.
        b12: Interaction coefficient parameter for the test function.
        seed: Random seed for reproducibility.

    Returns:
        A tuple ``(x_train, x_test, y_train, y_test)`` containing the training
        and testing inputs and outputs.
    """
    # Set-up simulation
    n_total = n_train + n_test
    out_dim, test_function_callable, bounds_list = get_input_spec(test_function)
    bounds = np.array(bounds_list, dtype=float)
    bounds_low = bounds[:, 0]
    bounds_high = bounds[:, 1]

    # Sample random data from test function
    rng = np.random.default_rng(seed)
    x_data = rng.uniform(bounds_low, bounds_high, size=(n_total, out_dim))
    if test_function == "parabola":
        y_data = test_function_callable(x_data, b1, b2, b12)
    else:
        y_data = test_function_callable(x_data)

    # Split data into training and testing sets
    x_train = x_data.copy()[:n_train]
    y_train = np.asarray(y_data.copy()[:n_train]).reshape(-1)

    x_test = x_data.copy()[n_train:]
    y_test = np.asarray(y_data.copy()[n_train:]).reshape(-1)

    return x_train, x_test, y_train, y_test


def sobol_plot(
    S1: Sequence[float],
    ST: Sequence[float],
    variables: list[str],
    S1_conf: Sequence[float],
    ST_conf: Sequence[float],
    test_function: str,
    plots_dir: Path = Path("plots"),
):
    """
    Plot first- and total-order Sobol sensitivity indices with confidence
    intervals and saves the figure.

    Args:
        S1: First-order sensitivity indices for each variable.
        ST: Total-order sensitivity indices for each variable.
        variables: Variable names.
        S1_conf: Confidence intervals for first-order indices.
        ST_conf: Confidence intervals for total-order indices.
        test_function: Test-function name used in the saved plot filename.
        plots_dir: Directory where plots are saved.

    Returns:
        None. This function saves the visualization to disk.
    """
    # Define colors for each variable
    colors = sns.color_palette("husl", len(variables))

    # Create a figure with subplots
    _fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # First Order Sensitivity Plot
    axes[0].bar(variables, S1, yerr=S1_conf, color=colors, alpha=0.7)
    axes[0].set_title("First Order Sensitivity Indices")
    axes[0].set_ylabel("Sensitivity Index")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    # Total Order Sensitivity Plot
    axes[1].bar(variables, ST, yerr=ST_conf, color=colors, alpha=0.7)
    axes[1].set_title("Total Order Sensitivity Indices")
    axes[1].set_ylabel("Sensitivity Index")
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    plots_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    path_to_plot = plots_dir / f"sensitivity_{test_function}_{timestamp}.png"
    plt.savefig(path_to_plot, bbox_inches="tight")
    print(f"Figure saved to {path_to_plot}")
