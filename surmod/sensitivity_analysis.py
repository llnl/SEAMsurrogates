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


def plot_test_predictions(x_test, y_test, gp_model, test_function: str) -> None:
    """
    Plot GP predictions against observed test values with uncertainty intervals.

    The function computes 95% prediction intervals, interval coverage, and
    test-set RMSE, then saves a predicted-versus-observed plot as a PNG file.

    Args:
        x_test: Test input features passed to the GP model.
        y_test: Observed test target values.
        gp_model: Fitted Gaussian process model providing ``predict()``, which
            returns the predictive mean and standard deviation.
        test_function: Test-function name used in the output filename.

    Returns:
        None. The plot is saved to the sensitivity-analysis plots directory.
    """
    prediction_mean, std_dev = gp_model.predict(x_test)

    prediction_mean = np.asarray(prediction_mean).reshape(-1)
    std_dev = np.asarray(std_dev).reshape(-1)
    observed = np.asarray(y_test).reshape(-1)

    Zscore = 1.96

    lower_bounds = prediction_mean - Zscore * std_dev
    upper_bounds = prediction_mean + Zscore * std_dev
    coverage = np.mean((observed >= lower_bounds) & (observed <= upper_bounds))
    test_rmse = np.sqrt(np.mean(np.square(observed - prediction_mean)))

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure()

    plt.errorbar(
        observed,
        prediction_mean,
        yerr=Zscore * std_dev,
        fmt="o",
        capsize=5,
        color="blue",
        alpha=0.7,
    )

    max_value = max(observed.max(), upper_bounds.max()) + 0.1
    min_value = min(observed.min(), lower_bounds.min()) - 0.1
    plt.plot([min_value, max_value], [min_value, max_value], "k-", linewidth=2)

    plt.ylabel("Predicted", fontsize=14)
    plt.xlabel("Observed", fontsize=14)
    plt.text(
        0.5,
        -0.15,
        f"RMSE: {test_rmse:.4f}, Coverage: {coverage:.2%}",
        ha="center",
        fontsize=14,
        transform=plt.gca().transAxes,
    )
    plt.tight_layout()

    plot_dir = (
        Path(__file__).parent.parent / "scripts" / "sensitivity_analysis" / "plots"
    )
    plot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    path_to_plot = plot_dir / f"test_predictions_{test_function}_{timestamp}.png"
    plt.savefig(path_to_plot, bbox_inches="tight")
    print(f"Figure saved to {path_to_plot}")


def sobol_plot(
    S1: Sequence[float],
    ST: Sequence[float],
    variables: list[str],
    S1_conf: Sequence[float],
    ST_conf: Sequence[float],
    test_function: str,
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

    plot_dir = (
        Path(__file__).parent.parent / "scripts" / "sensitivity_analysis" / "plots"
    )
    plot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    path_to_plot = plot_dir / f"sensitivity_{test_function}_{timestamp}.png"
    plt.savefig(path_to_plot, bbox_inches="tight")
    print(f"Figure saved to {path_to_plot}")
