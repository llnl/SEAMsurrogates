"""
Utility functions for simulating, evaluating, and visualizing surrogate modeling
sensitivity analysis experiments using benchmark engineering test problems.
"""

import copy
import os
from typing import Tuple, Callable, List, Sequence
import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error as mse

from surmod.test_functions import (
    parabola,
    otlcircuit,
    wingweight,
    piston,
    borehole,
)


def load_test_settings(
    objective_function: str,
) -> Tuple[int, Callable[[np.ndarray, float, float, float], np.ndarray]]:
    """
    Load the test function and its input dimension for simulating data.

    Args:
        objective_function (str): Name of the objective function to load.
            Must be one of 'parabola', 'otlcircuit', 'wingweight', or 'piston'.

    Returns:
        Tuple[int, Callable[[np.ndarray, float, float, float], np.ndarray]]:
            A tuple containing:
                - out_dim (int): The number of input dimensions for the selected
                    test function.
                - test_function (Callable): The test function to simulate data
                    from.

    Raises:
        ValueError: If the provided objective_function is not recognized.
    """
    if objective_function == "parabola":
        out_dim = 2
        test_function = copy.deepcopy(parabola)
    elif objective_function == "otlcircuit":
        out_dim = 6
        test_function = copy.deepcopy(otlcircuit)
    elif objective_function == "wingweight":
        out_dim = 10
        test_function = copy.deepcopy(wingweight)
    elif objective_function == "piston":
        out_dim = 7
        test_function = copy.deepcopy(piston)
    elif objective_function == "borehole":
        out_dim = 8
        test_function = copy.deepcopy(borehole)
    else:
        raise ValueError(
            f"Test function '{objective_function}' not found. "
            "Choose from 'parabola', 'otlcircuit', 'wingweight', 'piston', or 'borehole'."
        )
    return out_dim, test_function


def simulate_data(
    objective_function: str,
    num_train: int,
    num_test: int,
    b1: float,
    b2: float,
    b12: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate training and testing data from a selected test function.

    Args:
        objective_function (str): Name of the objective function to use.
            Must be one of 'parabola', 'otlcircuit', 'wingweight', or 'piston'.
        num_train (int): Number of training samples to generate.
        num_test (int): Number of testing samples to generate.
        b1 (float): First coefficient parameter for the test function.
        b2 (float): Second coefficient parameter for the test function.
        b12 (float): Interaction coefficient parameter for the test function.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - x_train (np.ndarray): Training input data of shape (num_train, input_dim).
            - x_test (np.ndarray): Testing input data of shape (num_test, input_dim).
            - y_train (np.ndarray): Training output data.
            - y_test (np.ndarray): Testing output data.
    """
    # Set-up simulation
    num_total = num_train + num_test
    out_dim, test_function = load_test_settings(objective_function)

    # Sample random data from test function
    np.random.seed(1)
    x_data = np.random.uniform(0, 1, size=(num_total, out_dim))
    y_data = test_function(x_data, b1, b2, b12)

    # Split data into training and testing sets
    x_train = x_data.copy()[:num_train]
    y_train = y_data.copy()[:num_train]

    x_test = x_data.copy()[num_train:]
    y_test = y_data.copy()[num_train:]

    return x_train, x_test, y_train, y_test


def plot_test_predictions(x_test, y_test, gp_model, objective_function: str) -> None:
    """
    Plot test set predictions vs. ground truth for a Gaussian Process model.

    Args:
        x_test (np.ndarray): Test input data of shape (num_test, input_dim).
        y_test (np.ndarray): True observed outputs for the test set.
        gp_model (Any): Trained Gaussian Process model with a predict method.
        objective_function (str): Name of the objective function, used for plot
            file naming.

    Returns:
        None, used for visualization purposes only.
    """
    prediction_mean, std_dev = gp_model.predict(x_test, return_std=True)
    observed = y_test
    Zscore = 1.96

    # Calculate Coverage
    lower_bounds = prediction_mean - Zscore * std_dev
    upper_bounds = prediction_mean + Zscore * std_dev
    coverage = np.mean((observed >= lower_bounds) & (observed <= upper_bounds))

    # Calculate RMSE
    test_rmse = np.sqrt(mse(observed, prediction_mean))

    # Set Seaborn style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Create a plot
    plt.figure()

    # Plot truth vs prediction mean with error bars
    plt.errorbar(
        observed,
        prediction_mean,
        yerr=Zscore * std_dev,
        fmt="o",
        capsize=5,
        color="blue",
        alpha=0.7,
    )

    # Add a line for y = x
    max_value = (
        max(np.max(observed), np.max(upper_bounds)) + 0.1
    )  # Extend the line slightly beyond the max values
    min_value = min(np.min(observed), np.min(lower_bounds)) - 0.1
    plt.plot([min_value, max_value], [min_value, max_value], "k-", linewidth=2)

    # Add labels and title
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

    if not os.path.exists("plots"):
        os.makedirs("plots")
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    path_to_plot = os.path.join(
        "plots", f"test_predictions_{objective_function}_{timestamp}.png"
    )
    plt.savefig(path_to_plot, bbox_inches="tight")
    print(f"Figure saved to {path_to_plot}")


def sobol_plot(
    S1: Sequence[float],
    ST: Sequence[float],
    variables: List[str],
    S1_conf: Sequence[float],
    ST_conf: Sequence[float],
    objective_function: str,
):
    """
    Plots first and total order Sobol sensitivity indices with confidence
    intervals and saves the figure.

    Args:
        S1 (Sequence[float]): First order sensitivity indices for each variable.
        ST (Sequence[float]): Total order sensitivity indices for each variable.
        variables (List[str]): List of variable names.
        S1_conf (Sequence[float]): Confidence intervals for first order indices.
        ST_conf (Sequence[float]): Confidence intervals for total order indices.
        objective_function (str): Name of the objective function, used in the
            saved plot filename.

    Returns:
        None, for visualization purposes only.
    """
    # Define colors for each variable
    colors = sns.color_palette("husl", len(variables))

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

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

    if not os.path.exists("plots"):
        os.makedirs("plots")
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    path_to_plot = os.path.join(
        "plots", f"sensitivity_{objective_function}_{timestamp}.png"
    )
    plt.savefig(path_to_plot, bbox_inches="tight")
    print(f"Figure saved to {path_to_plot}")
