"""Utility functions for the surmod package."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def log_results(log_message: str, path_to_log: Path) -> None:
    """
    Append log message to file.

    Args:
        log_message: String to write to the log file.
        path_to_log: Path object pointing to the log file.
    """
    path_to_log.parent.mkdir(parents=True, exist_ok=True)
    with open(path_to_log, "a", encoding="utf-8") as f:
        f.write(log_message + "\n")


def save_parity_plot(
    observed_values: np.ndarray,
    predicted_values: np.ndarray,
    output_path: Path,
    uncertainty: np.ndarray | None = None,
    title: str | None = None,
) -> None:
    """
    Save a parity plot comparing observed and predicted values.

    Args:
        observed_values: Observed target values.
        predicted_values: Predicted target values.
        output_path: Location where the plot image will be written.
        uncertainty: Optional predictive standard deviations for 95 percent
            intervals.
        title: Optional figure title.
    """
    observed = np.asarray(observed_values).reshape(-1)
    predicted = np.asarray(predicted_values).reshape(-1)

    if observed.shape != predicted.shape:
        raise ValueError(
            "observed_values and predicted_values must have matching shapes."
        )

    metric_lines = [f"RMSE: {np.sqrt(np.mean(np.square(observed - predicted))):.5f}"]
    interval_half_width = None

    if uncertainty is not None:
        std_dev = np.asarray(uncertainty).reshape(-1)
        if std_dev.shape != observed.shape:
            raise ValueError("uncertainty must have the same shape as observed_values.")
        interval_half_width = 1.96 * std_dev
        lower_bounds = predicted - interval_half_width
        upper_bounds = predicted + interval_half_width
        coverage = np.mean((observed >= lower_bounds) & (observed <= upper_bounds))
        metric_lines.append(f"Coverage: {coverage:.2%}")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 8))

    if interval_half_width is None:
        ax.scatter(observed, predicted, color="blue", alpha=0.7)
        lower_extent = predicted
        upper_extent = predicted
    else:
        ax.errorbar(
            observed,
            predicted,
            yerr=interval_half_width,
            fmt="o",
            capsize=5,
            color="blue",
            alpha=0.7,
        )
        lower_extent = predicted - interval_half_width
        upper_extent = predicted + interval_half_width

    min_value = min(observed.min(), lower_extent.min())
    max_value = max(observed.max(), upper_extent.max())
    value_range = max_value - min_value
    padding = 0.05 * value_range if value_range > 0 else 0.5
    axis_limits = (min_value - padding, max_value + padding)

    ax.plot(axis_limits, axis_limits, "k-", linewidth=2)
    ax.set_xlim(axis_limits)
    ax.set_ylim(axis_limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.set_xlabel("Observed", fontsize=14)
    ax.set_ylabel("Predicted", fontsize=14)

    if title:
        ax.set_title(title)

    ax.text(
        0.05,
        0.95,
        "\n".join(metric_lines),
        ha="left",
        va="top",
        fontsize=12,
        transform=ax.transAxes,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {output_path}")
