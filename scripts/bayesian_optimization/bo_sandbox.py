#!/usr/bin/env python3

"""
Animate Bayesian Optimization on synthetic test functions.

This script visualizes the BO process on 2D test functions, showing the test function
surface, acquisition function evolution, and GP mean predictions over iterations.
Supports EI, PI, UCB, PV, and random acquisition strategies.

Usage examples:

./bo_sandbox.py --help
./bo_sandbox.py
./bo_sandbox.py --test-function parabola --acquisition EI --init-design lhd --save-animation
./bo_sandbox.py --test-function parabola --acquisition EI --n-iter 15
./bo_sandbox.py --test-function parabola --acquisition random --n-iter 15 --n-initial 10
./bo_sandbox.py --test-function ackley --acquisition UCB --n-initial 3 --n-iter 20 --beta 2.0
./bo_sandbox.py --test-function branin --acquisition UCB --n-iter 20 --n-initial 3 --seed 2
./bo_sandbox.py --test-function branin --acquisition UCB --n-iter 20 --n-initial 3 --no-scale-x
"""

import argparse
import io
import time
from collections.abc import Generator
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import torch

from surmod import bayesian_optimization as bo
from surmod.test_functions import load_test_function
from surmod.utils import log_results


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Perform Bayesian optimization on synthetic test functions.",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    experiment = parser.add_argument_group("experiment options")
    bo_options = parser.add_argument_group("Bayesian optimization options")

    experiment.add_argument(
        "-f",
        "--test-function",
        type=str,
        default="parabola",
        help="Test function to use. Supported: parabola, ackley, branin, holder_table, griewank, six_hump_camel.",
    )
    experiment.add_argument(
        "-in",
        "--n-initial",
        type=int,
        default=10,
        help="Number of initial samples before Bayesian optimization.",
    )
    experiment.add_argument(
        "-it",
        "--n-iter",
        type=int,
        default=10,
        help="Number of Bayesian optimization acquisitions.",
    )
    experiment.add_argument(
        "--init-design",
        type=str,
        choices=["random", "lhd", "maximin_lhd"],
        default="random",
        help="Initial design strategy for BO.",
    )
    experiment.add_argument(
        "-save",
        "--save-animation",
        action="store_true",
        help="Save the animation instead of displaying it interactively.",
    )

    bo_options.add_argument(
        "-acq",
        "--acquisition",
        type=str,
        choices=["EI", "PI", "UCB", "PV", "random"],
        default="EI",
        help="Choice of acquisition function.",
    )
    bo_options.add_argument(
        "-beta",
        "--beta",
        type=float,
        default=2.0,
        help="Tuning parameter for UCB method only.",
    )
    bo_options.add_argument(
        "-k",
        "--kernel",
        type=str,
        choices=["rbf", "matern", "periodic"],
        default="matern",
        help="GP kernel function.",
    )
    bo_options.add_argument(
        "-i",
        "--isotropic",
        action="store_true",
        help="Use isotropic kernel (single lengthscale for all inputs).",
    )
    bo_options.add_argument(
        "--scale-x",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale the input values to [0,1] per dimension using training data.",
    )
    bo_options.add_argument(
        "-ny",
        "--normalize-y",
        action="store_true",
        default=False,
        help="Standardize outputs (maps to GPSurrogate.scale_outputs).",
    )
    bo_options.add_argument(
        "--fixed-nugget",
        type=float,
        default=None,
        metavar="VALUE",
        help="Set the white-noise variance (nugget) to VALUE instead of learning it.",
    )

    bo_options.add_argument(
        "--lengthscale-bounds",
        type=float,
        nargs=2,
        default=(1e-2, 100.0),
        metavar=("LOW", "HIGH"),
        help="Bounds for kernel lengthscale constraint.",
    )

    bo_options.add_argument(
        "--noise-bounds",
        type=float,
        nargs=2,
        default=(1e-8, 1e-1),
        metavar=("LOW", "HIGH"),
        help="Bounds for likelihood noise constraint.",
    )

    return parser.parse_args()


def run_bayesian_optimization(
    bopt: bo.BayesianOptimizer,
    x_grid: np.ndarray,
    x1_grid: np.ndarray,
) -> Generator[dict, None, None]:
    """
    Run Bayesian optimization and yield per-iteration diagnostics.

    Args:
        bopt: Configured Bayesian optimizer.
        x_grid: Candidate evaluation grid used for visualization.
        x1_grid: Meshgrid array used only for reshaping diagnostics.

    Yields:
        Snapshot dictionaries returned by ``bopt.step(...)`` for each
        acquisition iteration.
    """
    bopt.y_max_history = np.array([np.max(bopt.y_all_data)], dtype=float)

    for i in range(bopt.n_acquire):
        snapshot = bopt.step(
            x_grid=x_grid,
            grid_shape=x1_grid.shape,
            return_diagnostics=True,
        )
        snapshot["iteration"] = i

        x_next = snapshot["x_next"]
        y_next_scalar = snapshot["y_next"]
        y_max = snapshot["y_max"]
        x_best = snapshot["x_best"]
        gp_mean_max_location = snapshot["gp_mean_max_location"]
        gp_mean_max_value = snapshot["gp_mean_max_value"]

        print(
            f"\nIter. {i + 1}: acquired f(x)={y_next_scalar:.3g} at x=({x_next[0]:.3g},{x_next[1]:.3g})"
        )
        print(
            f"Iter. {i + 1}: max f(x)={y_max:.3g} at x=({x_best[0]:.3g},{x_best[1]:.3g})"
        )
        print(
            f"Iter. {i + 1}: max GP mean={gp_mean_max_value:.3g} "
            f"at x=({gp_mean_max_location[0]:.3g},{gp_mean_max_location[1]:.3g})"
        )

        yield snapshot


def _capture_frame(fig: matplotlib.figure.Figure, frames: list) -> None:
    """
    Capture the current Matplotlib figure and append it to a frame list.

    Args:
        fig: Figure to serialize to an image frame.
        frames: Mutable list collecting rendered frames.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    frames.append(imageio.imread(buf))
    buf.close()


def setup_figure(
    bopt: bo.BayesianOptimizer,
    x1_grid: np.ndarray,
    x2_grid: np.ndarray,
    y_grid: np.ndarray,
    x_sample: np.ndarray,
    synth_function: object,
    global_optima: list,
    test_function: str,
    kernel: str,
    n_initial: int,
    n_iteration: int,
) -> tuple[matplotlib.figure.Figure, dict, dict, dict]:
    """
    Create the initial Bayesian-optimization visualization layout.

    Args:
        bopt: Configured Bayesian optimizer.
        x1_grid: Meshgrid array for the first input dimension.
        x2_grid: Meshgrid array for the second input dimension.
        y_grid: Objective values evaluated on the plotting grid.
        x_sample: Initial sampled design points.
        synth_function: Synthetic objective function used for plotting bounds.
        global_optima: Known global optima locations for the test function.
        test_function: Test-function name used in figure titles.
        kernel: Kernel name used in figure titles.
        n_initial: Number of initial design points.
        n_iteration: Number of Bayesian optimization iterations.

    Returns:
        A tuple containing the figure, axes mapping, mutable plot handles, and
        metadata used by downstream animation helpers.
    """
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(
        f"Bayesian Optimization of {test_function} w/ {kernel} kernel\n",
        fontsize=16,
    )

    ax1 = fig.add_subplot(131, aspect="equal")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")

    title_lines = [
        f"{test_function} with {kernel} kernel",
        f"Initial Samples: {n_initial} | Acquired Samples: {n_iteration}",
    ]

    bounds_low = [b[0] for b in synth_function._bounds]
    bounds_high = [b[1] for b in synth_function._bounds]

    ax1.set_xlim(bounds_low[0] - 1, bounds_high[0] + 1)
    ax1.set_ylim(bounds_low[1] - 1, bounds_high[1] + 1)
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_title("\n".join(title_lines))
    contour = ax1.contourf(
        x1_grid, x2_grid, y_grid, levels=25, cmap="viridis_r", alpha=0.3
    )
    plt.colorbar(contour, ax=ax1, label=f"{test_function} (maximizing)")
    ax1.scatter(
        x_sample[:, 0],
        x_sample[:, 1],
        marker="x",
        color="green",
        label="Initial samples",
    )
    for idx, point in enumerate(global_optima):
        ax1.scatter(
            point[0],
            point[1],
            marker="x",
            color="red",
            label="Global Maximum" if idx == 0 else "",
        )
    ax1.legend(loc="upper right")

    x_grid = np.vstack([x1_grid.ravel(), x2_grid.ravel()]).T

    # Fit initial GP for visualization
    gp_initial = bopt.gp_model_fit()
    acq_init = bopt.score_candidates(x_grid)

    acq_init = acq_init.reshape(x1_grid.shape)
    acq_surface = ax2.plot_surface(x1_grid, x2_grid, acq_init, cmap="viridis_r")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_zlabel("Acquisition Value")
    ax2.set_title("Acquisition Function")

    mu_init, _ = gp_initial.predict(x_grid)
    mu_init = mu_init.reshape(x1_grid.shape)
    gp_mean_max_val = np.max(mu_init)
    gp_mean_max_loc = x_grid[np.argmax(mu_init), :]
    gp_surface = ax3.plot_surface(
        x1_grid, x2_grid, mu_init, cmap="viridis_r", alpha=0.6
    )
    gp_mean_dot = ax3.scatter(
        gp_mean_max_loc[0],
        gp_mean_max_loc[1],
        gp_mean_max_val,
        color="red",
        s=50,
        label="GP Mean Max",
    )
    ax3.set_xlabel("x1")
    ax3.set_ylabel("x2")
    ax3.set_zlabel("Value")
    ax3.set_title("Objective Function Contour and GP Mean Surface")
    ax3.contour(
        x1_grid, x2_grid, y_grid, levels=25, cmap="viridis_r", linestyles="solid"
    )
    ax3.legend()

    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4)
    plt.tight_layout()

    handles = {
        "acq_surface": acq_surface,
        "gp_surface": gp_surface,
        "gp_mean_dot": gp_mean_dot,
    }
    axes = {"ax1": ax1, "ax2": ax2, "ax3": ax3}
    meta = {"title_lines": title_lines}

    return fig, axes, handles, meta


def animate_optimization(
    snapshots: Generator[dict, None, None],
    fig: matplotlib.figure.Figure,
    axes: dict,
    handles: dict,
    x1_grid: np.ndarray,
    x2_grid: np.ndarray,
    save_animation: bool,
) -> tuple[list, np.ndarray, np.ndarray]:
    """
    Animate or step through Bayesian-optimization snapshots.

    Args:
        snapshots: Generator of per-iteration optimization snapshots.
        fig: Figure being updated.
        axes: Mapping of subplot names to axes.
        handles: Mutable mapping of plot artists that are updated in-place.
        x1_grid: Meshgrid array for the first input dimension.
        x2_grid: Meshgrid array for the second input dimension.
        save_animation: If ``True``, capture frames instead of pausing
            interactively.

    Returns:
        A tuple containing captured frames, maxima of acquired observations, and
        maxima of the GP posterior mean over time.
    """
    ax1, ax2, ax3 = axes["ax1"], axes["ax2"], axes["ax3"]
    frames = []
    acquired_maxima = []
    gp_mean_maxima = []

    first_acquired = True

    for snap in snapshots:
        x_next = snap["x_next"]

        ax1.scatter(
            x_next[0], x_next[1], color="blue", marker="s", label="Acquired point"
        )
        if first_acquired:
            ax1.legend(loc="upper right")
            first_acquired = False

        if save_animation:
            _capture_frame(fig, frames)
        else:
            plt.draw()
            plt.pause(0.6)

        handles["acq_surface"].remove()
        handles["acq_surface"] = ax2.plot_surface(
            x1_grid, x2_grid, snap["acq_values"], cmap="viridis_r"
        )

        if save_animation:
            _capture_frame(fig, frames)
        else:
            plt.draw()
            plt.pause(1.0)

        handles["gp_surface"].remove()
        handles["gp_mean_dot"].remove()
        handles["gp_surface"] = ax3.plot_surface(
            x1_grid, x2_grid, snap["mu"], cmap="viridis_r", alpha=0.6
        )
        loc = snap["gp_mean_max_location"]
        val = snap["gp_mean_max_value"]
        handles["gp_mean_dot"] = ax3.scatter(
            loc[0], loc[1], val, color="red", s=50, label="Maximum of GP Mean"
        )
        ax3.legend()

        if save_animation:
            _capture_frame(fig, frames)
        else:
            plt.draw()
            plt.pause(1.0)

        acquired_maxima.append(snap["acquired_max"])
        gp_mean_maxima.append(snap["gp_mean_max_value"])

    return frames, np.array(acquired_maxima), np.array(gp_mean_maxima)


def plot_convergence(
    acquired_maxima: np.ndarray,
    gp_mean_maxima: np.ndarray,
    global_optimum_value: float,
    title_lines: list[str],
    save_animation: bool,
    plots_dir: Path,
) -> None:
    """
    Plot convergence histories for acquired values and GP-mean maxima.

    Args:
        acquired_maxima: Best acquired objective value at each iteration.
        gp_mean_maxima: Best GP posterior mean value at each iteration.
        global_optimum_value: Known optimum value of the test function.
        title_lines: Title lines displayed on the figure.
        save_animation: If ``True``, save the figure instead of showing it.
        plots_dir: Directory where the convergence plot is saved.
    """
    _fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(
        acquired_maxima,
        color="red",
        label="Maximum of acquired points",
        marker="o",
        linestyle="--",
    )
    ax.plot(
        gp_mean_maxima,
        color="blue",
        label="Maximum of GP Mean",
        marker="o",
        linestyle="--",
    )
    ax.axhline(
        y=global_optimum_value,
        color="green",
        linestyle="-",
        linewidth=3,
        label="True Global Optimum",
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Maximum Value")
    ax.set_title("\n".join(title_lines))
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    if save_animation:
        plots_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%m%d_%H%M%S")
        path = plots_dir / f"track_max_{title_lines[0].split()[0]}_{ts}.png"
        plt.savefig(path)
        print(f"Convergence figure saved to {path}")
    else:
        plt.show()


def save_gif(frames: list, test_function: str, plots_dir: Path) -> None:
    """
    Save captured animation frames as a GIF.

    Args:
        frames: Rendered animation frames.
        test_function: Test-function name used in the output filename.
        plots_dir: Directory where the GIF is saved.
    """
    plots_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%m%d_%H%M%S")
    path = plots_dir / f"bayes_opt_animation_{test_function}_{ts}.gif"
    imageio.mimsave(path, frames, fps=2)
    print(f"Animation saved as {path}")


def main() -> None:
    """Run Bayesian optimization on a synthetic test function and visualize it."""
    args = parse_arguments()

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set plots directory relative to this script
    plots_dir = Path(__file__).parent / "plots"

    synth_function = load_test_function(args.test_function)
    bounds_low = [b[0] for b in synth_function._bounds]
    bounds_high = [b[1] for b in synth_function._bounds]

    x1 = np.linspace(bounds_low[0], bounds_high[0], 101)
    x2 = np.linspace(bounds_low[1], bounds_high[1], 101)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    x_grid = np.vstack([x1_grid.ravel(), x2_grid.ravel()]).T
    y_grid = np.array(
        [
            synth_function(torch.from_numpy(x.reshape(1, -1))).detach().numpy()
            for x in x_grid
        ]
    ).reshape(x1_grid.shape)

    global_optima, global_optimum_value = bo.get_synth_global_optima(args.test_function)

    x_sample, y_sample = bo.sample_data(
        args.test_function,
        bounds_low,
        bounds_high,
        args.n_initial,
        input_size=2,
        init_design=args.init_design,
        seed=args.seed,
    )

    bopt = bo.BayesianOptimizer(
        test_function=args.test_function,
        x_init=x_sample,
        y_init=y_sample,
        kernel=args.kernel,
        isotropic=args.isotropic,
        acquisition_function=args.acquisition,
        n_acquire=args.n_iter,
        seed=args.seed,
        beta=args.beta,
    )

    fig, axes, handles, meta = setup_figure(
        bopt=bopt,
        x1_grid=x1_grid,
        x2_grid=x2_grid,
        y_grid=y_grid,
        x_sample=x_sample,
        synth_function=synth_function,
        global_optima=global_optima,
        test_function=args.test_function,
        kernel=args.kernel,
        n_initial=args.n_initial,
        n_iteration=args.n_iter,
    )

    if not args.save_animation:
        plt.show(block=False)

    start_time = time.time()
    snapshots = run_bayesian_optimization(
        bopt,
        x_grid,
        x1_grid,
    )

    frames, acquired_maxima, gp_mean_maxima = animate_optimization(
        snapshots,
        fig,
        axes,
        handles,
        x1_grid,
        x2_grid,
        save_animation=args.save_animation,
    )
    elapsed_time = time.time() - start_time

    if args.save_animation and frames:
        save_gif(frames, args.test_function, plots_dir)

    plot_convergence(
        acquired_maxima,
        gp_mean_maxima,
        global_optimum_value,
        title_lines=meta["title_lines"],
        save_animation=args.save_animation,
        plots_dir=plots_dir,
    )

    # Log results
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    best_acquired = acquired_maxima[-1]
    best_gp_mean = gp_mean_maxima[-1]
    log_lines = [
        f"Run timestamp (%m%d_%H%M%S): {timestamp}",
        f"Test Function: {args.test_function}",
        f"Acquisition Function: {args.acquisition}",
        f"Kernel: {args.kernel}",
        f"Isotropic: {args.isotropic}",
        f"Initial design: {args.init_design}",
        f"Number of initial points: {args.n_initial}",
        f"Number of BO iterations: {args.n_iter}",
        f"Beta (UCB): {args.beta if args.acquisition == 'UCB' else 'N/A'}",
        f"Global optimum value: {global_optimum_value:.5e}",
        f"Best acquired value: {best_acquired:.5e}",
        f"Best GP mean value: {best_gp_mean:.5e}",
        f"Elapsed time for BO: {elapsed_time:.3f} seconds\n",
    ]
    log_message = "\n".join(log_lines)
    print(log_message)

    results_dir = Path(__file__).parent / "results"
    log_results(
        log_message,
        path_to_log=results_dir / f"{args.test_function}_{args.acquisition}_bo.txt",
    )


if __name__ == "__main__":
    main()
