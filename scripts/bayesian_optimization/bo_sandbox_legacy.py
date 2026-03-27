#!/usr/bin/env python3

"""
This script creates an animation of Bayesian Optimization (BO) on a chosen
synthetic objective function and plots performance of the chosen acquisition
function: Expected Improvement (EI), Probability of Improvement (PI),
Upper Confidence Bound (UCB), or random.

The approach is:
1. Sample an initial set of points from the chosen synthetic function
2. Train a GP model on the initial data
3. Compute the acquisition function over a grid of candidate points
4. Add the point with the highest acquisition value to the training data
5. Return to step 2 and repeat until the user-defined number of iterations
    is reached

Usage:

# Make script executable
chmod +x ./bo_sandbox.py

# See help.
./bo_sandbox.py -h

# Perform BO for a parabola, start with 5 points, use the RBF kernel,
#   and run the algorithm for 15 iterations
./bo_sandbox.py -f Parabola -in 5 -k rbf -it 15 -xi 0

# Perform BO for maximizing the Branin function, start with 5 points, use the
#   Matern kernel, and run the algorithm for 20 iterations.  Set random seed to 2.
./bo_sandbox.py -f Branin -in 5 -k matern -it 20 -s 2 -xi 0.01
"""

import argparse
import os
import io
from datetime import datetime
from typing import Generator

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
import imageio.v2 as imageio
import torch

from surmod import bayesian_optimization as bo
from surmod import gaussian_process_regression as gpr


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    """
    Get command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Perform Bayesian optimization with GP surrogate models.",
    )
    parser.add_argument("-it", "--n_iteration", type=int, default=10)
    parser.add_argument("-in", "--n_initial", type=int, default=10)
    parser.add_argument(
        "-k",
        "--kernel",
        type=str,
        choices=["matern", "rbf", "matern_dot"],
        default="matern",
    )
    parser.add_argument(
        "-acq",
        "--acquisition",
        type=str,
        choices=["EI", "PI", "UCB", "random"],
        default="EI",
    )
    parser.add_argument(
        "-f",
        "--objective_function",
        type=str,
        choices=["Parabola", "Ackley", "Griewank", "Branin", "HolderTable"],
        default="Parabola",
    )
    parser.add_argument("-i", "--isotropic", action="store_true")
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-save", "--save_animation", action="store_true")
    parser.add_argument("-xi", "--xi", type=float, default=0.0)
    parser.add_argument("-kappa", "--kappa", type=float, default=2.0)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Acquisition function helper
# ---------------------------------------------------------------------------


def compute_acquisition(
    acquisition: str,
    x_grid: np.ndarray,
    gp: object,
    y_max: float,
    xi: float,
    kappa: float,
) -> np.ndarray:
    """
    Compute acquisition values over x_grid for the given acquisition strategy.

    Parameters
    ----------
    acquisition : str
        Name of the acquisition function. One of ``"EI"``, ``"PI"``,
        ``"UCB"``, or ``"random"``.
    x_grid : np.ndarray of shape (n_points, n_features)
        Candidate points at which to evaluate the acquisition function.
    gp : object
        Fitted Gaussian Process model with a ``predict`` method.
    y_max : float
        Current maximum observed objective value.
    xi : float
        Exploration-exploitation trade-off parameter for EI and PI
        (non-negative).
    kappa : float
        Exploration-exploitation trade-off parameter for UCB (non-negative).

    Returns
    -------
    np.ndarray of shape (n_points,)
        Acquisition function values at each point in x_grid.

    Raises
    ------
    ValueError
        If ``acquisition`` is not one of the recognised strategies.
    """
    if acquisition == "EI":
        return bo.expected_improvement(x_grid, gp, y_max, xi=xi)
    elif acquisition == "PI":
        return bo.probability_of_improvement(x_grid, gp, y_max, xi=xi)
    elif acquisition == "UCB":
        return bo.upper_confidence_bound(x_grid, gp, kappa=kappa)
    elif acquisition == "random":
        return np.random.uniform(size=x_grid.shape[0])
    else:
        raise ValueError(f"Unknown acquisition function: {acquisition!r}")


# ---------------------------------------------------------------------------
# Core BO loop (pure logic, no plotting)
# ---------------------------------------------------------------------------


def run_bayesian_optimization(
    bopt: bo.BayesianOptimizer,
    x_grid: np.ndarray,
    x1_grid: np.ndarray,
    acquisition: str,
    xi: float,
    kappa: float,
) -> Generator[dict, None, None]:
    """
    Run the BO loop and yield a snapshot dict at each iteration.

    Parameters
    ----------
    bopt : bo.BayesianOptimizer
        Configured BayesianOptimizer instance with initial data already set.
    x_grid : np.ndarray of shape (n_points, n_features)
        Flattened grid of candidate points for GP prediction and acquisition
        evaluation.
    x1_grid : np.ndarray of shape (n_grid, n_grid)
        Meshgrid array for the first input dimension, used to reshape
        flat predictions back to 2-D for plotting.
    acquisition : str
        Name of the acquisition function. One of ``"EI"``, ``"PI"``,
        ``"UCB"``, or ``"random"``.
    xi : float
        Exploration-exploitation trade-off parameter for EI and PI
        (non-negative).
    kappa : float
        Exploration-exploitation trade-off parameter for UCB (non-negative).

    Yields
    ------
    dict
        Snapshot of the optimizer state after each acquisition step with keys:

        - ``iteration`` (int): Zero-based iteration index.
        - ``x_next`` (np.ndarray): Newly acquired input point.
        - ``y_next`` (float): Objective value at ``x_next``.
        - ``y_max`` (float): Maximum observed objective value so far.
        - ``x_best`` (np.ndarray): Input point achieving ``y_max``.
        - ``mu`` (np.ndarray): GP mean surface shaped like ``x1_grid``.
        - ``acq_values`` (np.ndarray): Acquisition surface shaped like
          ``x1_grid``.
        - ``gp_mean_max_value`` (float): Maximum value of the GP mean surface.
        - ``gp_mean_max_location`` (np.ndarray): Location of
          ``gp_mean_max_value`` in input space.
        - ``acquired_max`` (float): Maximum objective value among all acquired
          points.
    """
    gp = bopt.gp_model_fit()

    for i in range(bopt.n_acquire):
        # --- propose and evaluate ---
        x_next = bopt.propose_location(bopt.acquisition)
        y_next = bopt.evaluate_objective(x_next)

        # --- update optimizer state ---
        bopt.x_all_data = np.vstack((bopt.x_all_data, x_next.reshape(1, -1)))
        bopt.y_all_data = np.append(bopt.y_all_data, y_next)
        bopt.x_acquired = np.append(bopt.x_acquired, [x_next], axis=0)
        bopt.y_acquired = np.append(bopt.y_acquired, y_next)
        y_max = np.max(bopt.y_all_data)
        bopt.y_max_history = np.append(bopt.y_max_history, y_max)

        # --- refit GP ---
        gp = bopt.gp_model_fit()

        # --- GP mean surface ---
        mu = gp.predict(x_grid, return_std=False)
        if isinstance(mu, tuple):
            mu = mu[0]
        gp_mean_max_value = float(np.max(mu))
        gp_mean_max_location = x_grid[np.argmax(mu), :]
        mu_grid = mu.reshape(x1_grid.shape)

        # --- acquisition surface ---
        acq_values = compute_acquisition(acquisition, x_grid, gp, y_max, xi, kappa)
        acq_grid = acq_values.reshape(x1_grid.shape)

        # --- best observed point ---
        x_best = bopt.x_all_data[np.argmax(bopt.y_all_data), :]

        snapshot = dict(
            iteration=i,
            x_next=x_next,
            y_next=y_next,
            y_max=y_max,
            x_best=x_best,
            mu=mu_grid,
            acq_values=acq_grid,
            gp_mean_max_value=gp_mean_max_value,
            gp_mean_max_location=gp_mean_max_location,
            acquired_max=float(np.max(bopt.y_acquired)),
        )

        print(
            f"\nIter. {i+1}: acquired f(x)={y_next[0]:.3g} at x=({x_next[0]:.3g},{x_next[1]:.3g})"
        )
        print(
            f"Iter. {i+1}: max f(x)={y_max:.3g} at x=({x_best[0]:.3g},{x_best[1]:.3g})"
        )
        print(
            f"Iter. {i+1}: max GP mean={gp_mean_max_value:.3g} "
            f"at x=({gp_mean_max_location[0]:.3g},{gp_mean_max_location[1]:.3g})"
        )

        yield snapshot


# ---------------------------------------------------------------------------
# Figure setup
# ---------------------------------------------------------------------------


def _capture_frame(fig: matplotlib.figure.Figure, frames: list) -> None:
    """
    Render the current figure to a PNG buffer and append it to frames.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to capture.
    frames : list of np.ndarray
        List to which the captured frame is appended in-place.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    frames.append(imageio.imread(buf))
    buf.close()


def setup_figure(
    bopt_config: dict,
    x1_grid: np.ndarray,
    x2_grid: np.ndarray,
    y_grid: np.ndarray,
    x_sample: np.ndarray,
    synth_function: object,
    global_optima: list,
    objective_function: str,
    kernel: str,
    n_initial: int,
    n_iteration: int,
    acquisition: str,
    xi: float,
    kappa: float,
    gp_initial: object,
) -> tuple[matplotlib.figure.Figure, dict, dict, dict]:
    """
    Build the figure and axes and draw the initial state before any acquisition.

    Parameters
    ----------
    bopt_config : dict
        Configuration dict containing at least ``"y_sample"``
        (np.ndarray of shape (n_initial,)), the initial objective values used
        to seed the acquisition surface.
    x1_grid : np.ndarray of shape (n_grid, n_grid)
        Meshgrid array for the first input dimension.
    x2_grid : np.ndarray of shape (n_grid, n_grid)
        Meshgrid array for the second input dimension.
    y_grid : np.ndarray of shape (n_grid, n_grid)
        True objective function values evaluated on the meshgrid.
    x_sample : np.ndarray of shape (n_initial, 2)
        Initial sample points drawn from the objective function domain.
    synth_function : object
        Synthetic test function with a ``_bounds`` attribute (list of
        ``(low, high)`` tuples, one per input dimension).
    global_optima : list of array-like
        Coordinates of the known global maximum/maxima of the objective.
    objective_function : str
        Display name of the objective function (e.g. ``"Branin"``).
    kernel : str
        Kernel name used for the GP (e.g. ``"matern"``).
    n_initial : int
        Number of initial sample points.
    n_iteration : int
        Number of BO iterations (acquired points).
    acquisition : str
        Name of the acquisition function.
    xi : float
        Exploration-exploitation trade-off parameter for EI and PI.
    kappa : float
        Exploration-exploitation trade-off parameter for UCB.
    gp_initial : object
        GP model fitted on the initial sample, used to draw the initial
        acquisition and mean surfaces.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The constructed figure.
    axes : dict
        Dictionary with keys ``"ax1"``, ``"ax2"``, ``"ax3"`` mapping to the
        three subplot axes.
    handles : dict
        Mutable plot handles with keys ``"acq_surface"``, ``"gp_surface"``,
        and ``"gp_mean_dot"`` that are replaced each iteration.
    meta : dict
        Metadata dict containing ``"title_lines"`` (list of str) for reuse
        in the convergence plot.
    """
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(
        f"Bayesian Optimization of {objective_function} w/ {kernel} kernel\n",
        fontsize=16,
    )

    ax1 = fig.add_subplot(131, aspect="equal")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")

    title_lines = [
        f"{objective_function} with {kernel} kernel",
        f"Initial Samples: {n_initial} | Acquired Samples: {n_iteration}",
    ]

    bounds_low = [b[0] for b in synth_function._bounds]
    bounds_high = [b[1] for b in synth_function._bounds]

    # --- ax1: objective function contour ---
    ax1.set_xlim(bounds_low[0] - 1, bounds_high[0] + 1)
    ax1.set_ylim(bounds_low[1] - 1, bounds_high[1] + 1)
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_title("\n".join(title_lines))
    contour = ax1.contourf(
        x1_grid, x2_grid, y_grid, levels=25, cmap="inferno", alpha=0.3
    )
    plt.colorbar(contour, ax=ax1, label=f"Value of {objective_function}")
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

    # --- ax2: initial acquisition surface ---
    x_grid = np.vstack([x1_grid.ravel(), x2_grid.ravel()]).T
    acq_init = compute_acquisition(
        acquisition, x_grid, gp_initial, np.max(bopt_config["y_sample"]), xi, kappa
    )
    acq_init = acq_init.reshape(x1_grid.shape)
    acq_surface = ax2.plot_surface(x1_grid, x2_grid, acq_init, cmap="viridis")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_zlabel("Acquisition Value")
    ax2.set_title("Acquisition Function")

    # --- ax3: initial GP mean surface ---
    mu_init = gp_initial.predict(x_grid, return_std=False)
    if isinstance(mu_init, tuple):
        mu_init = mu_init[0]
    mu_init = mu_init.reshape(x1_grid.shape)
    gp_mean_max_val = float(np.max(mu_init))
    gp_mean_max_loc = x_grid[np.argmax(mu_init), :]
    gp_surface = ax3.plot_surface(x1_grid, x2_grid, mu_init, cmap="viridis", alpha=0.6)
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
    ax3.contour(x1_grid, x2_grid, y_grid, levels=25, cmap="inferno", linestyles="solid")
    ax3.legend()

    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4)
    plt.tight_layout()

    # Mutable plot handles that get swapped each iteration
    handles = dict(
        acq_surface=acq_surface, gp_surface=gp_surface, gp_mean_dot=gp_mean_dot
    )
    axes = dict(ax1=ax1, ax2=ax2, ax3=ax3)
    meta = dict(title_lines=title_lines)

    return fig, axes, handles, meta


# ---------------------------------------------------------------------------
# Animation / live update loop
# ---------------------------------------------------------------------------


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
    Consume snapshots from run_bayesian_optimization() and update the figure.

    Parameters
    ----------
    snapshots : Generator[dict, None, None]
        Generator yielding snapshot dicts as produced by
        ``run_bayesian_optimization``.
    fig : matplotlib.figure.Figure
        The figure to update in-place.
    axes : dict
        Dictionary with keys ``"ax1"``, ``"ax2"``, ``"ax3"`` mapping to the
        three subplot axes.
    handles : dict
        Mutable plot handles with keys ``"acq_surface"``, ``"gp_surface"``,
        and ``"gp_mean_dot"`` that are replaced each iteration.
    x1_grid : np.ndarray of shape (n_grid, n_grid)
        Meshgrid array for the first input dimension.
    x2_grid : np.ndarray of shape (n_grid, n_grid)
        Meshgrid array for the second input dimension.
    save_animation : bool
        If True, capture each frame to ``frames`` instead of displaying
        interactively.

    Returns
    -------
    frames : list of np.ndarray
        PNG frames captured at each animation step. Empty if
        ``save_animation`` is False.
    acquired_maxima : np.ndarray of shape (n_iter,)
        Maximum objective value among all acquired points at each iteration.
    gp_mean_maxima : np.ndarray of shape (n_iter,)
        Maximum GP mean surface value at each iteration.
    """
    ax1, ax2, ax3 = axes["ax1"], axes["ax2"], axes["ax3"]
    frames = []
    acquired_maxima = []
    gp_mean_maxima = []

    first_acquired = True

    for snap in snapshots:
        x_next = snap["x_next"]

        # -- ax1: plot newly acquired point --
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

        # -- ax2: update acquisition surface --
        handles["acq_surface"].remove()
        handles["acq_surface"] = ax2.plot_surface(
            x1_grid, x2_grid, snap["acq_values"], cmap="viridis"
        )

        if save_animation:
            _capture_frame(fig, frames)
        else:
            plt.draw()
            plt.pause(1.0)

        # -- ax3: update GP mean surface --
        handles["gp_surface"].remove()
        handles["gp_mean_dot"].remove()
        handles["gp_surface"] = ax3.plot_surface(
            x1_grid, x2_grid, snap["mu"], cmap="viridis", alpha=0.6
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


# ---------------------------------------------------------------------------
# Convergence plot
# ---------------------------------------------------------------------------


def plot_convergence(
    acquired_maxima: np.ndarray,
    gp_mean_maxima: np.ndarray,
    global_optimum_value: float,
    title_lines: list[str],
    save_animation: bool,
) -> None:
    """
    Plot the maximum acquired value and GP mean maximum against the true
    global optimum.

    Parameters
    ----------
    acquired_maxima : np.ndarray of shape (n_iter,)
        Maximum objective value among all acquired points at each iteration.
    gp_mean_maxima : np.ndarray of shape (n_iter,)
        Maximum GP mean surface value at each iteration.
    global_optimum_value : float
        True global maximum of the objective function, drawn as a horizontal
        reference line.
    title_lines : list of str
        Lines of text joined to form the plot title.
    save_animation : bool
        If True, save the figure to ``plots/`` instead of displaying it
        interactively.
    """
    fig, ax = plt.subplots(figsize=(18, 6))
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
        os.makedirs("plots", exist_ok=True)
        ts = datetime.now().strftime("%m%d_%H%M%S")
        path = os.path.join("plots", f"track_max_{title_lines[0].split()[0]}_{ts}.png")
        plt.savefig(path)
        print(f"Convergence figure saved to {path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Save GIF helper
# ---------------------------------------------------------------------------


def save_gif(frames: list, objective_function: str) -> None:
    """
    Save the list of captured frames as an animated GIF under plots/.

    Parameters
    ----------
    frames : list of np.ndarray
        PNG frames as returned by ``animate_optimization``.
    objective_function : str
        Name of the objective function, used to label the output file.
    """
    os.makedirs("plots", exist_ok=True)
    ts = datetime.now().strftime("%m%d_%H%M%S")
    path = os.path.join("plots", f"bayes_opt_animation_{objective_function}_{ts}.gif")
    imageio.mimsave(path, frames, fps=2)
    print(f"Animation saved as {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_arguments()
    os.environ["MPLCONFIGDIR"] = os.getcwd()
    np.random.seed(args.seed)

    # --- setup objective function and grid ---
    synth_function = gpr.load_test_function(args.objective_function)
    bounds_low = [b[0] for b in synth_function._bounds]
    bounds_high = [b[1] for b in synth_function._bounds]

    x1 = np.linspace(bounds_low[0], bounds_high[0], 100)
    x2 = np.linspace(bounds_low[1], bounds_high[1], 100)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    x_grid = np.vstack([x1_grid.ravel(), x2_grid.ravel()]).T
    y_grid = np.array(
        [
            synth_function(torch.from_numpy(x.reshape(1, -1))).detach().numpy()
            for x in x_grid
        ]
    ).reshape(x1_grid.shape)

    global_optima, global_optimum_value = bo.get_synth_global_optima(
        args.objective_function
    )

    # --- sample initial points ---
    x_sample, y_sample = bo.sample_data(
        args.objective_function,
        bounds_low,
        bounds_high,
        args.n_initial,
        input_size=2,
    )

    # --- build optimizer ---
    bopt = bo.BayesianOptimizer(
        objective_function=args.objective_function,
        x_init=x_sample,
        y_init=y_sample,
        kernel=args.kernel,
        isotropic=args.isotropic,
        acquisition_function=args.acquisition,
        n_acquire=args.n_iteration,
        seed=args.seed,
        xi=args.xi,
        kappa=args.kappa,
    )
    gp_initial = bopt.gp_model_fit()

    # --- set up figure ---
    fig, axes, handles, meta = setup_figure(
        bopt_config=dict(y_sample=y_sample),
        x1_grid=x1_grid,
        x2_grid=x2_grid,
        y_grid=y_grid,
        x_sample=x_sample,
        synth_function=synth_function,
        global_optima=global_optima,
        objective_function=args.objective_function,
        kernel=args.kernel,
        n_initial=args.n_initial,
        n_iteration=args.n_iteration,
        acquisition=args.acquisition,
        xi=args.xi,
        kappa=args.kappa,
        gp_initial=gp_initial,
    )

    if not args.save_animation:
        plt.show(block=False)

    # --- run BO and animate ---
    snapshots = run_bayesian_optimization(
        bopt,
        x_grid,
        x1_grid,
        acquisition=args.acquisition,
        xi=args.xi,
        kappa=args.kappa,
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

    # --- save GIF if requested ---
    if args.save_animation and frames:
        save_gif(frames, args.objective_function)

    # --- convergence plot ---
    plot_convergence(
        acquired_maxima,
        gp_mean_maxima,
        global_optimum_value,
        title_lines=meta["title_lines"],
        save_animation=args.save_animation,
    )


if __name__ == "__main__":
    main()
