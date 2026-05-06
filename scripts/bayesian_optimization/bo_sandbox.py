#!/usr/bin/env python3

"""
This script creates an animation of Bayesian Optimization on a chosen
synthetic objective function and plots performance of the chosen acquisition
function: Expected Improvement (EI), Probability of Improvement (PI),
Upper Confidence Bound (UCB), Predictive Variance (PV), or random.
"""

import argparse
import io
import os
from datetime import datetime
from typing import Generator

import imageio.v2 as imageio
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import torch

from surmod import bayesian_optimization as bo
from surmod.gpytorch_gaussian_process import GPSurrogate
from surmod.test_functions import load_test_function


def parse_arguments() -> argparse.Namespace:
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
        choices=["matern", "rbf", "periodic"],
        default="matern",
    )
    parser.add_argument(
        "-acq",
        "--acquisition",
        type=str,
        choices=["EI", "PI", "UCB", "PV", "random"],
        default="EI",
    )
    parser.add_argument(
        "-f",
        "--objective_function",
        type=str,
        choices=["Parabola", "Ackley", "Griewank", "Branin", "HolderTable"],
        default="Parabola",
    )
    parser.add_argument(
        "--init_design",
        type=str,
        choices=["random", "lhd", "maximin_lhd"],
        default="random",
        help="Initial design strategy for BO.",
    )
    parser.add_argument("-i", "--isotropic", action="store_true")
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-save", "--save_animation", action="store_true")
    parser.add_argument("-beta", "--beta", type=float, default=2.0)
    return parser.parse_args()


def run_bayesian_optimization(
    bopt: bo.BayesianOptimizer,
    x_grid: np.ndarray,
    x1_grid: np.ndarray,
) -> Generator[dict, None, None]:
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
            f"\nIter. {i+1}: acquired f(x)={y_next_scalar:.3g} at x=({x_next[0]:.3g},{x_next[1]:.3g})"
        )
        print(
            f"Iter. {i+1}: max f(x)={y_max:.3g} at x=({x_best[0]:.3g},{x_best[1]:.3g})"
        )
        print(
            f"Iter. {i+1}: max GP mean={gp_mean_max_value:.3g} "
            f"at x=({gp_mean_max_location[0]:.3g},{gp_mean_max_location[1]:.3g})"
        )

        yield snapshot


def _capture_frame(fig: matplotlib.figure.Figure, frames: list) -> None:
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
    objective_function: str,
    kernel: str,
    n_initial: int,
    n_iteration: int,
    gp_initial: object,
) -> tuple[matplotlib.figure.Figure, dict, dict, dict]:

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

    x_grid = np.vstack([x1_grid.ravel(), x2_grid.ravel()]).T

    bopt.gp_model = gp_initial
    acq_init = bopt.score_candidates(x_grid)

    acq_init = acq_init.reshape(x1_grid.shape)
    acq_surface = ax2.plot_surface(x1_grid, x2_grid, acq_init, cmap="viridis")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_zlabel("Acquisition Value")
    ax2.set_title("Acquisition Function")

    mu_init, _ = gp_initial.predict(x_grid)
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

    handles = dict(
        acq_surface=acq_surface, gp_surface=gp_surface, gp_mean_dot=gp_mean_dot
    )
    axes = dict(ax1=ax1, ax2=ax2, ax3=ax3)
    meta = dict(title_lines=title_lines)

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
            x1_grid, x2_grid, snap["acq_values"], cmap="viridis"
        )

        if save_animation:
            _capture_frame(fig, frames)
        else:
            plt.draw()
            plt.pause(1.0)

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


def plot_convergence(
    acquired_maxima: np.ndarray,
    gp_mean_maxima: np.ndarray,
    global_optimum_value: float,
    title_lines: list[str],
    save_animation: bool,
) -> None:
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


def save_gif(frames: list, objective_function: str) -> None:
    os.makedirs("plots", exist_ok=True)
    ts = datetime.now().strftime("%m%d_%H%M%S")
    path = os.path.join("plots", f"bayes_opt_animation_{objective_function}_{ts}.gif")
    imageio.mimsave(path, frames, fps=2)
    print(f"Animation saved as {path}")


def main() -> None:
    args = parse_arguments()
    os.environ["MPLCONFIGDIR"] = os.getcwd()
    np.random.seed(args.seed)

    synth_function = load_test_function(args.objective_function)
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

    global_optima, global_optimum_value = bo.get_synth_global_optima(
        args.objective_function
    )

    x_sample, y_sample = bo.sample_data(
        args.objective_function,
        bounds_low,
        bounds_high,
        args.n_initial,
        input_size=2,
        init_design=args.init_design,
        seed=args.seed,
    )

    bopt = bo.BayesianOptimizer(
        objective_function=args.objective_function,
        x_init=x_sample,
        y_init=y_sample,
        kernel=args.kernel,
        isotropic=args.isotropic,
        acquisition_function=args.acquisition,
        n_acquire=args.n_iteration,
        seed=args.seed,
        beta=args.beta,
    )

    gp_initial = GPSurrogate(
        x_train=x_sample,
        y_train=y_sample,
        kernel=args.kernel,
        isotropic=args.isotropic,
        scale_inputs=True,
        scale_outputs=True,
    )
    gp_initial.fit()

    fig, axes, handles, meta = setup_figure(
        bopt=bopt,
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
        gp_initial=gp_initial,
    )

    if not args.save_animation:
        plt.show(block=False)

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

    if args.save_animation and frames:
        save_gif(frames, args.objective_function)

    plot_convergence(
        acquired_maxima,
        gp_mean_maxima,
        global_optimum_value,
        title_lines=meta["title_lines"],
        save_animation=args.save_animation,
    )


if __name__ == "__main__":
    main()
