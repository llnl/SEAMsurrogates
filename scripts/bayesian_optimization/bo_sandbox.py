#!/usr/bin/env python3

"""
This script creates an animation of Bayesian Optimization (BO).

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
from datetime import datetime
import io

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import torch

from surmod import bayesian_optimization as bo
from surmod import gaussian_process_regression as gp


def parse_arguments():
    """Get command line arguments."""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Perform Bayesian optimization with GP surrogate models to maximize synthetic test functions.",
    )

    parser.add_argument(
        "-it",
        "--n_iteration",
        type=int,
        default=10,
        help="Choose number of iterations (data points to acquire)",
    )

    parser.add_argument(
        "-in",
        "--n_initial",
        type=int,
        default=10,
        help="Choose number of initial sample points",
    )

    parser.add_argument(
        "-k",
        "--kernel",
        type=str,
        choices=["matern", "rbf", "matern_dot"],
        default="matern",
        help="Choose kernel",
    )

    parser.add_argument(
        "-acq",
        "--acquisition",
        type=str,
        choices=["EI", "PI", "UCB", "random"],
        default="EI",
        help="Choose acquisition function",
    )

    parser.add_argument(
        "-f",
        "--objective_function",
        type=str,
        choices=[
            "Parabola",
            "Ackley",
            "Griewank",
            "Branin",
            "HolderTable",
        ],
        default="Parabola",
        help="Choose objective function",
    )

    parser.add_argument(
        "-i",
        "--isotropic",
        action="store_true",
        help="Specify that the kernel function is isotropic (same length scale for all inputs).",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Set random seed for reproducibility.",
    )

    parser.add_argument(
        "-save",
        "--save_animation",
        action="store_true",
        help="Option to save iterative plot as an animation.",
    )

    parser.add_argument(
        "-xi",
        "--xi",
        type=float,
        default=0.0,
        help="Exploration-exploitation trade-off parameter for EI and PI acquisition functions (non-negative float).",
    )

    parser.add_argument(
        "-kappa",
        "--kappa",
        type=float,
        default=2.0,
        help="Exploration-exploitation trade-off parameter for UCB acquisition function (non-negative float).",
    )

    args = parser.parse_args()

    return args


def main():
    # Parse command-line arguments
    args = parse_arguments()
    kernel = args.kernel
    n_initial = args.n_initial
    n_iteration = args.n_iteration
    acquisition = args.acquisition
    objective_function = args.objective_function
    isotropic = args.isotropic
    seed = args.seed
    save_animation = args.save_animation
    xi = args.xi
    kappa = args.kappa

    os.environ["MPLCONFIGDIR"] = os.getcwd()  # useful for Lightning AI

    # Initial set-up
    np.random.seed(seed)
    synth_function = gp.load_test_function(objective_function)
    bounds_low = [b[0] for b in synth_function._bounds]
    bounds_high = [b[1] for b in synth_function._bounds]

    # Sample initial points
    x_sample, y_sample = bo.sample_data(
        objective_function,
        bounds_low,
        bounds_high,
        n_initial,
        input_size=2,
    )

    # Create BayesianOptimizer instance
    bopt = bo.BayesianOptimizer(
        objective_function=objective_function,
        x_init=x_sample,
        y_init=y_sample,
        kernel=kernel,
        isotropic=isotropic,
        acquisition_function=acquisition,
        n_acquire=n_iteration,
        seed=seed,
        xi=xi,
        kappa=kappa,
    )

    # Fit the initial GP model
    model = bopt.gp_model_fit()

    # Generate a grid for contour plotting
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

    # Initialize lists to store maximum values
    acquired_maxima = np.array([])
    gp_mean_maxima = np.array([])

    # Create a figure with subplots
    fig = plt.figure(figsize=(18, 6))

    # Add a global title
    global_title = f"Bayesian Optimization of {objective_function} Objective Function w/ {kernel} kernel\n"
    fig.suptitle(global_title, fontsize=16)
    fig.text(0.5, 0.9, "", ha="center", fontsize=14)

    # Create subplots
    ax1 = fig.add_subplot(131, aspect="equal")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")

    title_lines = [
        f"{objective_function} with {kernel} kernel",
        f"Initial Samples: {n_initial} | Acquired Samples: {n_iteration}",
    ]

    # Plot the contours of the objective function on ax1
    # add 1 to axis limits to see clearly acquired points on boundary
    ax1.set_xlim(bounds_low[0] - 1, bounds_high[0] + 1)
    ax1.set_ylim(bounds_low[1] - 1, bounds_high[1] + 1)
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_title("\n".join(title_lines))
    contour = ax1.contourf(
        x1_grid, x2_grid, y_grid, levels=25, cmap="inferno", alpha=0.3
    )
    plt.colorbar(contour, ax=ax1, label="Value of " + objective_function + " function")
    ax1.scatter(
        x_sample[:, 0],
        x_sample[:, 1],
        marker="x",
        color="green",
        label="Initial samples",
    )

    # Add locations of global optima to first plot
    global_optima, global_optimum_value = bo.get_synth_global_optima(objective_function)
    for idx, point in enumerate(global_optima):
        ax1.scatter(
            point[0],
            point[1],
            marker="x",
            color="red",
            label="Global Maximum" if idx == 0 else "",
        )
    ax1.legend(loc="upper right")

    # Plot the initial acquisition function surface on ax2
    if acquisition == "EI":
        acquisition_values = bo.expected_improvement(
            x_grid,
            np.max(y_sample),
            model,
            xi=xi,
        )
    elif acquisition == "PI":
        acquisition_values = bo.probability_of_improvement(
            x_grid,
            model,
            np.max(y_sample),
            xi=xi,
        )
    elif acquisition == "UCB":
        acquisition_values = bo.upper_confidence_bound(
            x_grid,
            model,
            kappa=kappa,
        )
    elif acquisition == "random":
        acquisition_values = np.random.uniform(size=x_grid.shape[0])
    else:
        raise ValueError(
            "Invalid acquisition function. Choose 'EI', 'PI', 'UCB', or 'random."
        )
    acquisition_values = acquisition_values.reshape(x1_grid.shape)
    acquisition_surface = ax2.plot_surface(  # type: ignore
        x1_grid, x2_grid, acquisition_values, cmap="viridis"
    )
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_zlabel("Acquisition Value")  # type: ignore
    ax2.set_title("Acquisition Function")
    scatter = ax2.scatter(
        x_sample[:, 0],
        x_sample[:, 1],
        y_sample.flatten(),
        color="green",
        label="Sampled Points",
    )
    scatter.remove()  # Remove initial sample points to avoid clutter/distraction

    # Plot the initial GP mean surface on ax3
    mu = model.predict(x_grid, return_std=False)
    if isinstance(mu, tuple):
        mu = mu[0]  # take the mean
    mu = mu.reshape(x1_grid.shape)
    gp_mean_max_value = np.max(mu)
    gp_mean_max_location = x_grid[np.argmax(mu), :]
    gp_surface = ax3.plot_surface(x1_grid, x2_grid, mu, cmap="viridis", alpha=0.6)  # type: ignore
    gp_mean_max = ax3.scatter(
        gp_mean_max_location[0],
        gp_mean_max_location[1],
        gp_mean_max_value,
        color="red",
        s=50,  # type: ignore
        label="GP Mean Max",
    )
    ax3.set_xlabel("x1")
    ax3.set_ylabel("x2")
    ax3.set_zlabel("Value")  # type: ignore
    ax3.legend()
    ax3.set_title("Objective Function Contour and GP Mean Surface")
    ax3.contour(x1_grid, x2_grid, y_grid, levels=25, cmap="inferno", linestyles="solid")

    # Adjust layout to fit plots better
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4)
    plt.tight_layout()

    if not save_animation:
        plt.show(block=False)

    # Update plots as new acquisition sample point is added in the learning
    #   process

    frames = []

    for i in range(n_iteration):
        # Propose next point and evaluate via the class
        x_next = bopt.propose_location(bopt.acquisition)
        y_next = bopt.evaluate_objective(x_next)

        # Print acquired sample value and location
        print(
            f"\nIter. {i + 1}: acquired f(x)={y_next[0]:.3} at x=({x_next[0]:.3},{x_next[1]:.3})"
        )

        # Add the new sample to the optimizer's data
        bopt.x_all_data = np.vstack((bopt.x_all_data, x_next.reshape(1, -1)))
        bopt.y_all_data = np.append(bopt.y_all_data, y_next)
        bopt.x_acquired = np.append(bopt.x_acquired, [x_next], axis=0)
        bopt.y_acquired = np.append(bopt.y_acquired, y_next)
        y_max = np.max(bopt.y_all_data)
        bopt.y_max_history = np.append(bopt.y_max_history, y_max)

        # Plot the new sample on objective function plot
        ax1.scatter(
            x_next[0],
            x_next[1],
            color="blue",
            marker="s",
            label="Acquired point",
        )
        if i == 0:
            ax1.legend(loc="upper right")

        if save_animation:
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            img = imageio.imread(buf)
            frames.append(img)
            buf.close()
        else:
            plt.draw()
            plt.pause(0.6)

        # Re-fit the GP model
        model = bopt.gp_model_fit()

        # Update the acquisition function surface with the new sample
        if acquisition == "EI":
            acquisition_values = bo.expected_improvement(x_grid, y_max, model, xi=xi)
        elif acquisition == "PI":
            acquisition_values = bo.probability_of_improvement(
                x_grid, model, y_max, xi=xi
            )
        elif acquisition == "UCB":
            acquisition_values = bo.upper_confidence_bound(x_grid, model, kappa=kappa)
        elif acquisition == "random":
            acquisition_values = np.random.uniform(size=x_grid.shape[0])
        else:
            raise ValueError(
                "Invalid acquisition function. Choose 'EI', 'PI', 'UCB', or 'random."
            )

        acquisition_values = acquisition_values.reshape(x1_grid.shape)
        acquisition_surface.remove()

        # Plot acquisition function
        acquisition_surface = ax2.plot_surface(  # type: ignore
            x1_grid,
            x2_grid,
            acquisition_values,
            cmap="viridis",
        )

        if save_animation:
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            img = imageio.imread(buf)
            frames.append(img)
            buf.close()
        else:
            plt.draw()
            plt.pause(1.0)

        # Update the GP mean surface based on new sample
        mu = model.predict(x_grid, return_std=False)
        gp_mean_max_value = np.max(mu)
        gp_mean_max_location = x_grid[np.argmax(mu), :]
        if isinstance(mu, tuple):
            mu = mu[0]  # take the mean
        mu = mu.reshape(x1_grid.shape)
        gp_surface.remove()
        gp_mean_max.remove()

        # Highlight location of the maximum value of the GP mean on the surface
        gp_surface = ax3.plot_surface(x1_grid, x2_grid, mu, cmap="viridis", alpha=0.6)  # type: ignore
        gp_mean_max = ax3.scatter(
            gp_mean_max_location[0],
            gp_mean_max_location[1],
            gp_mean_max_value,
            color="red",
            s=50,  # type: ignore
            label="Maximum of GP Mean",
        )
        ax3.legend()

        if save_animation:
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            img = imageio.imread(buf)
            frames.append(img)
            buf.close()
        else:
            plt.draw()
            plt.pause(1.0)

        # Maximum of objective function
        x_best = bopt.x_all_data[np.argmax(bopt.y_all_data), :]

        # Print current black-box evaluated best max and GP mean max
        print(
            f"Iter. {i + 1}: max f(x)={y_max:.3} at x=({x_best[0]:.3},{x_best[1]:.3})"
        )
        print(
            f"Iter. {i + 1}: max GP mean={gp_mean_max_value:.3} at x=({gp_mean_max_location[0]:.3},{gp_mean_max_location[1]:.3})"
        )

        # Save the current best maxima
        acquired_maxima = np.append(acquired_maxima, np.max(bopt.y_acquired))
        gp_mean_maxima = np.append(gp_mean_maxima, gp_mean_max_value)

    if save_animation and frames:
        # Create plots folder if it doesn't exist
        plots_folder = "plots"
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        save_path = os.path.join(
            plots_folder,
            f"bayes_opt_animation_{objective_function}_{timestamp}.gif",
        )
        imageio.mimsave(save_path, frames, fps=2)
        print(f"Animation saved as {save_path}")

    # Create a single plot for the maximum values over iterations
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
    # Add green horizontal line for the true global optimum value
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
        if not os.path.exists("plots"):
            os.makedirs("plots")
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        filepath = os.path.join(
            "plots", f"track_max_{objective_function}_{timestamp}.png"
        )
        plt.savefig(filepath)
        print(f"Figure saved to {filepath}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
