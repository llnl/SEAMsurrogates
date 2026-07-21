#!/usr/bin/env python3
"""
This script simulates data from a test function, fits a Gaussian process to the
data, and saves a log message and plot of the fitted surface if desired.

Usage:

# Make script executable
chmod +x ./gp_sandbox.py

# See help.
./gp_sandbox.py -h

# Smooth parabola function with an isotropic Matern kernel.
./gp_sandbox.py --objective_function=Parabola --kernels=matern --isotropic --plots

# Smooth parabola function with an anisotropic Matern kernel.
./gp_sandbox.py --objective_function=Parabola --kernels=matern --plots

# Smooth Branin test function with an RBF kernel.
./gp_sandbox.py --objective_function=Branin --kernels=rbf --seed 1 --plots

# Smooth Ackley function with an RBF kernel, save results in log, 200 training
#   points, 3 values of alpha.
./gp_sandbox.py --objective_function=Ackley -k rbf -p -l -tr 200 -a 0.001 0.01 0.1

# Smooth HolderTable function with RBF and Matern kernels and 3 values of alpha.
#   Save plot and log file.
./gp_sandbox.py -f "HolderTable" -k rbf matern -p -l -a 0.002 0.04 0.08
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from surmod.test_functions import simulate_data

from surmod.gaussian_process import GPSurrogate


def parse_arguments():
    """Get command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="A script to train GP surrogate models on synthetic test functions (BoTorch GPSurrogate).",
    )

    parser.add_argument(
        "-f",
        "--objective_function",
        type=str,
        choices=["Parabola", "Ackley", "Branin", "HolderTable", "Griewank"],
        default="Parabola",
        help="Choose objective function.",
    )

    parser.add_argument(
        "-tr",
        "--n_train",
        type=int,
        default=100,
        help="Number of points to have in training data set.",
    )

    parser.add_argument(
        "-te",
        "--n_test",
        type=int,
        default=100,
        help="Number of points to have in testing data set.",
    )

    parser.add_argument(
        "-sx",
        "--scale_x",
        action="store_true",
        default=False,
        help="Scale the input values to [0,1] per dimension using training data.",
    )

    parser.add_argument(
        "-ny",
        "--normalize_y",
        action="store_true",
        default=False,
        help="Standardize outputs (maps to GPSurrogate.scale_outputs).",
    )

    parser.add_argument(
        "--fixed_nugget",
        type=float,
        default=None,
        help="Fix the likelihood noise (nugget).",
    )

    parser.add_argument(
        "-k",
        "--kernels",
        type=str,
        nargs="+",
        choices=["matern", "rbf", "periodic"],
        default=["matern"],
        help="Choice of kernel function from 'rbf', 'matern', or 'periodic'.",
    )

    parser.add_argument(
        "-l",
        "--log",
        action="store_true",
        help="Save output in file based on objective function and kernel; if file exists, append.",
    )

    parser.add_argument(
        "-p",
        "--plots",
        action="store_true",
        help="Save parity plot (observed vs predicted) with 95 percent intervals.",
    )

    parser.add_argument(
        "-i",
        "--isotropic",
        action="store_true",
        help="Specify that the kernel function is isotropic (same length scale "
        "for all inputs).",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )

    return parser.parse_args()


def log_results(log_message: str, path_to_log: Path) -> None:
    path_to_log.parent.mkdir(parents=True, exist_ok=True)
    with open(path_to_log, "a", encoding="utf-8") as f:
        f.write(log_message + "\n")


def main():
    """Simulate data, train GP model, evaluate, and plot/log results."""
    args = parse_arguments()
    objective_function = args.objective_function
    kernels = args.kernels
    n_train = args.n_train
    n_test = args.n_test
    scale_x = args.scale_x
    normalize_y = args.normalize_y
    fixed_nugget = args.fixed_nugget
    plots = args.plots
    do_log = args.log
    isotropic = args.isotropic
    seed = args.seed

    # Generate test and train data sets
    x_train, x_test, y_train, y_test = simulate_data(
        objective_function,
        n_train,
        n_test,
        seed=seed,
    )

    y_train_1d = np.asarray(y_train).reshape(-1)
    y_test_1d = np.asarray(y_test).reshape(-1)

    if fixed_nugget is not None:
        fixed_noise = float(fixed_nugget)
        eps = max(1e-8, abs(fixed_noise) * 1e-6)
        noise_bounds = (fixed_noise - eps, fixed_noise + eps)
    else:
        fixed_noise = None
        noise_bounds = (1e-8, 1e-1)

    for kernel in kernels:
        gp = GPSurrogate(
            x_train=x_train,
            y_train=y_train_1d,
            x_test=x_test,
            y_test=y_test_1d,
            kernel=kernel,
            isotropic=isotropic,
            scale_inputs=scale_x,
            scale_outputs=normalize_y,
            fixed_noise=fixed_noise,
            noise_bounds=noise_bounds,
        )

        start_time = time.perf_counter()
        gp.fit()
        elapsed_time = time.perf_counter() - start_time

        pred_train_mean, _pred_train_std = gp.predict(x_train)
        pred_test_mean, pred_test_std = gp.predict(x_test)

        train_mae = mean_absolute_error(y_train_1d, pred_train_mean)
        test_mae = mean_absolute_error(y_test_1d, pred_test_mean)

        train_mse = mean_squared_error(y_train_1d, pred_train_mean)
        test_mse = mean_squared_error(y_test_1d, pred_test_mean)

        train_max_abserr, train_max_input = gp.compute_max_error(
            pred_train_mean, y_train_1d, x_train
        )
        test_max_abserr, test_max_input = gp.compute_max_error(
            pred_test_mean, y_test_1d, x_test
        )
        fitted_params = gp.get_fitted_parameters()
        lower = pred_test_mean - 1.96 * pred_test_std
        upper = pred_test_mean + 1.96 * pred_test_std
        coverage = np.mean((y_test_1d >= lower) & (y_test_1d <= upper))

        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        log_lines = [
            f"Run timestamp (%m%d_%H%M%S): {timestamp}",
            f"Test Function: {objective_function}",
            f"Number of training points: {n_train}",
            f"Number of testing points: {n_test}",
            f"Kernel: {kernel}",
            f"Isotropic kernel: {isotropic}",
            f"Learned noise: {fitted_params.get('noise')}",
            f"Learned outputscale: {fitted_params.get('outputscale')}",
            f"Learned lengthscale(s): {fitted_params.get('lengthscale')}",
            f"Scale x values: {scale_x}",
            f"Standardize outputs (normalize_y): {normalize_y}",
            f"Fixed nugget: {fixed_nugget}",
            f"Noise bounds: {noise_bounds if noise_bounds is not None else (1e-8, 1e-1)}",
            f"Train MSE: {train_mse:.5e}",
            f"Test MSE: {test_mse:.5e}",
            f"Test 95% interval coverage: {coverage:.2%}",
            f"Train Max abs err:  {train_max_abserr:.5e} | Location: {train_max_input}",
            f"Test Max abs err:   {test_max_abserr:.5e} | Location: {test_max_input}",
            f"Train Mean abs err: {train_mae:.5e}",
            f"Test Mean abs err:  {test_mae:.5e}",
            f"Elapsed time for training GP: {elapsed_time:.3f} seconds\n",
        ]
        log_message = "\n".join(log_lines)
        print(log_message)

        if do_log:
            log_results(
                log_message,
                path_to_log=Path("output_log")
                / f"{objective_function}_{kernel}_nugget-{fixed_nugget if fixed_nugget is not None else 'learned'}.txt",
            )

        if plots:
            gp.plot_test_predictions(dataset=objective_function)

        gp.plot_predictive_mean(
            test_mse=test_mse,
            objective_function=objective_function,
            scale_x=scale_x,
            normalize_y=normalize_y,
        )

        gp.plot_predictive_std_dev(
            test_mse=test_mse,
            objective_function=objective_function,
            scale_x=scale_x,
            normalize_y=normalize_y,
        )


if __name__ == "__main__":
    main()
