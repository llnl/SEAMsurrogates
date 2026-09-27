#!/usr/bin/env python3
"""
This script simulates data from a test function, fits a Gaussian process to the
data, and saves a log message and plot of the fitted surface.

Usage examples:

./gp_sandbox.py --help
./gp_sandbox.py
./gp_sandbox.py --test-function parabola --kernel matern --isotropic
./gp_sandbox.py --test-function parabola --kernel matern
./gp_sandbox.py --test-function branin --kernel rbf --seed 1
./gp_sandbox.py --test-function ackley -k rbf -tr 200
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from surmod.gaussian_process import GPSurrogate
from surmod.test_functions import simulate_data


def parse_arguments():
    """Get command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train GP surrogate models on synthetic test functions.",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    experiment = parser.add_argument_group("experiment options")
    gp_options = parser.add_argument_group("GP model options")

    experiment.add_argument(
        "-f",
        "--test-function",
        type=str,
        default="parabola",
        help="Test function to use. Supported: parabola, ackley, branin, holder_table, griewank, six_hump_camel.",
    )

    experiment.add_argument(
        "-tr",
        "--n-train",
        type=int,
        default=100,
        help="Number of training samples.",
    )

    experiment.add_argument(
        "-te",
        "--n-test",
        type=int,
        default=100,
        help="Number of test samples.",
    )

    gp_options.add_argument(
        "-k",
        "--kernel",
        type=str,
        choices=["rbf", "matern", "periodic"],
        default="matern",
        help="GP kernel function.",
    )

    gp_options.add_argument(
        "-i",
        "--isotropic",
        action="store_true",
        help="Use isotropic kernel (single lengthscale for all inputs).",
    )

    gp_options.add_argument(
        "-sx",
        "--scale-x",
        action="store_true",
        default=False,
        help="Scale the input values to [0,1] per dimension using training data.",
    )

    gp_options.add_argument(
        "-ny",
        "--normalize-y",
        action="store_true",
        default=False,
        help="Standardize outputs (maps to GPSurrogate.scale_outputs).",
    )

    gp_options.add_argument(
        "--fixed-nugget",
        type=float,
        default=None,
        help="Fix the likelihood noise (nugget).",
    )

    gp_options.add_argument(
        "--lengthscale-bounds",
        type=float,
        nargs=2,
        default=(1e-2, 100.0),
        metavar=("LOW", "HIGH"),
        help="Bounds for kernel lengthscale constraint.",
    )

    gp_options.add_argument(
        "--noise-bounds",
        type=float,
        nargs=2,
        default=(1e-8, 1e-1),
        metavar=("LOW", "HIGH"),
        help="Bounds for likelihood noise constraint.",
    )

    return parser.parse_args()


def log_results(log_message: str, path_to_log: Path) -> None:
    path_to_log.parent.mkdir(parents=True, exist_ok=True)
    with open(path_to_log, "a", encoding="utf-8") as f:
        f.write(log_message + "\n")


def main():
    """Simulate data, train GP model, evaluate, and plot/log results."""
    args = parse_arguments()
    test_function = args.test_function
    kernel = args.kernel
    n_train = args.n_train
    n_test = args.n_test
    scale_x = args.scale_x
    normalize_y = args.normalize_y
    fixed_nugget = args.fixed_nugget
    isotropic = args.isotropic
    lengthscale_bounds = tuple(args.lengthscale_bounds)
    noise_bounds = tuple(args.noise_bounds)
    seed = args.seed

    # Define script-relative directories
    plots_dir = Path(__file__).parent / "plots"

    # Generate test and train data sets
    x_train, x_test, y_train, y_test = simulate_data(
        test_function,
        n_train,
        n_test,
        seed=seed,
    )

    # Handle fixed nugget
    fixed_noise = float(fixed_nugget) if fixed_nugget is not None else None
    noise_bounds_to_use = None if fixed_noise is not None else noise_bounds

    gp = GPSurrogate(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        kernel=kernel,
        isotropic=isotropic,
        scale_inputs=scale_x,
        scale_outputs=normalize_y,
        fixed_noise=fixed_noise,
        lengthscale_bounds=lengthscale_bounds,
        noise_bounds=noise_bounds_to_use,
        seed=seed,
    )

    start_time = time.perf_counter()
    gp.fit()
    elapsed_time = time.perf_counter() - start_time

    pred_train_mean, _pred_train_std = gp.predict(x_train)
    pred_test_mean, pred_test_std = gp.predict(x_test)

    train_mae = mean_absolute_error(y_train, pred_train_mean)
    test_mae = mean_absolute_error(y_test, pred_test_mean)

    train_rmse = root_mean_squared_error(y_train, pred_train_mean)
    test_rmse = root_mean_squared_error(y_test, pred_test_mean)

    train_max_abserr, train_max_input = gp.compute_max_error(
        pred_train_mean, y_train, x_train
    )
    test_max_abserr, test_max_input = gp.compute_max_error(
        pred_test_mean, y_test, x_test
    )
    fitted_params = gp.get_fitted_parameters()
    lower = pred_test_mean - 1.96 * pred_test_std
    upper = pred_test_mean + 1.96 * pred_test_std
    coverage = np.mean((y_test >= lower) & (y_test <= upper))

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    log_lines = [
        f"Run timestamp (%m%d_%H%M%S): {timestamp}",
        f"Test Function: {test_function}",
        f"Number of training points: {n_train}",
        f"Number of testing points: {n_test}",
        f"Kernel: {kernel}",
        f"Isotropic kernel: {isotropic}",
        f"Learned noise: {fitted_params.get('noise')}",
        f"Learned outputscale: {fitted_params.get('outputscale')}",
        f"Learned lengthscale(s): {fitted_params.get('lengthscale')}",
        f"Scale x: {scale_x}",
        f"Normalize y: {normalize_y}",
        f"Fixed nugget: {fixed_nugget}",
        f"Lengthscale bounds: {lengthscale_bounds}",
        f"Noise bounds: {noise_bounds_to_use if fixed_noise is None else 'N/A (fixed)'}",
        f"Train RMSE: {train_rmse:.5e}",
        f"Test RMSE: {test_rmse:.5e}",
        f"Test 95% interval coverage: {coverage:.2%}",
        f"Train Max abs err:  {train_max_abserr:.5e} | Location: {train_max_input}",
        f"Test Max abs err:   {test_max_abserr:.5e} | Location: {test_max_input}",
        f"Train Mean abs err: {train_mae:.5e}",
        f"Test Mean abs err:  {test_mae:.5e}",
        f"Elapsed time for training GP: {elapsed_time:.3f} seconds\n",
    ]
    log_message = "\n".join(log_lines)
    print(log_message)

    results_dir = Path(__file__).parent / "results"
    log_results(
        log_message,
        path_to_log=results_dir
        / f"{test_function}_{kernel}_nugget-{fixed_nugget if fixed_nugget is not None else 'learned'}.txt",
    )

    gp.plot_test_predictions(dataset=test_function, plots_dir=plots_dir)

    gp.plot_predictive_mean(
        test_rmse=test_rmse,
        test_function=test_function,
        scale_x=scale_x,
        normalize_y=normalize_y,
        plots_dir=plots_dir,
    )

    gp.plot_predictive_std_dev(
        test_rmse=test_rmse,
        test_function=test_function,
        scale_x=scale_x,
        normalize_y=normalize_y,
        plots_dir=plots_dir,
    )


if __name__ == "__main__":
    main()
