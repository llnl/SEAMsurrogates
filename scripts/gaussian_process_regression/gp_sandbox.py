#!/usr/bin/env python3
"""
This script simulates data from a test function, fits a Gaussian process to the
data, and saves a log message and plot of the fitted surface if desired.

"""


import argparse
import itertools
import os
import time
from datetime import datetime

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from surmod.test_functions import simulate_data

from surmod.gpytorch_gaussian_process import GPSurrogate


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
        "--num_train",
        type=int,
        default=100,
        help="Number of points to have in training data set.",
    )

    parser.add_argument(
        "-te",
        "--num_test",
        type=int,
        default=100,
        help="Number of points to have in testing data set.",
    )

    parser.add_argument(
        "-nx",
        "--normalize_x",
        action="store_true",
        default=False,
        help="Normalize the input values to mean 0 and unit variance per dimension using training data.",
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
        help="Fix the likelihood noise (nugget). Implemented by setting noise_bounds to nugget +/- nugget/10000.",
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
        help="Specify that the kernel function is isotropic (same length scale for all inputs).",
    )

    return parser.parse_args()


def log_results(log_message: str, path_to_log: str) -> None:
    os.makedirs(os.path.dirname(path_to_log), exist_ok=True)
    with open(path_to_log, "a", encoding="utf-8") as f:
        f.write(log_message + "\n")


def nugget_to_bounds(nugget: float) -> tuple[float, float]:
    if nugget <= 0.0:
        raise ValueError("--fixed_nugget must be > 0.")
    delta = 1e-16
    low = max(nugget - delta, 1e-20)
    high = nugget + delta
    return (low, high)


def main():
    """Simulate data, train GP model, evaluate, and plot/log results."""
    args = parse_arguments()

    objective_function = args.objective_function
    kernels = args.kernels
    num_train = args.num_train
    num_test = args.num_test
    normalize_x = args.normalize_x
    scale_x = args.scale_x
    normalize_y = args.normalize_y
    fixed_nugget = args.fixed_nugget
    plots = args.plots
    do_log = args.log
    isotropic = args.isotropic

    # Generate test and train data sets
    x_train, x_test, y_train, y_test = simulate_data(
        objective_function,
        num_train,
        num_test,
    )

    y_train_1d = np.asarray(y_train).reshape(-1)
    y_test_1d = np.asarray(y_test).reshape(-1)

    scaler_x_train = None

    if normalize_x and scale_x:
        raise ValueError("Choose either normalize_x or scale_x, not both.")

    if normalize_x or scale_x:
        if normalize_x:
            print(
                "Input data is being normalized to have mean 0, variance 1, in each dimension based on training data.\n"
            )
            scaler_x_train = StandardScaler()

        if scale_x:
            print(
                "Input data is being scaled using min-max scaling in each dimension based on training data.\n"
            )
            scaler_x_train = MinMaxScaler()

        scaler_x_train.fit(x_train)  # type: ignore
        x_train = scaler_x_train.transform(x_train)  # type: ignore
        x_test = scaler_x_train.transform(x_test)  # type: ignore

    noise_bounds = None
    if fixed_nugget is not None:
        noise_bounds = nugget_to_bounds(float(fixed_nugget))

    for kernel in kernels:
        gp_model = GPSurrogate(
            x_train=x_train,
            y_train=y_train_1d,
            x_test=x_test,
            y_test=y_test_1d,
            kernel=kernel,
            isotropic=isotropic,
            # Inputs already optionally normalized/scaled above, avoid double scaling
            scale_inputs=False,
            scale_outputs=normalize_y,
            noise_bounds=noise_bounds if noise_bounds is not None else (1e-16, 1e-1),
        )

        start_time = time.perf_counter()
        gp_model.fit()
        elapsed_time = time.perf_counter() - start_time

        pred_train_mean, _pred_train_std = gp_model.predict(x_train)
        pred_test_mean, pred_test_std = gp_model.predict(x_test)

        train_mae = float(mean_absolute_error(y_train_1d, pred_train_mean))
        test_mae = float(mean_absolute_error(y_test_1d, pred_test_mean))

        train_mse = float(mean_squared_error(y_train_1d, pred_train_mean))
        test_mse = float(mean_squared_error(y_test_1d, pred_test_mean))

        train_max_abserr, train_max_input = gp_model.compute_max_error(
            pred_train_mean, y_train_1d, x_train
        )
        test_max_abserr, test_max_input = gp_model.compute_max_error(
            pred_test_mean, y_test_1d, x_test
        )

        lower = pred_test_mean - 1.96 * pred_test_std
        upper = pred_test_mean + 1.96 * pred_test_std
        coverage = float(np.mean((y_test_1d >= lower) & (y_test_1d <= upper)))

        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        log_lines = [
            f"Run timestamp (%m%d_%H%M%S): {timestamp}",
            f"Test Function: {objective_function}",
            f"Number of training points: {num_train}",
            f"Number of testing points: {num_test}",
            f"Kernel: {kernel}",
            f"Isotropic kernel: {isotropic}",
            f"Normalize x values: {normalize_x}",
            f"Scale x values: {scale_x}",
            f"Standardize outputs (normalize_y): {normalize_y}",
            f"Fixed nugget: {fixed_nugget}",
            f"Noise bounds: {noise_bounds if noise_bounds is not None else (1e-16, 1e-1)}",
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
                path_to_log=os.path.join(
                    "output_log",
                    f"{objective_function}_{kernel}_nugget-{fixed_nugget if fixed_nugget is not None else 'learned'}.txt",
                ),
            )

        if plots:
            gp_model.plot_test_predictions(objective_data_name=objective_function)


if __name__ == "__main__":
    main()