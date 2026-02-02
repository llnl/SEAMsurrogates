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
./gp_sandbox.py --objective_function=Branin --kernels=rbf --plots

# Smooth Ackley function with an RBF kernel, save results in log, 200 training
#   points, 3 values of alpha.
./gp_sandbox.py --objective_function=Ackley -k rbf -p -l -tr 200 -a 0.001 0.01 0.1

# Smooth HolderTable function with RBF and Matern kernels and 3 values of alpha.
#   Save plot and log file.
./gp_sandbox.py -f "HolderTable" -k rbf matern -p -l -a 0.002 0.04 0.08
"""

import argparse
import itertools
import os
import time
import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from surmod import gaussian_process_regression as gp


def parse_arguments():
    """Get command line arguments."""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="A script to train GP surrogate models on synthetic test"
        "functions.",
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
        "-sx",
        "--scale_x",
        action="store_true",
        default=False,
        help="Whether or not to scale the input x values using min max scaling.",
    )

    parser.add_argument(
        "-ny",
        "--normalize_y",
        action="store_true",
        default=False,
        help="Whether or not to normalize the output y values in the"
        " GaussianProcessRegressor.",
    )

    parser.add_argument(
        "-a",
        "--alphas",
        type=float,
        nargs="+",
        default=[1e-5],
        help="Specify one or more variances of additional Gaussian measurement "
        "noise on the training observations (e.g., 1e-10 1e-08 1e-06).",
    )

    parser.add_argument(
        "-k",
        "--kernels",
        type=str,
        nargs="+",
        choices=["matern", "rbf", "matern_dot"],
        default=["matern"],
        help="Choice of kernel function from 'rbf', 'matern', or 'matern_dot'.",
    )

    parser.add_argument(
        "-l",
        "--log",
        action="store_true",
        help="Save output in file based on objective function and kernel; if "
        "file already exists, new runs will be appended to end of existing file.",
    )

    parser.add_argument(
        "-p",
        "--plots",
        action="store_true",
        help="Plot the objective function contour with the GP mean surface, and "
        "GP standard deviation heatmap.",
    )

    parser.add_argument(
        "-i",
        "--isotropic",
        action="store_true",
        help="Specify that the kernel function is isotropic (same length scale "
        "for all inputs).",
    )

    args = parser.parse_args()

    return args


def main():
    """Simulate data from test function, train GP model, predict model on
    hold-out data, and plot or log results.
    """

    # Parse command-line arguments
    args = parse_arguments()
    objective_function = args.objective_function
    kernels = args.kernels
    alphas = args.alphas
    num_train = args.num_train
    num_test = args.num_test
    scale_x = args.scale_x
    normalize_y = args.normalize_y
    plots = args.plots
    log = args.log
    isotropic = args.isotropic

    # Generate test and train data sets
    x_train, x_test, y_train, y_test = gp.simulate_data(
        objective_function,
        num_train,
        num_test,
    )

    # Initialize for plotting purposes
    scaler_x_train = None
    if scale_x:
        # Create the scaler and fit it on training data
        scaler_x_train = MinMaxScaler()
        scaler_x_train.fit(x_train)

        # Transform both train and test sets
        x_train = scaler_x_train.transform(x_train)
        x_test = scaler_x_train.transform(x_test)

    for kernel, alpha in itertools.product(kernels, alphas):
        # Instantiate GP model
        gp_model = GaussianProcessRegressor(
            kernel=gp.get_kernel(kernel, 2, isotropic),
            alpha=alpha,
            n_restarts_optimizer=5,
            random_state=42,
            normalize_y=normalize_y,
        )

        # Train GP model
        start_time = time.perf_counter()
        gp_model.fit(x_train, y_train)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # Evaluate GP model at train and test inputs
        pred_train = gp_model.predict(x_train, return_std=False)
        pred_test = gp_model.predict(x_test, return_std=False)

        # Evaluate Mean Absolute Error (MAE) with trained GP model
        train_mae = mean_absolute_error(y_train, pred_train)
        test_mae = mean_absolute_error(y_test, pred_test)

        # Evaluate Mean Square Error (MSE) with trained GP model
        train_mse = mean_squared_error(y_train, pred_train)
        test_mse = mean_squared_error(y_test, pred_test)

        # Evaluate Maximum Absolute Error (MaAE) with trained GP model
        train_max_abserr, train_max_input = gp.compute_max_error(
            pred_train, y_train, x_train  # type: ignore
        )
        test_max_abserr, test_max_input = gp.compute_max_error(
            pred_test, y_test, x_test  # type: ignore
        )

        # Prepare the log message
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
        log_lines = [
            f"Run timestamp (%m%d_%H%M%S): {timestamp}",
            f"Test Function: {objective_function}",
            f"Number of training points: {num_train}",
            f"Number of testing points: {num_test}",
            f"Kernel: {gp_model.kernel_}",
            f"Regularization value alpha: {alpha}",
            f"Scale x values: {scale_x}",
            f"Normalize y values: {normalize_y}",
            f"Train MSE: {train_mse:.5e}",
            f"Test MSE: {test_mse:.5e}",
            f"Train Max abs err:  {train_max_abserr:.5e} | Location: {train_max_input}",
            f"Test Max abs err:   {test_max_abserr:.5e} | Location: {test_max_input}",
            f"Train Mean abs err: {train_mae:.5e}",
            f"Test Mean abs err:  {test_mae:.5e}",
            f"Elapsed time for training GP: {elapsed_time:.3f} seconds\n",
        ]

        log_message = "\n".join(log_lines)

        print(log_message)

        gp.plot_test_predictions(x_test, y_test, gp_model, objective_function)

        if log:
            gp.log_results(
                log_message,
                path_to_log=os.path.join(
                    "output_log", f"{objective_function}_{kernel}_alpha-{alpha}.txt"
                ),
            )

        if plots:

            gp.plot_gp_mean_prediction(
                x_train,
                y_train,
                gp_model,
                test_mse,
                kernel,
                objective_function,
                alpha,
                scale_x,
                normalize_y,
                input_scaler=scaler_x_train,
            )

            gp.plot_gp_std_dev_prediction(
                x_train,
                gp_model,
                test_mse,
                kernel,
                objective_function,
                alpha,
                scale_x,
                normalize_y,
                input_scaler=scaler_x_train,
            )


if __name__ == "__main__":
    main()
