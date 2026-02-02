#!/usr/bin/env python3

"""
This script trains a Gaussian Process (GP) surrogate model on a chosen dataset.
It provides options for different kernel types, isotropic/anisotropic kernels,
normalization, plotting, and logging results.

Usage:

# Make script executable
chmod +x ./gp_fromdata.py

# See help for all options
./gp_fromdata.py -h

# Example usages:

# Train a GP with 200 training points and an isotropic RBF kernel
./gp_fromdata.py --num_train=200 --kernel=rbf --isotropic

# Train a GP with 200 training points and an anisotropic RBF kernel
./gp_fromdata.py --num_train=200 --kernel=rbf

# Train a GP with 200 training points, Matern kernel,
# normalize y, and plot results
./gp_fromdata.py --num_train=200 --kernel=matern --normalize_y --plot

# Train a GP with 300 training points, 300 train samples, Matern kernel,
# and save results to a log file
./gp_fromdata.py --num_train=300 --kernel=matern --log

"""

import argparse
import os
import time
import datetime

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from surmod import gaussian_process_regression as gp, data_processing


def parse_arguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="A script to train GP surrogate models for the JAG dataset.",
    )

    parser.add_argument(
        "-d",
        "--data",
        type=str,
        choices=["JAG", "borehole"],
        default = "JAG",
        help="Which dataset to use (defualt: JAG)."
    )

    parser.add_argument(
        "-tr",
        "--num_train",
        type=int,
        default=400,
        help="Number of train samples (default: 400).",
    )

    parser.add_argument(
        "-te",
        "--num_test",
        type=int,
        default=100,
        help="Number of test samples (default: 100).",
    )

    parser.add_argument(
        "-ny",
        "--normalize_y",
        action="store_true",
        help="Whether or not to normalize the output values in the"
        " GaussianProcessRegressor.",
    )

    parser.add_argument(
        "-k",
        "--kernel",
        type=str,
        choices=["matern", "rbf", "matern_dot"],
        default="matern",
        help="Invalid choice of kernel function.",
    )

    parser.add_argument(
        "-i",
        "--isotropic",
        action="store_true",
        help="Specify that the kernel function is isotropic (same length scale"
        " for all inputs).",
    )

    parser.add_argument(
        "-l",
        "--log",
        action="store_true",
        help="Save output in file based on objective function and kernel; if"
        " file already exists, new runs will be appended to end of existing file.",
    )

    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Plot the gp prediction and true output parity plot"
        "values with 1.96*sd confidence bars.",
    )

    parser.add_argument(
        "--LHD",
        action="store_true",
        help="Use an LHD design.",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random number generator seed.",
    )

    args = parser.parse_args()

    return args


def main():
    """
    Trains and evaluates a Gaussian Process (GP) surrogate model on the JAG ICF
    dataset.
    """
    # Parse command line arguments
    args = parse_arguments()
    data = args.data
    num_train = args.num_train
    num_test = args.num_test
    num_test = args.num_test
    normalize_y = args.normalize_y
    kernel = args.kernel
    isotropic = args.isotropic
    log = args.log
    plot = args.plot
    seed = args.seed

    # Check data availability
    num_samples = num_test + num_train
    if num_samples > 10000:
        raise ValueError(
            f"Requested samples ({num_samples}) exceed existing dataset(s) size "
            "limit (10000)."
        )

    # Load and split data
    df = data_processing.load_data(dataset= data, n_samples=num_samples, random=False)
    x_train, x_test, y_train, y_test = data_processing.split_data(
        df=df, LHD=False, n_train=num_train, seed=seed
    )

    # Instantiate GP model
    gp_model = GaussianProcessRegressor(
        kernel=gp.get_kernel(kernel, x_train.shape[1], isotropic),
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
    pred_train = gp_model.predict(x_train)
    pred_test = gp_model.predict(x_test)

    # Evaluate Mean Absolute Error (MAE) with trained GP model
    train_mae = mean_absolute_error(y_train, pred_train)
    test_mae = mean_absolute_error(y_test, pred_test)

    # Evaluate Mean Square Error (MSE) with trained GP model
    train_mse = mean_squared_error(y_train, pred_train)
    test_mse = mean_squared_error(y_test, pred_test)

    # Evaluate Maximum Absolute Error (MSE) with trained GP model
    train_max_abserr, train_max_input = gp.compute_max_error(
        pred_train, y_train, x_train  # type: ignore
    )

    test_max_abserr, test_max_input = gp.compute_max_error(pred_test, y_test, x_test)  # type: ignore

    # Prepare the log message
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_lines = [
        f"Run timestamp (%m%d_%H%M%S): {timestamp}",
        f"Test Function: {data} ",
        f"Number of training points: {num_train}",
        f"Number of testing points: {num_test}",
        f"Kernel: {gp_model.kernel_}",
        f"Isotropic kernel: {isotropic}",
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

    if log:
        gp.log_results(
            log_message,
            path_to_log=os.path.join("output_log", f"{data}_Results.txt"),
        )

    if plot:
        gp.plot_test_predictions(x_test, y_test, gp_model, objective_data_name=data)


if __name__ == "__main__":
    main()
