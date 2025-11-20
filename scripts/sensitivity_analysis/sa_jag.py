#!/usr/bin/env python3

"""
This script performs a sensitivity analysis on the JAG dataset by training a
Gaussian Process (GP) surrogate model. It allows for flexible kernel selection,
length scale adjustment, and exclusion of specific input columns.
The script evaluates model performance, computes Sobol sensitivity indices,
and saves relevant plots.

Note:
- For JAG data there are 5 input variables: x1, x2, x3, x4, x5.
- Column exclusion uses zero-based indexing:
    0 = x1, 1 = x2, 2 = x3, 3 = x4, 4 = x5.

Usage:

# Make script executable
chmod +x ./sa_jag.py

# Get help
./sa_jag.py -h

# Perform sensitivity analysis with 200 training points, excluding columns 3 and 4
./sa_jag.py -n 200 --exclude 3 4

# Perform sensitivity analysis with 150 training points, excluding columns 1 and 2,
#  and save results to log file
./sa_jag.py -n 150 -e 1 2 --log
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from SALib.analyze import sobol
from SALib.sample import saltelli
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error as mse,
)

from surmod import (
    jag,
    gaussian_process_regression as gp,
    sensitivity_analysis as sa,
)


def parse_arguments():
    """Get command line arguments."""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="A script to perform a sensitivity analysis of the JAG dataset.",
    )

    parser.add_argument(
        "-ns",
        "--num_samples",
        type=int,
        default=10000,
        help="Number of sample points to have for initial and acquisition points.",
    )

    parser.add_argument(
        "-e",
        "--exclude",
        type=int,
        nargs="+",
        help="Zero-based column indices to exclude from fitting the surrogate model. "
             "Valid values for JAG dataset: 0=x1, 1=x2, 2=x3, 3=x4, 4=x5.",
    )

    parser.add_argument(
        "-n",
        "--num_train",
        type=int,
        default=100,
        help="Number of points to have in training data set.",
    )

    parser.add_argument(
        "--log",
        action="store_true",
        help="Save output in file based on objective function and kernel; if "
        "   file already exists, new runs will be appended to end of existing file.",
    )

    args = parser.parse_args()

    return args


def main():
    """
    Trains and evaluates a Gaussian process surrogate model on the JAG dataset,
    performs sensitivity analysis, and generates relevant plots and logs.
    """
    # Parse command line arguments
    args = parse_arguments()
    num_samples = args.num_samples
    alpha = 1e-8
    num_train = args.num_train
    log = args.log
    exclude = args.exclude

    df = jag.load_data(n_samples=num_samples, random=False)
    x_train, x_test, y_train, y_test = jag.split_data(df, n_train=args.num_train)
    dim = x_train.shape[0]

    if exclude is not None:
        x_train = np.copy(np.delete(x_train, exclude, axis=1))
        x_test = np.copy(np.delete(x_test, exclude, axis=1))

    # Train the Gaussian process surrogate model
    gp_model = GaussianProcessRegressor(
        kernel=gp.get_kernel("matern", dim, isotropic=True),
        alpha=alpha,
        n_restarts_optimizer=5,
        random_state=42,
        normalize_y=True,
    )

    gp_model.fit(x_train, y_train)

    # Evaluate GP model at train and test inputs
    pred_train = gp_model.predict(x_train)
    pred_test = gp_model.predict(x_test)
    # If pred_train or pred_test is a tuple, get the first element (usually the
    #  mean prediction)
    if isinstance(pred_train, tuple):
        pred_train = pred_train[0]
    if isinstance(pred_test, tuple):
        pred_test = pred_test[0]

    # Evaluate Mean Absolute Error (MAE) with trained GP model
    train_mae = mean_absolute_error(y_train, pred_train)
    test_mae = mean_absolute_error(y_test, pred_test)

    # Evaluate Mean Square Error (MSE) with trained GP model
    train_mse = mse(y_train, pred_train)
    test_mse = mse(y_test, pred_test)

    # Evaluate Maximum Absolute Error (MSE) with trained GP model
    train_max_abserr, train_max_input = gp.compute_max_error(
        pred_train, y_train, x_train
    )
    test_max_abserr, test_max_input = gp.compute_max_error(pred_test, y_test, x_test)

    variable_names = ["x1", "x2", "x3", "x4", "x5"]
    __, k = x_train.shape
    bounds = []
    for i in range(k):
        bounds += [[np.min(x_train[:, i]), np.max(x_train[:, i])]]

    if exclude is not None:
        variable_names = np.copy(np.delete(variable_names, exclude))

    problem = {"num_vars": k, "names": variable_names, "bounds": bounds}

    param_values = saltelli.sample(problem, 2**13, calc_second_order=False)

    Y = gp_model.predict(param_values)
    Si = sobol.analyze(problem, Y, calc_second_order=False)
    print(Si["ST"] - Si["S1"])

    # Prepare the log message
    num_test = num_samples - num_train

    log_message = (
        f"Number of training points: {num_train}\n"
        f"Number of testing points: {num_test}\n"
        f"Kernel: {gp_model.kernel_}\n"
        f"Train MSE: {train_mse:.3e}\n"
        f"Test MSE: {test_mse:.3e}\n"
        f"Train Max abs err:  {train_max_abserr:.3e} | Location: {train_max_input}\n"
        f"Test Max abs err:   {test_max_abserr:.3e} | Location: {test_max_input}\n"
        f"Train MAE: {train_mae:.3e}\n"
        f"Test MAE:  {test_mae:.3e}\n"
    )

    print(log_message)

    if log:
        gp.log_results(
            log_message, path_to_log=os.path.join("output_log", "Jag_Results.txt")
        )

    sa.plot_test_predictions(x_test, y_test, gp_model, "JAG")
    plt.figure()
    sa.sobol_plot(
        Si["S1"],
        Si["ST"],
        problem["names"],
        Si["S1_conf"],
        Si["ST_conf"],
        "JAG",
    )


if __name__ == "__main__":
    main()
