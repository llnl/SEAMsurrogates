#!/usr/bin/env python3

"""
This script performs a sensitivity analysis on a chosen dataset by training a
Gaussian Process (GP) surrogate model. It allows for flexible kernel selection,
length scale adjustment, and exclusion of specific input columns.
The script evaluates model performance, computes Sobol sensitivity indices,
and saves relevant plots.

Note:
- For JAG data there are 5 input variables
- For borehole data there are 8 input variables
- Column exclusion uses zero-based indexing

Usage:

# Make script executable
chmod +x ./sa_fromdata.py

# Get help
./sa_fromdata.py -h

# Perform sensitivity analysis with 200 training points, 150 testing points,
# excluding columns 3 and 4
./sa_fromdata.py -tr 200 -te 150 --exclude 3 4

# Perform sensitivity analysis with 150 training points, 100 testing points,
#  excluding columns 1 and 2, and save results to log file
./sa_fromdata.py -tr 150 -e 1 2 --log
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from SALib.analyze import sobol
from SALib.sample import saltelli
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error as mse,
)

from surmod import (
    gaussian_process_regression as gp,
    sensitivity_analysis as sa,
    data_processing
)


def parse_arguments():
    """Get command line arguments."""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="A script to perform a sensitivity analysis of the JAG dataset.",
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
        "-nx",
        "--normalize_x",
        action="store_true",
        default=False,
        help="Whether or not to normalize the input values by removing the "
        "mean and scaling to unit-variance per dimension.",
    )

    parser.add_argument(
        "-e",
        "--exclude",
        type=int,
        nargs="+",
        help="Zero-based column indices to exclude from fitting the surrogate model. "
        "Valid values for JAG dataset: 0=x1, 1=x2, 2=x3, 3=x4, 4=x5." \
        "Valid values for borehole dataset: 0=rw, 1=r, 2=Tu, 3=Hu, 4=Tl, 5=Hl, 6=L, 7=Kw.",
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
    data = args.data
    normalize_x = args.normalize_x
    alpha = 1e-8
    num_train = args.num_train
    num_test = args.num_test
    log = args.log
    exclude = args.exclude

    # Check data availability
    num_samples = num_test + num_train
    if num_samples > 10000:
        raise ValueError(
            f"Requested samples ({num_samples}) exceed existing dataset(s) size "
            "limit (10000)."
        )

    df = data_processing.load_data(dataset=data, n_samples=num_samples, random=False)
    x_train, x_test, y_train, y_test = data_processing.split_data(df, n_train=num_train)

    # Initial variable names per dataset
    if data == "JAG":
        variable_names = np.array(["x1", "x2", "x3", "x4", "x5"])
    elif data == "borehole":
        variable_names = np.array(["rw", "r", "Tu", "Hu", "Tl", "Hl", "L", "Kw"])

    # Apply exclusions consistently to x and variable_names
    if exclude is not None:
        x_train = np.delete(x_train, exclude, axis=1)
        x_test  = np.delete(x_test,  exclude, axis=1)
        variable_names = np.delete(variable_names, exclude) # type: ignore

    # Now get correct dimension AFTER exclusion
    _, dim = x_train.shape

    if normalize_x:
        # Feature scaling: mean 0, std 1 using training data only
        x_scaler = StandardScaler()
        x_train = x_scaler.fit_transform(x_train)
        x_test = x_scaler.transform(x_test)

    # Train the Gaussian process surrogate model with correct dim
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

    # Build bounds from the modified x_train
    bounds = []
    for i in range(dim):
        bounds.append([np.min(x_train[:, i]), np.max(x_train[:, i])])

    # SALib problem definition, all dimensions must match
    problem = {
        "num_vars": dim,
        "names": list(variable_names), # type: ignore
        "bounds": bounds,
    }

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
            log_message, path_to_log=os.path.join("output_log", f"{data}_Results.txt")
        )

    sa.plot_test_predictions(x_test, y_test, gp_model, data)
    plt.figure()
    sa.sobol_plot(
        Si["S1"],
        Si["ST"],
        problem["names"],
        Si["S1_conf"],
        Si["ST_conf"],
        data,
    )


if __name__ == "__main__":
    main()
