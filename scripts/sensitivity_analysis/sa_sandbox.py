#!/usr/bin/env python3

"""
This script simulates data from a test function, fits a Gaussian process,
and runs a sensitivity analysis with the fitted GP model.

Note: Column exclusion uses zero-based indexing.

Usage:

# Make script executable
chmod +x ./sa_sandbox.py

# Get help
./sa_sandbox.py -h

# Perform sensitivity analysis on otlcircuit function with 200 training points
./sa_sandbox.py -f otlcircuit -tr 200

# Perform sensitivity analysis on wingweight function with 150 training points,
# excluding columns 2 and 3 (zero-based indexing), and save results to log file
./sa_sandbox.py -f wingweight -tr 150 -e 2 3 -l
"""

import argparse
import os
import time
import datetime

import matplotlib.pyplot as plt
import numpy as np
from SALib.analyze import sobol
from SALib.sample import saltelli
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from surmod import (
    sensitivity_analysis as sa,
    gaussian_process_regression as gp,
)


def parse_arguments():
    """Get command line arguments."""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Perform a sensitivity analysis with a GP surrogate model.",
    )

    parser.add_argument(
        "--b1",
        type=float,
        default=1,
        help="parabola beta_1 parameter",
    )

    parser.add_argument(
        "--b2",
        type=float,
        default=1,
        help="parabola beta_2 parameter",
    )

    parser.add_argument(
        "--b12",
        type=float,
        default=1,
        help="parabola beta_12 parameter",
    )

    parser.add_argument(
        "-e",
        "--exclude",
        type=int,
        nargs="+",
        help="Columns to exclude from fitting the surrogate model",
    )

    parser.add_argument(
        "-f",
        "--objective_function",
        type=str,
        choices=["parabola", "otlcircuit", "piston", "wingweight", "borehole"],
        default="parabola",
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
        "-l",
        "--log",
        action="store_true",
        help="Save output in file based on objective function and kernel; if file"
        " already exists, new runs will be appended to end of existing file.",
    )

    parser.add_argument(
        "-i",
        "--isotropic",
        action="store_true",
        help="Specify that the kernel function is isotropic (same length scale"
        " for all inputs).",
    )

    args = parser.parse_args()

    return args


def main():
    """
    Run a full workflow for surrogate-based sensitivity analysis using
    Gaussian Processes. Simulate data from test function, train GP model, predict
    model on hold-out data, and plot or log results.
    """

    # Parse command-line arguments
    args = parse_arguments()
    objective_function = args.objective_function
    num_train = args.num_train
    num_test = args.num_test
    log = args.log
    b1 = args.b1
    b2 = args.b2
    b12 = args.b12
    exclude = args.exclude
    isotropic = args.isotropic

    regular_dim, __ = sa.load_test_settings(objective_function)

    # Generate test and train data sets
    x_train, x_test, y_train, y_test = sa.simulate_data(
        objective_function, num_train, num_test, b1, b2, b12
    )
    if exclude is not None:
        x_train = np.copy(np.delete(x_train, exclude, axis=1))
        x_test = np.copy(np.delete(x_test, exclude, axis=1))

    dim = x_train.shape[1]

    gp_model = GaussianProcessRegressor(
        kernel=gp.get_kernel(kernel="matern", dim=dim, isotropic=isotropic),
        n_restarts_optimizer=5,
        random_state=42,
        normalize_y=True,
    )

    # Train GP model
    start_time = time.perf_counter()
    gp_model.fit(x_train, y_train)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # Evaluate GP model at train and test inputs
    pred_train = gp_model.predict(x_train)
    pred_test = gp_model.predict(x_test)
    # If pred_train or pred_test is a tuple, get the first element (usually the mean prediction)
    if isinstance(pred_train, tuple):
        pred_train = pred_train[0]
    if isinstance(pred_test, tuple):
        pred_test = pred_test[0]

    # Evaluate Mean Absolute Error (MAE) with trained GP model
    train_mae = mean_absolute_error(y_train, pred_train)
    test_mae = mean_absolute_error(y_test, pred_test)

    # Evaluate Mean Square Error (MSE) with trained GP model
    train_mse = mean_squared_error(y_train, pred_train)
    test_mse = mean_squared_error(y_test, pred_test)

    # Evaluate Maximum Absolute Error (MeAE) with trained GP model
    train_max_abserr, train_max_input = gp.compute_max_error(
        pred_train, y_train, x_train
    )
    test_max_abserr, test_max_input = gp.compute_max_error(pred_test, y_test, x_test)

    if objective_function == "wingweight":
        variable_names = [
            "S_w",
            "W_fw",
            "A",
            "Lambda",
            "q",
            "lambda",
            "t_c",
            "N_z",
            "W_dg",
            "W_p",
        ]
    elif objective_function == "borehole":
        variable_names = ["rw", "r", "Tu", "Hu", "Tl", "Hl", "L", "Kw"]
    elif objective_function == "otlcircuit":
        variable_names = ["R_b1", "R_b2", "R_f", "R_c1", "R_c2", "Beta"]
    elif objective_function == "piston":
        variable_names = ["M", "S", "V_0", "k", "P_0", "T_a", "T_0"]
    else:
        variable_names = [f"x{i}" for i in range(1, regular_dim + 1)]

    if exclude is not None:
        variable_names = np.copy(np.delete(variable_names, exclude))

    bounds = [[0, 1]] * dim

    problem = {"num_vars": dim, "names": variable_names, "bounds": bounds}

    param_values = saltelli.sample(problem, 2**13, calc_second_order=False)

    Y = gp_model.predict(param_values)
    Si = sobol.analyze(problem, Y, calc_second_order=False)

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    # Prepare the log message
    log_message = (
        f"Run timestamp (%m%d_%H%M%S): {timestamp}\n"
        f"Test Function: {objective_function}\n"
        f"Number of training points: {num_train}\n"
        f"Number of testing points: {num_test}\n"
        f"Kernel: {gp_model.kernel_}\n"
        f"Train MSE: {train_mse:.3e}\n"
        f"Test MSE: {test_mse:.3e}\n"
        f"Train Max abs err:  {train_max_abserr:.3e} | Location: {train_max_input}\n"
        f"Test Max abs err:   {test_max_abserr:.3e} | Location: {test_max_input}\n"
        f"Train MAE: {train_mae:.3e}\n"
        f"Test MAE:  {test_mae:.3e}\n"
        f"Elapsed time for training GP: {elapsed_time:.3f} seconds\n"
    )

    print(log_message)

    if log:
        gp.log_results(
            log_message,
            path_to_log=os.path.join("output_log", f"{objective_function}.txt"),
        )

    sa.plot_test_predictions(x_test, y_test, gp_model, objective_function)

    sa.sobol_plot(
        Si["S1"],
        Si["ST"],
        problem["names"],
        Si["S1_conf"],
        Si["ST_conf"],
        objective_function,
    )

    if objective_function == "parabola":
        input1 = np.linspace(0, 1, 100)
        input2 = np.linspace(0, 1, 100)
        grid_input1, grid_input2 = np.meshgrid(input1, input2)
        x_test = np.column_stack((grid_input1.flatten(), grid_input2.flatten()))
        predictions = gp_model.predict(x_test)

        plt.figure()
        plt.tricontourf(
            x_test[:, 0], x_test[:, 1], predictions, levels=50, cmap="viridis"
        )
        plt.title("GP Model Prediction for Parabola")
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
        plt.savefig(
            os.path.join(
                "plots", f"{b1}_{b2}_{b12}_{objective_function}_{timestamp}.png"
            )
        )


if __name__ == "__main__":
    main()
