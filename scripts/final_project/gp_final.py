#!/usr/bin/env python3

"""
This script trains a Gaussian Process (GP) surrogate model on the hubble dataset.
It provides options for different kernel types, isotropic/anisotropic kernels,
normalization, plotting, and logging results.

Usage:

# Make script executable
chmod +x ./gp_jag.py

# See help for all options
./gp_jag.py -h

# Example usages:

# Train a GP with 200 training points, 1000 total samples, and an isotropic RBF
# kernel
./gp_jag.py --num_samples=1000 --num_train=200 --kernel=rbf --isotropic

# Train a GP with 200 training points, 1000 total samples, and an anisotropic
# RBF kernel
./gp_jag.py --num_samples=1000 --num_train=200 --kernel=rbf

# Train a GP with 200 training points, 1500 total samples, Matern kernel,
# normalize y, and plot results
./gp_jag.py --num_samples=1500 --num_train=200 --kernel=matern --normalize_y --plot

# Train a GP with 300 training points, 2000 total samples, Matern kernel,
# and save results to a log file
./gp_jag.py --num_samples=2000 --num_train=300 --kernel=matern --log

"""

import argparse
import os
import time
import datetime

import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from surmod import gaussian_process_regression as gp, leo_drag

def main():
    """
    Trains and evaluates a Gaussian Process (GP) surrogate model on the JAG ICF
    dataset.
    """
    # Parse command line arguments
    num_samples = 10000
    num_train = 100
    normalize_y = True
    choices=["matern", "rbf", "matern_dot"]
    kernel = choices[1]
    isotropic = True
    log = False
    plot = True
    seed = 49

    # Load and split data
    df = leo_drag.load_data(n_samples=num_samples, random=False)
    x_train, x_test, y_train, y_test = leo_drag.split_data(
        df=df, LHD=False, n_train=num_train, seed=seed
    )

    # Instantiate GP model
    gp_model = GaussianProcessRegressor(
        kernel=gp.get_kernel(kernel, x_train.shape[1], isotropic),
        n_restarts_optimizer=5,
        random_state=seed,
        normalize_y=normalize_y,
    )

    # Train GP model
    start_time = time.perf_counter()
    gp_model.fit(x_train, y_train)
    #print(x_train)
    #print(y_train)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # Evaluate GP model at train and test inputs
    pred_train = gp_model.predict(x_train)
    pred_test = gp_model.predict(x_test)
    print("debugging")
    print(gp_model.predict([[6.521328e+03, 4.27113e+02, 1.088285e+03,5.996811e-01,1.68e-01,-2.26E+00,-1.45E+00,50]]))
    print(gp_model.predict([[7.55466e+03,2.103953e+02,1.609433e+03,2.648062e-01,8.17372e-01,-2.44E-01,9.677259e-01,20]]))

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
    num_test = num_samples - num_train

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_lines = [
        f"Run timestamp (%m%d_%H%M%S): {timestamp}",
        "Test Function: LEO Drag",
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
            path_to_log=os.path.join("output_log", "JAG_Results.txt"),
        )

    plt.hist(y_train, bins=30, alpha=0.5, label='Train')
    plt.hist(y_test, bins=30, alpha=0.5, label='Test')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Overlayed Histograms')
    plt.savefig("train_vs_test.png")

    if plot:
        gp.plot_test_predictions(x_test, y_test, gp_model, objective_data_name="LEO Drag")


if __name__ == "__main__":
    main()
