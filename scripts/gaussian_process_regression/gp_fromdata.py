#!/usr/bin/env python3
"""
Train a GP surrogate model on a chosen dataset using the BoTorch-based GPSurrogate.

Usage examples:

./gp_fromdata.py --num_train=200 --kernel=rbf --isotropic
./gp_fromdata.py --num_train=200 --kernel=matern
./gp_fromdata.py --num_train=200 --kernel=matern --normalize_y --plot
./gp_fromdata.py --num_train=300 --kernel=matern --log
"""

import argparse
import os
import time
from datetime import datetime

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from surmod import data_processing 

from surmod.gpytorch_gaussian_process import GPSurrogate


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="A script to train GP surrogate models for the JAG dataset (BoTorch GPSurrogate).",
    )

    parser.add_argument(
        "-d",
        "--data",
        type=str,
        choices=["JAG", "borehole"],
        default="JAG",
        help="Which dataset to use (default: JAG).",
    )

    parser.add_argument(
        "-tr",
        "--num_train",
        type=int,
        default=400,
        help="Number of train samples.",
    )

    parser.add_argument(
        "-te",
        "--num_test",
        type=int,
        default=100,
        help="Number of test samples.",
    )

    parser.add_argument(
        "-ny",
        "--normalize_y",
        action="store_true",
        help="Standardize outputs (maps to GPSurrogate.scale_outputs).",
    )

    parser.add_argument(
        "-k",
        "--kernel",
        type=str,
        choices=["matern", "rbf", "periodic"],
        default="matern",
        help="Kernel type for GPSurrogate.",
    )

    parser.add_argument(
        "-i",
        "--isotropic",
        action="store_true",
        help="Use isotropic kernel (shared lengthscale). Default is ARD.",
    )

    parser.add_argument(
        "--scale_inputs",
        dest="scale_inputs",
        action="store_true",
        default=True,
        help="Normalize inputs to unit cube (GPSurrogate.scale_inputs).",
    )

    parser.add_argument(
        "--no-scale_inputs",
        dest="scale_inputs",
        action="store_false",
        help="Disable input normalization.",
    )

    parser.add_argument(
        "--lengthscale_bounds",
        type=float,
        nargs=2,
        default=(1e-2, 100.0),
        metavar=("LOW", "HIGH"),
        help="Bounds for kernel lengthscale constraint.",
    )

    parser.add_argument(
        "--noise_bounds",
        type=float,
        nargs=2,
        default=(1e-16, 1e-1),
        metavar=("LOW", "HIGH"),
        help="Bounds for likelihood noise constraint.",
    )

    parser.add_argument(
        "-l",
        "--log",
        action="store_true",
        help="Append results to output log file.",
    )

    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Create observed vs predicted parity plot with 95 percent intervals.",
    )

    parser.add_argument(
        "--LHD",
        action="store_true",
        help="Use an LHD design (passed into split_data if supported).",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random number generator seed.",
    )

    return parser.parse_args()


def log_results(log_message: str, path_to_log: str) -> None:
    os.makedirs(os.path.dirname(path_to_log), exist_ok=True)
    with open(path_to_log, "a", encoding="utf-8") as f:
        f.write(log_message + "\n")


def main():
    args = parse_arguments()

    data = args.data
    num_train = args.num_train
    num_test = args.num_test
    normalize_y = args.normalize_y
    kernel = args.kernel
    isotropic = args.isotropic
    scale_inputs = args.scale_inputs
    lengthscale_bounds = tuple(args.lengthscale_bounds)
    noise_bounds = tuple(args.noise_bounds)
    do_log = args.log
    do_plot = args.plot
    seed = args.seed
    use_lhd = args.LHD

    # Check data availability
    num_samples = num_test + num_train
    if num_samples > 10000:
        raise ValueError(
            f"Requested samples ({num_samples}) exceed existing dataset(s) size limit (10000)."
        )

    # Load and split data
    df = data_processing.load_data(dataset=data, n_samples=num_samples, random=False)
    x_train, x_test, y_train, y_test = data_processing.split_data(
        df=df, LHD=use_lhd, n_train=num_train, seed=seed
    )

    # Ensure y is 1D float array for metrics
    y_train_1d = np.asarray(y_train).reshape(-1)
    y_test_1d = np.asarray(y_test).reshape(-1)

    # Build and fit BoTorch GP surrogate
    gp_model = GPSurrogate(
        x_train=x_train,
        y_train=y_train_1d,
        x_test=x_test,
        y_test=y_test_1d,
        kernel=kernel,
        isotropic=isotropic,
        scale_inputs=scale_inputs,
        scale_outputs=normalize_y,
        lengthscale_bounds=lengthscale_bounds,
        noise_bounds=noise_bounds,
    )

    start_time = time.perf_counter()
    gp_model.fit()
    elapsed_time = time.perf_counter() - start_time

    # Predict on train/test
    pred_train_mean, _pred_train_std = gp_model.predict(x_train)
    pred_test_mean, pred_test_std = gp_model.predict(x_test)

    # Metrics (match your previous ones, plus coverage from GPSurrogate.evaluate)
    train_mae = float(mean_absolute_error(y_train_1d, pred_train_mean))
    test_mae = float(mean_absolute_error(y_test_1d, pred_test_mean))

    train_mse = float(mean_squared_error(y_train_1d, pred_train_mean))
    test_mse = float(mean_squared_error(y_test_1d, pred_test_mean))

    # Max absolute error locations
    train_max_abserr, train_max_input = gp_model.compute_max_error(
        pred_train_mean, y_train_1d, x_train
    )
    test_max_abserr, test_max_input = gp_model.compute_max_error(
        pred_test_mean, y_test_1d, x_test
    )

    # 95 percent interval coverage on test using your model's std
    lower = pred_test_mean - 1.96 * pred_test_std
    upper = pred_test_mean + 1.96 * pred_test_std
    coverage = float(np.mean((y_test_1d >= lower) & (y_test_1d <= upper)))

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    log_lines = [
        f"Run timestamp (%m%d_%H%M%S): {timestamp}",
        f"Test Function: {data}",
        f"Number of training points: {num_train}",
        f"Number of testing points: {num_test}",
        f"Kernel: {kernel}",
        f"Isotropic kernel: {isotropic}",
        f"Scale inputs: {scale_inputs}",
        f"Standardize outputs (normalize_y): {normalize_y}",
        f"Lengthscale bounds: {lengthscale_bounds}",
        f"Noise bounds: {noise_bounds}",
        f"Train MSE: {train_mse:.5e}",
        f"Test MSE: {test_mse:.5e}",
        f"Test 95% interval coverage: {coverage:.2%}",
        f"Train Max abs err:  {train_max_abserr:.5e} | Location: {train_max_input}",
        f"Test Max abs err:   {test_max_abserr:.5e} | Location: {test_max_input}",
        f"Train Mean abs err: {train_mae:.5e}",
        f"Test Mean abs err:  {test_mae:.5e}",
        f"Elapsed time for training GP: {elapsed_time:.3f} seconds",
    ]
    log_message = "\n".join(log_lines) + "\n"

    print(log_message)

    if do_log:
        log_results(
            log_message,
            path_to_log=os.path.join("output_log", f"{data}_Results.txt"),
        )

    if do_plot:
        # Uses your class method that calls evaluate() internally
        gp_model.plot_test_predictions(objective_data_name=data)


if __name__ == "__main__":
    main()