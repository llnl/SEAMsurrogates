#!/usr/bin/env python3
"""
Train GP surrogate models on datasets from data/.

This script trains Gaussian Process surrogates on real datasets, with options
for kernel selection, data scaling, and hyperparameter tuning. Results are saved
as plots and logs.

Usage examples:

./gp_fromdata.py --help
./gp_fromdata.py
./gp_fromdata.py -d JAG --n-train 200 --kernel rbf --isotropic
./gp_fromdata.py -d JAG --n-train 300 --kernel matern
./gp_fromdata.py -d borehole -tr 400 -te 100 -k matern --normalize-y
./gp_fromdata.py -d borehole --n-train 200 --kernel matern
./gp_fromdata.py -d hst_H --n-train 200 --kernel matern --normalize-y
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from surmod import data_processing
from surmod.gaussian_process import GPSurrogate


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train GP surrogate models on datasets from data/.",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    data_options = parser.add_argument_group("data options")
    gp_options = parser.add_argument_group("GP model options")

    data_options.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=list(data_processing.DATASET_CONFIG.keys()),
        default="JAG",
        help="Which dataset to use.",
    )

    data_options.add_argument(
        "-tr",
        "--n-train",
        type=int,
        default=50,
        help="Number of training samples.",
    )

    data_options.add_argument(
        "-te",
        "--n-test",
        type=int,
        default=500,
        help="Number of test samples.",
    )

    data_options.add_argument(
        "--LHD",
        action="store_true",
        help="Use an LHD design (passed into split_data if supported).",
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
        f.write(log_message)


def main():
    """
    Trains and evaluates a Gaussian Process (GP) surrogate model on a dataset
    contained in a csv file.
    """
    # Parse command line arguments
    args = parse_arguments()
    dataset = args.dataset
    n_train = args.n_train
    n_test = args.n_test
    normalize_y = args.normalize_y
    kernel = args.kernel
    isotropic = args.isotropic
    scale_x = args.scale_x
    fixed_nugget = args.fixed_nugget
    lengthscale_bounds = tuple(args.lengthscale_bounds)
    noise_bounds = tuple(args.noise_bounds)
    seed = args.seed
    use_lhd = args.LHD

    # Set output directories relative to this script
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"
    plots_dir = script_dir / "plots"

    # Check data availability
    n_samples = n_test + n_train
    if n_samples > 10000:
        raise ValueError(
            f"Requested samples ({n_samples}) exceed existing dataset(s) size limit (10000)."
        )

    # Load and split data
    df = data_processing.load_data(dataset=dataset, n_samples=n_samples, random=False)
    x_train, x_test, y_train, y_test = data_processing.split_data(
        df=df, LHD=use_lhd, n_train=n_train, seed=seed
    )

    # Build and fit BoTorch GP surrogate
    # Handle fixed nugget
    fixed_noise = fixed_nugget
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

    # Predict on train/test
    pred_train_mean, _pred_train_std = gp.predict(x_train)
    pred_test_mean, pred_test_std = gp.predict(x_test)

    # Metrics (match your previous ones, plus coverage from GPSurrogate.evaluate)
    train_mae = mean_absolute_error(y_train, pred_train_mean)
    test_mae = mean_absolute_error(y_test, pred_test_mean)

    train_rmse = root_mean_squared_error(y_train, pred_train_mean)
    test_rmse = root_mean_squared_error(y_test, pred_test_mean)

    # Max absolute error locations
    train_max_abserr, train_max_input = gp.compute_max_error(
        pred_train_mean, y_train, x_train
    )
    test_max_abserr, test_max_input = gp.compute_max_error(
        pred_test_mean, y_test, x_test
    )

    # 95% confidence interval coverage on test data
    lower = pred_test_mean - 1.96 * pred_test_std
    upper = pred_test_mean + 1.96 * pred_test_std
    coverage = np.mean((y_test >= lower) & (y_test <= upper))

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    log_lines = [
        f"Run timestamp (%m%d_%H%M%S): {timestamp}",
        f"Test Function: {dataset}",
        f"Number of training points: {n_train}",
        f"Number of testing points: {n_test}",
        f"Kernel: {kernel}",
        f"Isotropic kernel: {isotropic}",
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
        f"Training time: {elapsed_time:.3f} seconds",
    ]
    log_message = "\n".join(log_lines) + "\n"

    print(log_message)

    log_results(
        log_message,
        path_to_log=results_dir / f"{dataset}.txt",
    )

    gp.plot_test_predictions(dataset=dataset, plots_dir=plots_dir)


if __name__ == "__main__":
    main()
