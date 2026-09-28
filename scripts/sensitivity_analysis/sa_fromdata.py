#!/usr/bin/env python3

"""
Perform sensitivity analysis on datasets from data/ using GP surrogates.

This script trains a GP surrogate on real data and computes Sobol sensitivity
indices to identify important input variables. Supports variable exclusion and
customizable GP configurations.

Usage examples:

./sa_fromdata.py --help
./sa_fromdata.py
./sa_fromdata.py -d JAG -tr 200 -te 150 --exclude x4 x5
./sa_fromdata.py -d JAG -tr 200 -te 100 --kernel periodic
./sa_fromdata.py -d borehole -tr 400 -te 100 -k matern --normalize-y
./sa_fromdata.py -d borehole -tr 400 -te 100 -k matern --normalize-y --exclude r Tu
./sa_fromdata.py -d JAG -tr 200 -te 100 --kernel periodic --no-scale-x
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from SALib.analyze import sobol
from SALib.sample import saltelli
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from surmod import data_processing
from surmod import sensitivity_analysis as sa
from surmod.gaussian_process import GPSurrogate
from surmod.utils import log_results


def parse_arguments():
    """Get command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Perform sensitivity analysis on datasets from data/ using GP surrogates.",
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
        help="Which dataset to use (default: JAG).",
    )

    data_options.add_argument(
        "-tr",
        "--n-train",
        type=int,
        default=400,
        help="Number of training samples.",
    )

    data_options.add_argument(
        "-te",
        "--n-test",
        type=int,
        default=100,
        help="Number of test samples.",
    )

    data_options.add_argument(
        "-e",
        "--exclude",
        type=str,
        nargs="+",
        help=(
            "Variable names to exclude from fitting the surrogate model. "
            "Valid values for JAG dataset: x1, x2, x3, x4, x5. "
            "Valid values for borehole dataset: rw, r, Tu, Hu, Tl, Hl, L, Kw."
        ),
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
        default=False,
        help="Use isotropic kernel (single lengthscale for all inputs).",
    )

    gp_options.add_argument(
        "--scale-x",
        action=argparse.BooleanOptionalAction,
        default=True,
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
        metavar="VALUE",
        help="Set the white-noise variance (nugget) to VALUE instead of learning it.",
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


def main():
    """Run surrogate-based sensitivity analysis on a dataset."""
    args = parse_arguments()
    dataset = args.dataset
    scale_x = args.scale_x
    normalize_y = args.normalize_y
    n_train = args.n_train
    n_test = args.n_test
    exclude = args.exclude
    fixed_nugget = args.fixed_nugget
    lengthscale_bounds = tuple(args.lengthscale_bounds)
    noise_bounds = tuple(args.noise_bounds)
    seed = args.seed

    # Check data availability
    n_samples = n_test + n_train
    if n_samples > 10000:
        raise ValueError(
            f"Requested samples ({n_samples}) exceed existing dataset(s) size limit (10000)."
        )

    df = data_processing.load_data(dataset=dataset, n_samples=n_samples, random=False)
    x_train, x_test, y_train, y_test = data_processing.split_data(
        df, n_train=n_train, seed=seed
    )

    # Get variable names from dataset config (all columns except the last one which is 'y')
    variable_names = data_processing.DATASET_CONFIG[dataset]["columns"][:-1]

    # Apply exclusions consistently
    if exclude is not None:
        # Convert variable names to indices
        exclude_indices = []
        for var_name in exclude:
            if var_name not in variable_names:
                raise ValueError(
                    f"Variable '{var_name}' not found in dataset '{dataset}'. "
                    f"Valid variables: {variable_names}"
                )
            exclude_indices.append(variable_names.index(var_name))

        x_train = np.delete(x_train, exclude_indices, axis=1)
        x_test = np.delete(x_test, exclude_indices, axis=1)
        variable_names = [name for name in variable_names if name not in exclude]

    _, dim = x_train.shape

    # Handle fixed nugget
    fixed_noise = fixed_nugget
    noise_bounds_to_use = None if fixed_noise is not None else noise_bounds

    # Train GPSurrogate
    gp_model = GPSurrogate(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        kernel=args.kernel,
        isotropic=args.isotropic,
        scale_inputs=scale_x,
        scale_outputs=normalize_y,
        fixed_noise=fixed_noise,
        lengthscale_bounds=lengthscale_bounds,
        noise_bounds=noise_bounds_to_use,
        seed=seed,
    )
    gp_model.fit()

    # Predict
    pred_train_mean, _ = gp_model.predict(x_train)
    pred_test_mean, _ = gp_model.predict(x_test)

    # Metrics
    train_mae = mean_absolute_error(y_train, pred_train_mean)
    test_mae = mean_absolute_error(y_test, pred_test_mean)

    train_rmse = root_mean_squared_error(y_train, pred_train_mean)
    test_rmse = root_mean_squared_error(y_test, pred_test_mean)

    train_max_abserr, train_max_input = GPSurrogate.compute_max_error(
        pred_train_mean, y_train, x_train
    )
    test_max_abserr, test_max_input = GPSurrogate.compute_max_error(
        pred_test_mean, y_test, x_test
    )

    # Bounds for SALib (use observed range of the (possibly scaled) x_train)
    bounds = [
        [float(np.min(x_train[:, i])), float(np.max(x_train[:, i]))] for i in range(dim)
    ]

    problem = {
        "num_vars": dim,
        "names": list(variable_names),  # type: ignore
        "bounds": bounds,
    }

    param_values = saltelli.sample(problem, 2**13, calc_second_order=False)

    # Predict on SALib samples
    Y_mean, _Y_std = gp_model.predict(param_values)
    Y = np.asarray(Y_mean).reshape(-1)

    Si = sobol.analyze(problem, Y, calc_second_order=False)
    print(Si["ST"] - Si["S1"])

    # Log message
    log_message = (
        f"Number of training points: {n_train}\n"
        f"Number of testing points: {n_test}\n"
        f"Kernel: {args.kernel}\n"
        f"Isotropic: {args.isotropic}\n"
        f"Scale x: {scale_x}\n"
        f"Normalize y: {normalize_y}\n"
        f"Fixed nugget: {fixed_nugget}\n"
        f"Lengthscale bounds: {lengthscale_bounds}\n"
        f"Noise bounds: {noise_bounds_to_use if fixed_noise is None else 'N/A (fixed)'}\n"
        f"Train RMSE: {train_rmse:.3e}\n"
        f"Test RMSE: {test_rmse:.3e}\n"
        f"Train Max abs err: {train_max_abserr:.3e} | Location: {train_max_input}\n"
        f"Test Max abs err: {test_max_abserr:.3e} | Location: {test_max_input}\n"
        f"Train MAE: {train_mae:.3e}\n"
        f"Test MAE: {test_mae:.3e}\n"
    )
    print(log_message)

    results_dir = Path(__file__).parent / "results"
    log_results(log_message, path_to_log=results_dir / f"{dataset}.txt")

    # Parity plot: assumes you updated sa.plot_test_predictions to call gp_model.predict(x) -> (mean,std)
    sa.plot_test_predictions(x_test, y_test, gp_model, dataset)

    plt.figure()
    sa.sobol_plot(
        Si["S1"],
        Si["ST"],
        problem["names"],
        Si["S1_conf"],
        Si["ST_conf"],
        dataset,
    )


if __name__ == "__main__":
    main()
