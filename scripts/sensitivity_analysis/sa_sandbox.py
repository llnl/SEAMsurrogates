#!/usr/bin/env python3

"""
Perform sensitivity analysis on synthetic test functions using GP surrogates.

This script trains a GP surrogate on test function data and computes Sobol
sensitivity indices to identify important input variables. Supports variable
exclusion and customizable test functions.

Usage examples:

./sa_sandbox.py --help
./sa_sandbox.py
./sa_sandbox.py --test-function otlcircuit --n-train 200
./sa_sandbox.py --test-function otlcircuit --n-train 200 --exclude Beta
./sa_sandbox.py -f parabola --b1 2 --b2 1 --b12 0.5
./sa_sandbox.py -f wingweight -tr 150 -e S_w A
./sa_sandbox.py -f otlcircuit -tr 200 -e R_b1 R_f
./sa_sandbox.py -f otlcircuit -tr 200 -e R_b1 R_f --no-scale-x
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from SALib.analyze import sobol
from SALib.sample import saltelli
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from surmod import sensitivity_analysis as sa
from surmod.gaussian_process import GPSurrogate
from surmod.test_functions import get_input_spec, get_variable_names
from surmod.utils import log_results


def parse_arguments():
    """Get command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Perform sensitivity analysis on synthetic test functions using GP surrogates.",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    experiment = parser.add_argument_group("experiment options")
    gp_options = parser.add_argument_group("GP options")
    parabola = parser.add_argument_group("parabola options")

    experiment.add_argument(
        "-f",
        "--test-function",
        type=str,
        choices=["parabola", "otlcircuit", "piston", "wingweight", "borehole"],
        default="parabola",
        help="Choose test function.",
    )

    experiment.add_argument(
        "-tr",
        "--n-train",
        type=int,
        default=100,
        help="Number of training samples.",
    )

    experiment.add_argument(
        "-te",
        "--n-test",
        type=int,
        default=100,
        help="Number of points to have in testing data set.",
    )

    experiment.add_argument(
        "-e",
        "--exclude",
        type=str,
        nargs="+",
        help="Variable names to exclude from fitting the surrogate model",
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

    parabola.add_argument(
        "--b1",
        type=float,
        default=1,
        help="Parabola coefficient for the x1^2 term.",
    )
    parabola.add_argument(
        "--b2",
        type=float,
        default=1,
        help="Parabola coefficient for the x2^2 term.",
    )
    parabola.add_argument(
        "--b12",
        type=float,
        default=1,
        help="Parabola coefficient for the interaction term.",
    )

    return parser.parse_args()


def main():
    """
    Run a full workflow for surrogate-based sensitivity analysis using
    GPSurrogate. Simulate data from test function, train GP model, predict
    model on hold-out data, and plot or log results.
    """
    args = parse_arguments()
    test_function = args.test_function
    n_train = args.n_train
    n_test = args.n_test
    b1 = args.b1
    b2 = args.b2
    b12 = args.b12
    exclude = args.exclude
    kernel = args.kernel
    isotropic = args.isotropic
    lengthscale_bounds = tuple(args.lengthscale_bounds)
    noise_bounds_arg = tuple(args.noise_bounds)
    seed = args.seed

    # Set output directories relative to this script
    plots_dir = Path(__file__).parent / "plots"
    results_dir = Path(__file__).parent / "results"

    _, _, bounds_list = get_input_spec(test_function)
    bounds = np.array(bounds_list, dtype=float)

    x_train, x_test, y_train, y_test = sa.simulate_data(
        test_function, n_train, n_test, b1, b2, b12, seed=seed
    )

    # Get variable names from test_functions module
    variable_names = get_variable_names(test_function)

    # Apply exclusions by converting variable names to indices
    if exclude is not None:
        exclude_indices = []
        for var_name in exclude:
            if var_name not in variable_names:
                raise ValueError(
                    f"Variable '{var_name}' not found in {test_function}. "
                    f"Valid variables: {variable_names}"
                )
            exclude_indices.append(variable_names.index(var_name))

        x_train = np.copy(np.delete(x_train, exclude_indices, axis=1))
        x_test = np.copy(np.delete(x_test, exclude_indices, axis=1))
        bounds = np.delete(bounds, exclude_indices, axis=0)
        variable_names = [name for name in variable_names if name not in exclude]

    dim = x_train.shape[1]

    # Handle fixed nugget
    fixed_noise = args.fixed_nugget
    noise_bounds_to_use = None if fixed_noise is not None else noise_bounds_arg

    gp_model = GPSurrogate(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        kernel=kernel,
        isotropic=isotropic,
        scale_inputs=args.scale_x,
        scale_outputs=args.normalize_y,
        fixed_noise=fixed_noise,
        lengthscale_bounds=lengthscale_bounds,
        noise_bounds=noise_bounds_to_use,
        seed=seed,
    )

    start_time = time.perf_counter()
    gp_model.fit()
    elapsed_time = time.perf_counter() - start_time

    pred_train, _ = gp_model.predict(x_train)
    pred_test, _ = gp_model.predict(x_test)

    train_mae = mean_absolute_error(y_train, pred_train)
    test_mae = mean_absolute_error(y_test, pred_test)

    train_rmse = root_mean_squared_error(y_train, pred_train)
    test_rmse = root_mean_squared_error(y_test, pred_test)

    train_max_abserr, train_max_input = GPSurrogate.compute_max_error(
        pred_train, y_train, x_train
    )
    test_max_abserr, test_max_input = GPSurrogate.compute_max_error(
        pred_test, y_test, x_test
    )

    problem = {
        "num_vars": dim,
        "names": variable_names,
        "bounds": bounds.tolist(),
    }

    param_values = saltelli.sample(problem, 2**13, calc_second_order=False)

    Y_mean, _ = gp_model.predict(param_values)
    Y = np.asarray(Y_mean).reshape(-1)

    Si = sobol.analyze(problem, Y, calc_second_order=False)

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    log_message = (
        f"Run timestamp (%m%d_%H%M%S): {timestamp}\n"
        f"Test Function: {test_function}\n"
        f"Number of training points: {n_train}\n"
        f"Number of testing points: {n_test}\n"
        f"Kernel: {kernel}\n"
        f"Isotropic: {isotropic}\n"
        f"Fixed nugget: {args.fixed_nugget}\n"
        f"Lengthscale bounds: {lengthscale_bounds}\n"
        f"Noise bounds: {noise_bounds_to_use if fixed_noise is None else 'N/A (fixed)'}\n"
        f"Train RMSE: {train_rmse:.3e}\n"
        f"Test RMSE: {test_rmse:.3e}\n"
        f"Train Max abs err: {train_max_abserr:.3e} | Location: {train_max_input}\n"
        f"Test Max abs err: {test_max_abserr:.3e} | Location: {test_max_input}\n"
        f"Train MAE: {train_mae:.3e}\n"
        f"Test MAE: {test_mae:.3e}\n"
        f"Elapsed time for training GP: {elapsed_time:.3f} seconds\n"
    )

    print(log_message)

    log_results(
        log_message,
        path_to_log=results_dir / f"{test_function}.txt",
    )

    # Assumes sa.plot_test_predictions was updated earlier to use gp_model.predict(x)->(mean,std)
    sa.plot_test_predictions(x_test, y_test, gp_model, test_function)

    sa.sobol_plot(
        Si["S1"],
        Si["ST"],
        problem["names"],
        Si["S1_conf"],
        Si["ST_conf"],
        test_function,
    )

    if test_function == "parabola":
        input1 = np.linspace(bounds[0, 0], bounds[0, 1], 100)
        input2 = np.linspace(bounds[1, 0], bounds[1, 1], 100)
        grid_input1, grid_input2 = np.meshgrid(input1, input2)
        x_grid = np.column_stack((grid_input1.flatten(), grid_input2.flatten()))

        preds_mean, _ = gp_model.predict(x_grid)

        plt.figure()
        plt.tricontourf(
            x_grid[:, 0], x_grid[:, 1], preds_mean, levels=50, cmap="viridis"
        )
        plt.title("GP Model Prediction for Parabola")

        plots_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        plt.savefig(plots_dir / f"{b1}_{b2}_{b12}_{test_function}_{timestamp}.png")


if __name__ == "__main__":
    main()
