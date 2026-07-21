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
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from SALib.analyze import sobol
from SALib.sample import saltelli
from sklearn.metrics import mean_absolute_error, mean_squared_error

from surmod import sensitivity_analysis as sa

from surmod.gaussian_process import GPSurrogate


def parse_arguments():
    """Get command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Perform a sensitivity analysis with a GP surrogate model.",
    )

    parser.add_argument("--b1", type=float, default=1, help="parabola beta_1 parameter")
    parser.add_argument("--b2", type=float, default=1, help="parabola beta_2 parameter")
    parser.add_argument(
        "--b12", type=float, default=1, help="parabola beta_12 parameter"
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
        "--n_train",
        type=int,
        default=100,
        help="Number of points to have in training data set.",
    )

    parser.add_argument(
        "-te",
        "--n_test",
        type=int,
        default=100,
        help="Number of points to have in testing data set.",
    )

    parser.add_argument(
        "-l",
        "--log",
        action="store_true",
        help="Save output in file based on objective function and kernel; if file already exists, append.",
    )

    parser.add_argument(
        "-i",
        "--isotropic",
        action="store_true",
        help="Use isotropic kernel (same lengthscale for all inputs).",
    )

    parser.add_argument(
        "--fixed_nugget",
        type=float,
        default=None,
        help="Fix likelihood noise by setting noise_bounds to nugget +/- nugget/10000.",
    )

    return parser.parse_args()


def log_results(log_message: str, path_to_log: Path | str) -> None:
    path = Path(path_to_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(log_message + "\n")


def nugget_to_bounds(nugget: float) -> tuple[float, float]:
    if nugget <= 0.0:
        raise ValueError("--fixed_nugget must be > 0.")
    delta = nugget / 10000.0
    low = max(nugget - delta, 1e-20)
    high = nugget + delta
    return (low, high)


def main():
    """
    Run a full workflow for surrogate-based sensitivity analysis using
    GPSurrogate. Simulate data from test function, train GP model, predict
    model on hold-out data, and plot or log results.
    """
    args = parse_arguments()
    objective_function = args.objective_function
    n_train = args.n_train
    n_test = args.n_test
    do_log = args.log
    b1 = args.b1
    b2 = args.b2
    b12 = args.b12
    exclude = args.exclude
    isotropic = args.isotropic

    regular_dim, __ = sa.load_test_settings(objective_function)

    x_train, x_test, y_train, y_test = sa.simulate_data(
        objective_function, n_train, n_test, b1, b2, b12
    )

    if exclude is not None:
        x_train = np.copy(np.delete(x_train, exclude, axis=1))
        x_test = np.copy(np.delete(x_test, exclude, axis=1))

    dim = x_train.shape[1]

    noise_bounds = None
    if args.fixed_nugget is not None:
        noise_bounds = nugget_to_bounds(float(args.fixed_nugget))

    gp_model = GPSurrogate(
        x_train=x_train,
        y_train=np.asarray(y_train).reshape(-1),
        x_test=x_test,
        y_test=np.asarray(y_test).reshape(-1),
        kernel="matern",
        isotropic=isotropic,
        scale_inputs=False,  # your SA data are already in [0,1]
        scale_outputs=True,  # matches old normalize_y=True intent
        noise_bounds=noise_bounds if noise_bounds is not None else (1e-16, 1e-1),
    )

    start_time = time.perf_counter()
    gp_model.fit()
    elapsed_time = time.perf_counter() - start_time

    pred_train, _ = gp_model.predict(x_train)
    pred_test, _ = gp_model.predict(x_test)

    y_train_1d = np.asarray(y_train).reshape(-1)
    y_test_1d = np.asarray(y_test).reshape(-1)

    train_mae = mean_absolute_error(y_train_1d, pred_train)
    test_mae = mean_absolute_error(y_test_1d, pred_test)

    train_mse = mean_squared_error(y_train_1d, pred_train)
    test_mse = mean_squared_error(y_test_1d, pred_test)

    train_max_abserr, train_max_input = GPSurrogate.compute_max_error(
        pred_train, y_train_1d, x_train
    )
    test_max_abserr, test_max_input = GPSurrogate.compute_max_error(
        pred_test, y_test_1d, x_test
    )

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
        variable_names = list(
            np.delete(np.array(variable_names, dtype=object), exclude)
        )

    bounds = [[0.0, 1.0]] * dim
    problem = {"n_vars": dim, "names": variable_names, "bounds": bounds}

    param_values = saltelli.sample(problem, 2**13, calc_second_order=False)

    Y_mean, _ = gp_model.predict(param_values)
    Y = np.asarray(Y_mean).reshape(-1)

    Si = sobol.analyze(problem, Y, calc_second_order=False)

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    log_message = (
        f"Run timestamp (%m%d_%H%M%S): {timestamp}\n"
        f"Test Function: {objective_function}\n"
        f"Number of training points: {n_train}\n"
        f"Number of testing points: {n_test}\n"
        f"Kernel: matern\n"
        f"Isotropic: {isotropic}\n"
        f"Fixed nugget: {args.fixed_nugget}\n"
        f"Noise bounds: {noise_bounds if noise_bounds is not None else (1e-16, 1e-1)}\n"
        f"Train MSE: {train_mse:.3e}\n"
        f"Test MSE: {test_mse:.3e}\n"
        f"Train Max abs err:  {train_max_abserr:.3e} | Location: {train_max_input}\n"
        f"Test Max abs err:   {test_max_abserr:.3e} | Location: {test_max_input}\n"
        f"Train MAE: {train_mae:.3e}\n"
        f"Test MAE:  {test_mae:.3e}\n"
        f"Elapsed time for training GP: {elapsed_time:.3f} seconds\n"
    )

    print(log_message)

    if do_log:
        log_results(
            log_message,
            path_to_log=Path("output_log") / f"{objective_function}.txt",
        )

    # Assumes sa.plot_test_predictions was updated earlier to use gp_model.predict(x)->(mean,std)
    sa.plot_test_predictions(x_test, y_test_1d, gp_model, objective_function)

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
        x_grid = np.column_stack((grid_input1.flatten(), grid_input2.flatten()))

        preds_mean, _ = gp_model.predict(x_grid)

        plt.figure()
        plt.tricontourf(
            x_grid[:, 0], x_grid[:, 1], preds_mean, levels=50, cmap="viridis"
        )
        plt.title("GP Model Prediction for Parabola")

        Path("plots").mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        plt.savefig(
            Path("plots") / f"{b1}_{b2}_{b12}_{objective_function}_{timestamp}.png"
        )


if __name__ == "__main__":
    main()
