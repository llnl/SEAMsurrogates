#!/usr/bin/env python3

"""
This script performs a sensitivity analysis on a chosen dataset by training a
Gaussian Process (GP) surrogate model. It allows for flexible kernel selection,
length scale adjustment, and exclusion of specific input variables.
The script evaluates model performance, computes Sobol sensitivity indices,
and saves relevant plots.

Note:
- For JAG data there are 5 input variables: x1, x2, x3, x4, x5
- For borehole data there are 8 input variables: rw, r, Tu, Hu, Tl, Hl, L, Kw

Usage:

# Make script executable
chmod +x ./sa_fromdata.py

# Get help
./sa_fromdata.py -h

# Perform sensitivity analysis with 200 training points, 150 testing points,
# excluding variables x4 and x5
./sa_fromdata.py -tr 200 -te 150 --exclude x4 x5

# Perform sensitivity analysis with 150 training points, 100 testing points,
# excluding variables x2 and x3, and save results to log file
./sa_fromdata.py -tr 150 -e x2 x3 --log
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from SALib.analyze import sobol
from SALib.sample import saltelli
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error as rmse

from surmod import sensitivity_analysis as sa, data_processing

from surmod.gaussian_process import GPSurrogate


def parse_arguments():
    """Get command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="A script to perform a sensitivity analysis of the JAG dataset.",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=list(data_processing.DATASET_CONFIG.keys()),
        default="JAG",
        help="Which dataset to use (default: JAG).",
    )

    parser.add_argument(
        "-nx",
        "--normalize_x",
        action="store_true",
        default=False,
        help="Whether or not to normalize the input values by removing the mean and scaling to unit-variance per dimension.",
    )

    parser.add_argument(
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

    parser.add_argument(
        "-tr",
        "--n_train",
        type=int,
        default=400,
        help="Number of train samples (default: 400).",
    )

    parser.add_argument(
        "-te",
        "--n_test",
        type=int,
        default=100,
        help="Number of test samples (default: 100).",
    )

    parser.add_argument(
        "--log",
        action="store_true",
        help="Append results to results/<dataset>.txt",
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
    Trains and evaluates a GP surrogate model on the chosen dataset,
    performs Sobol sensitivity analysis, and generates plots/logs.
    """
    args = parse_arguments()
    dataset = args.dataset
    normalize_x = args.normalize_x
    n_train = args.n_train
    n_test = args.n_test
    do_log = args.log
    exclude = args.exclude

    # Check data availability
    n_samples = n_test + n_train
    if n_samples > 10000:
        raise ValueError(
            f"Requested samples ({n_samples}) exceed existing dataset(s) size limit (10000)."
        )

    df = data_processing.load_data(dataset=dataset, n_samples=n_samples, random=False)
    x_train, x_test, y_train, y_test = data_processing.split_data(df, n_train=n_train)

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

    x_scaler = None
    if normalize_x:
        x_scaler = StandardScaler()
        x_train = x_scaler.fit_transform(x_train)
        x_test = x_scaler.transform(x_test)

    # Fixed nugget -> noise_bounds
    noise_bounds = None
    if args.fixed_nugget is not None:
        noise_bounds = nugget_to_bounds(float(args.fixed_nugget))

    # Train GPSurrogate
    gp_model = GPSurrogate(
        x_train=x_train,
        y_train=np.asarray(y_train).reshape(-1),
        x_test=x_test,
        y_test=np.asarray(y_test).reshape(-1),
        kernel="matern",
        isotropic=True,
        # you already optionally StandardScaler'ed X above, avoid double scaling
        scale_inputs=False,
        # keep output standardization on (matches your old normalize_y=True intent)
        scale_outputs=True,
        noise_bounds=noise_bounds if noise_bounds is not None else (1e-16, 1e-1),
    )
    gp_model.fit()

    # Predict
    pred_train_mean, _ = gp_model.predict(x_train)
    pred_test_mean, _ = gp_model.predict(x_test)

    y_train_1d = np.asarray(y_train).reshape(-1)
    y_test_1d = np.asarray(y_test).reshape(-1)

    # Metrics
    train_mae = mean_absolute_error(y_train_1d, pred_train_mean)
    test_mae = mean_absolute_error(y_test_1d, pred_test_mean)

    train_rmse = rmse(y_train_1d, pred_train_mean)
    test_rmse = rmse(y_test_1d, pred_test_mean)

    train_max_abserr, train_max_input = GPSurrogate.compute_max_error(
        pred_train_mean, y_train_1d, x_train
    )
    test_max_abserr, test_max_input = GPSurrogate.compute_max_error(
        pred_test_mean, y_test_1d, x_test
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
        f"Kernel: matern\n"
        f"Isotropic: True\n"
        f"Normalize x values: {normalize_x}\n"
        f"Fixed nugget: {args.fixed_nugget}\n"
        f"Noise bounds: {noise_bounds if noise_bounds is not None else (1e-16, 1e-1)}\n"
        f"Train RMSE: {train_rmse:.3e}\n"
        f"Test RMSE: {test_rmse:.3e}\n"
        f"Train Max abs err:  {train_max_abserr:.3e} | Location: {train_max_input}\n"
        f"Test Max abs err:   {test_max_abserr:.3e} | Location: {test_max_input}\n"
        f"Train MAE: {train_mae:.3e}\n"
        f"Test MAE:  {test_mae:.3e}\n"
    )
    print(log_message)

    if do_log:
        results_dir = Path(__file__).parent / "results"
        log_results(log_message, path_to_log=results_dir / f"{dataset}.txt")

    # Parity plot: assumes you updated sa.plot_test_predictions to call gp_model.predict(x) -> (mean,std)
    sa.plot_test_predictions(x_test, y_test_1d, gp_model, dataset)

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
