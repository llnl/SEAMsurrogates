#!/usr/bin/env python3

"""
This script demonstrates a Bayesian Optimization (BO) routine on a chosen
dataset and plots performance based on max yield obtained of various acquisition
function choices: Expected Improvement (EI), Probability of Improvement (PI),
Upper Confidence Bound (UCB), random.

The approach is:
1. Obtain an initial set of training data from chosen dataset
2. Train a GP model on the training data
3. Compute the acquisition function at the remaining data points
4. Add the point with the highest value based on the acquisition function of
    choice to the training data
5. Return to step 2 and repeat until the user defined number of acquired points
    is reached

Usage:

# Make script executable
chmod +x ./bo_fromdata.py

# See help.
./bo_fromdata.py -h

# Perform BO with 5 initial starting points, 30 iterations, and a Matern kernel
./bo_fromdata.py -in 5 -it 30 -k matern

# Perform BO with 10 initial starting points, 30 iterations, and an RBF kernel
./bo_fromdata.py -in 10 -it 30 -k rbf
"""

import argparse

from surmod import bayesian_optimization as bo, data_processing


def parse_arguments():
    """Get command line arguments."""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Perform Bayesian optimization on JAG data.",
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
        "-ny",
        "--normalize_y",
        action="store_true",
        help="Whether or not to normalize the output values in the"
        " GaussianProcessRegressor.",
    )

    parser.add_argument(
        "-it",
        "--num_iter",
        type=int,
        default=10,
        help="Number of BO iterations (number of data points to acquire).",
    )

    parser.add_argument(
        "-in",
        "--num_init",
        type=int,
        default=5,
        help="Number of initial sample points.",
    )

    parser.add_argument(
        "-k",
        "--kernel",
        type=str,
        choices=["matern", "rbf", "matern_dot"],
        default="matern",
        help="Choose kernel.",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Set random seed for reproducibility.",
    )

    parser.add_argument(
        "-xi",
        "--xi",
        type=float,
        default=0.0,
        help="Exploration-exploitation trade-off parameter for EI and PI acquisition functions (non-negative float).",
    )

    parser.add_argument(
        "-kappa",
        "--kappa",
        type=float,
        default=2.0,
        help="Exploration-exploitation trade-off parameter for UCB acquisition function (non-negative float).",
    )

    args = parser.parse_args()

    return args


def main():
    # Parse command-line arguments
    args = parse_arguments()
    data = args.data
    normalize_y = args.normalize_y
    kernel = args.kernel
    num_init = args.num_init
    num_iter = args.num_iter
    seed = args.seed

    # Check data availability
    num_samples = num_init + num_iter
    if num_samples > 10000:
        raise ValueError(
            f"Total samples ({num_samples}) exceed existing dataset(s) size "
            "limit (10000)."
        )

    df = data_processing.load_data(dataset=data, n_samples=num_samples, random=False)
    x = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()

    bayes_opt_EI = bo.BayesianOptimizer(
        data,
        x,
        y,
        normalize_y,
        kernel,
        isotropic=False,
        acquisition_function="EI",
        n_acquire=num_iter,
        seed=seed,
    )

    bayes_opt_PI = bo.BayesianOptimizer(
        data,
        x,
        y,
        normalize_y,
        kernel,
        isotropic=False,
        acquisition_function="PI",
        n_acquire=num_iter,
        seed=seed,
    )

    bayes_opt_UCB = bo.BayesianOptimizer(
        data,
        x,
        y,
        normalize_y,
        kernel,
        isotropic=False,
        acquisition_function="UCB",
        n_acquire=num_iter,
        seed=seed,
    )

    bayes_opt_rand = bo.BayesianOptimizer(
        data,
        x,
        y,
        normalize_y,
        kernel,
        isotropic=False,
        acquisition_function="random",
        n_acquire=num_iter,
        seed=seed,
    )

    # Run Bayesian Optimization for different acquisition functions
    max_yield_history_EI = bayes_opt_EI.bayes_opt(df, num_init)[2]
    max_yield_history_PI = bayes_opt_PI.bayes_opt(df, num_init)[2]
    max_yield_history_UCB = bayes_opt_UCB.bayes_opt(df, num_init)[2]
    max_yield_history_random = bayes_opt_rand.bayes_opt(df, num_init)[2]

    bo.plot_acquisition_comparison(
        max_yield_history_EI,
        max_yield_history_PI,
        max_yield_history_UCB,
        max_yield_history_random,
        kernel,
        num_iter,
        num_init,
        data,
    )


if __name__ == "__main__":
    main()
