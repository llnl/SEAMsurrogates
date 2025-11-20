#!/usr/bin/env python3

"""
This script demonstrates a Bayesian Optimization (BO) routine on the JAG
data and plots performance based on max yield obtained of various acquisition
function choices: Expected Improvement (EI), Probability of Improvement (PI),
Upper Confidence Bound (UCB), random.

The approach is:
1. Obtain an initial set of training data from JAG dataset
2. Train a GP model on the training data
3. Compute the acquisition function at the remaining data points
4. Add the point with the highest value based on the acquisition function of
    choice to the training data
5. Return to step 2 and repeat until the user defined number of acquired points
    is reached

Usage:

# Make script executable
chmod +x ./bo_jag.py

# See help.
./bo_jag.py -h

# Perform BO with 5 initial starting points, 30 iterations, and a Matern kernel
./bo_jag.py -in 5 -it 30 -k matern

# Perform BO with 10 initial starting points, 30 iterations, and an RBF kernel
./bo_jag.py -in 10 -it 30 -k rbf
"""

import argparse
import sys

from surmod import bayesian_optimization as bo, jag


def parse_arguments():
    """Get command line arguments."""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Perform Bayesian optimization on JAG data.",
    )

    parser.add_argument(
        "-ns",
        "--num_samples",
        type=int,
        default=10000,
        help="Number of sample points to have for initial and acquisition points.",
    )

    parser.add_argument(
        "-ny",
        "--normalize_y",
        action="store_true",
        help="Scale all columns of the data to [0,1]",
    )

    parser.add_argument(
        "-it",
        "--n_iter",
        type=int,
        default=10,
        help="Number of BO iterations (number of data points to acquire).",
    )

    parser.add_argument(
        "-in",
        "--n_init",
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

    args = parser.parse_args()

    return args


def main():
    # Parse command-line arguments
    args = parse_arguments()
    num_samples = args.num_samples
    normalize_y = args.normalize_y
    kernel = args.kernel
    n_init = args.n_init
    n_iter = args.n_iter
    seed = args.seed

    # Warning and terminate if n_iter + n_init > num_samples
    if n_iter + n_init > num_samples:
        print(
            "Warning: The sum of n_iter and n_init is greater than num_samples."
            " Please check/change your input values."
        )
        sys.exit(1)

    df = jag.load_data(n_samples=num_samples, random=False)
    x = df.iloc[:, :5].to_numpy()
    y = df.iloc[:, -1].to_numpy()

    bayes_opt_EI = bo.BayesianOptimizer(
        "JAG",
        x,
        y,
        normalize_y,
        kernel,
        isotropic=False,
        acquisition_function="EI",
        n_acquire=n_iter,
        seed=seed,
    )

    bayes_opt_PI = bo.BayesianOptimizer(
        "JAG",
        x,
        y,
        normalize_y,
        kernel,
        isotropic=False,
        acquisition_function="PI",
        n_acquire=n_iter,
        seed=seed,
    )

    bayes_opt_UCB = bo.BayesianOptimizer(
        "JAG",
        x,
        y,
        normalize_y,
        kernel,
        isotropic=False,
        acquisition_function="UCB",
        n_acquire=n_iter,
        seed=seed,
    )

    bayes_opt_rand = bo.BayesianOptimizer(
        "JAG",
        x,
        y,
        normalize_y,
        kernel,
        isotropic=False,
        acquisition_function="random",
        n_acquire=n_iter,
        seed=seed,
    )

    # Run Bayesian Optimization for different acquisition functions
    max_yield_history_EI = bayes_opt_EI.bayes_opt(df, n_init)[2]
    max_yield_history_PI = bayes_opt_PI.bayes_opt(df, n_init)[2]
    max_yield_history_UCB = bayes_opt_UCB.bayes_opt(df, n_init)[2]
    max_yield_history_random = bayes_opt_rand.bayes_opt(df, n_init)[2]

    bo.plot_acquisition_comparison(
        max_yield_history_EI,
        max_yield_history_PI,
        max_yield_history_UCB,
        max_yield_history_random,
        kernel,
        n_iter,
        n_init,
        "JAG",
    )


if __name__ == "__main__":
    main()
