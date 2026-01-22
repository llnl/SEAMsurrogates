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
./bo_jag.py -in 5 -it 30 -k matern -xi 0 -kappa 0

# Perform BO with 10 initial starting points, 30 iterations, and an RBF kernel
./bo_jag.py -in 10 -it 30 -k rbf -xi 0.01 -kappa 0.01 -s 3
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
        help="Normalize target values by removing the mean and scaling to unit variance before fitting the GP.",
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
    num_samples = args.num_samples
    normalize_y = args.normalize_y
    kernel = args.kernel
    n_init = args.n_init
    n_iter = args.n_iter
    seed = args.seed
    xi = args.xi
    kappa = args.kappa

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

    # Configure acquisition functions with their specific parameters
    acquisition_configs = {
        "EI": {"xi": xi},
        "PI": {"xi": xi},
        "UCB": {"kappa": kappa},
        "random": {},
    }

    # Create optimizers and run Bayesian Optimization for each acquisition function
    max_yield_histories = {}
    for acq_func, params in acquisition_configs.items():
        optimizer = bo.BayesianOptimizer(
            "JAG",
            x,
            y,
            normalize_y,
            kernel,
            isotropic=False,
            acquisition_function=acq_func,
            n_acquire=n_iter,
            seed=seed,
            **params,
        )
        max_yield_histories[acq_func] = optimizer.bayes_opt(df, n_init)[2]

    bo.plot_acquisition_comparison(
        max_yield_histories["EI"],
        max_yield_histories["PI"],
        max_yield_histories["UCB"],
        max_yield_histories["random"],
        kernel,
        n_iter,
        n_init,
        "JAG",
        xi=xi,
        kappa=kappa,
    )


if __name__ == "__main__":
    main()
