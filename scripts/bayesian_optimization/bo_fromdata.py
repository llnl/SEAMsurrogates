#!/usr/bin/env python3

"""
This script demonstrates Bayesian Optimization on a chosen dataset and compares
performance across acquisition functions:
- Expected Improvement (EI)
- Probability of Improvement (PI)
- Upper Confidence Bound (UCB)
- Predictive Variance (PV)
- Random

Usage examples:

./bo_fromdata.py --dataset=JAG --num_iter=15 --num_init=10
./bo_fromdata.py --dataset=borehole --num_iter=20 --kernel=rbf --seed=123
./bo_fromdata.py --dataset=JAG --kernel=matern --beta=2.0 --init_design=lhd
./bo_fromdata.py --dataset=borehole --init_design=maximin_lhd --fixed_nugget=1e-7
"""

import argparse

from surmod import bayesian_optimization as bo, data_processing


def nugget_to_bounds(nugget: float) -> tuple[float, float]:
    if nugget <= 0.0:
        raise ValueError("--fixed_nugget must be > 0.")
    delta = 1e-16
    low = max(nugget - delta, 1e-20)
    high = nugget + delta
    return (low, high)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Perform Bayesian optimization on dataset data.",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=["JAG", "borehole"],
        default="JAG",
        help="Which dataset to use.",
    )

    parser.add_argument(
        "-it",
        "--num_iter",
        type=int,
        default=10,
        help="Number of BO iterations.",
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
        choices=["matern", "rbf", "periodic"],
        default="matern",
        help="Choose kernel.",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "-beta",
        "--beta",
        type=float,
        default=2.0,
        help="Exploration parameter for UCB.",
    )

    parser.add_argument(
        "--init_design",
        type=str,
        choices=["random", "lhd", "maximin_lhd"],
        default="random",
        help="Initial design strategy. For dataset BO, LHD-based designs are matched to nearest dataset rows.",
    )

    parser.add_argument(
        "--fixed_nugget",
        type=float,
        default=None,
        help="Fix GP likelihood noise tightly around this nugget value.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    dataset = args.dataset
    kernel = args.kernel
    num_init = args.num_init
    num_iter = args.num_iter
    seed = args.seed

    num_samples = num_init + num_iter
    if num_samples > 10000:
        raise ValueError(
            f"Total samples ({num_samples}) exceed existing dataset size limit (10000)."
        )

    df = data_processing.load_data(dataset=dataset, n_samples=10000, random=False)

    if num_init > len(df):
        raise ValueError(
            f"num_init ({num_init}) cannot exceed dataset size ({len(df)})."
        )

    if num_init + num_iter > len(df):
        raise ValueError(
            f"num_init + num_iter ({num_init + num_iter}) exceeds dataset size ({len(df)})."
        )

    data = df.to_numpy()
    x = data[:, :-1]
    y = data[:, -1]

    # Keep maximin-LHD settings internal, not exposed on CLI
    if args.init_design == "maximin_lhd":
        init_design_kwargs = dict(
            T0=10.0,
            c=0.95,
            it=2000,
            p=50,
            profile="GEOM",
            Imax=100,
            jitter=False,
        )
    else:
        init_design_kwargs = {}

    # If user does not specify, keep nugget small by default
    default_noise_bounds = (1e-8, 1e-6)
    noise_bounds = default_noise_bounds
    fixed_noise = None

    if args.fixed_nugget is not None:
        fixed_noise = float(args.fixed_nugget)
        low, high = default_noise_bounds
        if not (low <= fixed_noise <= high):
            margin = max(abs(fixed_noise) * 0.1, 1e-12)
            noise_bounds = (
                min(low, fixed_noise - margin),
                max(high, fixed_noise + margin),
            )

    bayes_opt_EI = bo.BayesianOptimizer(
        data,
        x,
        y,
        kernel,
        isotropic=False,
        acquisition_function="EI",
        n_acquire=num_iter,
        seed=seed,
        noise_bounds=noise_bounds,
        fixed_noise=fixed_noise,
        init_design=args.init_design,
        init_design_kwargs=init_design_kwargs,
    )

    bayes_opt_PI = bo.BayesianOptimizer(
        data,
        x,
        y,
        kernel,
        isotropic=False,
        acquisition_function="PI",
        n_acquire=num_iter,
        seed=seed,
        noise_bounds=noise_bounds,
        fixed_noise=fixed_noise,
        init_design=args.init_design,
        init_design_kwargs=init_design_kwargs,
    )

    bayes_opt_UCB = bo.BayesianOptimizer(
        data,
        x,
        y,
        kernel,
        isotropic=False,
        acquisition_function="UCB",
        n_acquire=num_iter,
        seed=seed,
        noise_bounds=noise_bounds,
        fixed_noise=fixed_noise,
        init_design=args.init_design,
        init_design_kwargs=init_design_kwargs,
        beta=args.beta,
    )

    bayes_opt_PV = bo.BayesianOptimizer(
        data,
        x,
        y,
        kernel,
        isotropic=False,
        acquisition_function="PV",
        n_acquire=num_iter,
        seed=seed,
        noise_bounds=noise_bounds,
        fixed_noise=fixed_noise,
        init_design=args.init_design,
        init_design_kwargs=init_design_kwargs,
    )

    bayes_opt_rand = bo.BayesianOptimizer(
        data,
        x,
        y,
        kernel,
        isotropic=False,
        acquisition_function="random",
        n_acquire=num_iter,
        seed=seed,
        noise_bounds=noise_bounds,
        fixed_noise=fixed_noise,
        init_design=args.init_design,
        init_design_kwargs=init_design_kwargs,
    )

    max_y_history_EI = bayes_opt_EI.bayes_opt(df, num_init)[2]
    max_y_history_PI = bayes_opt_PI.bayes_opt(df, num_init)[2]
    max_y_history_UCB = bayes_opt_UCB.bayes_opt(df, num_init)[2]
    max_y_history_PV = bayes_opt_PV.bayes_opt(df, num_init)[2]
    max_y_history_random = bayes_opt_rand.bayes_opt(df, num_init)[2]

    bo.plot_acquisition_comparison(
        max_y_history_EI,
        max_y_history_PI,
        max_y_history_UCB,
        max_y_history_PV,
        max_y_history_random,
        kernel,
        num_iter,
        num_init,
        f"{dataset}_{args.init_design}",
        beta=args.beta,
    )


if __name__ == "__main__":
    main()
