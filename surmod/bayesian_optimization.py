import os
from typing import Sequence, Union, List, Tuple, Optional
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch

from scipy.optimize import minimize
from scipy.stats import norm

from sklearn.gaussian_process import GaussianProcessRegressor

from surmod import gaussian_process_regression as gp


def sample_parabola(
    n_initial: int,
    bounds_low: Union[float, Sequence[float], np.ndarray],
    bounds_high: Union[float, Sequence[float], np.ndarray],
    input_size: int,
    radius: float = 7,
) -> np.ndarray:
    """
    Generates random sample points outside a specified radius from the origin.

    Points are sampled uniformly within the given bounds, and only those lying
    outside the specified radius from the origin are included.

    Args:
        n_initial (int): Number of sample points to generate.
        bounds_low (float, sequence of float, or np.ndarray): Lower bounds for
            each input dimension.
        bounds_high (float, sequence of float, or np.ndarray): Upper bounds for
            each input dimension.
        input_size (int): Number of dimensions for each sample point.
        radius (float, optional): Exclusion radius around the origin. No points
            will be generated within this radius. Defaults to 7.

    Returns:
        np.ndarray: Array of shape (n_initial, input_size) containing the
        generated sample points.
    """
    samples = []

    while len(samples) < n_initial:
        # Generate a single point
        x_point = np.random.uniform(bounds_low, bounds_high, size=input_size)

        # Calculate the distance from the origin
        distance = np.linalg.norm(x_point)

        # Check if the point is outside the excluded radius
        if distance > radius:
            samples.append(x_point)

    return np.array(samples)


def sample_data(
    objective_function: str,
    bounds_low: Union[float, Sequence[float], np.ndarray],
    bounds_high: Union[float, Sequence[float], np.ndarray],
    n_initial: int,
    input_size: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates sample input and output data using the specified objective function.

    Depending on the objective function, this function generates random sample
    points within the given bounds, evaluates the objective function at those
    points, and returns the resulting input-output pairs.

    Args:
        objective_function (str): Name of the objective function to sample from.
            If "Parabola", uses a custom sampling method.
        bounds_low (float, sequence of float, or np.ndarray): Lower bounds for
            each input dimension.
        bounds_high (float, sequence of float, or np.ndarray): Upper bounds for
            each input dimension.
        n_initial (int): Number of sample points to generate.
        input_size (int, optional): Number of dimensions for each sample point.
            Defaults to 2.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            - x_sample: Array of shape (n_initial, input_size) with input sample
                points.
            - y_sample: Array of shape (n_initial, ...) with corresponding
                function outputs.
    """
    test_function = gp.load_test_function(objective_function)

    if objective_function == "Parabola":
        x_data = sample_parabola(n_initial, bounds_low, bounds_high, input_size)
    else:
        x_data = np.random.uniform(
            bounds_low, bounds_high, size=(n_initial, input_size)
        )

    x_data = torch.Tensor(x_data)
    y_data = test_function(x_data)

    x_sample = x_data.clone().detach().numpy()
    y_sample = y_data.clone().detach().numpy()

    return x_sample, y_sample


def get_synth_global_optima(
    objective_function: str,
) -> Tuple[List[List[float]], float]:
    """
    Return the locations and value of the global optima for a given objective
        function.

    Args:
        objective_function (str): The name of the objective function. Supported
            values are: "Ackley", "Branin", "Griewank", "HolderTable", "Parabola".

    Returns:
        Tuple[List[List[float]], float]: A tuple containing a list of coordinates
            and the global optimum value.

    Raises:
        ValueError: If the provided objective_function name is not recognized.
    """
    global_optima = {
        "Ackley": ([[0, 0]], 0.0),
        "Branin": (
            [[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]],
            -0.397887,
        ),
        "Griewank": ([[0, 0]], 0.0),
        "HolderTable": (
            [
                [8.05502, 9.66459],
                [-8.05502, -9.66459],
                [-8.05502, 9.66459],
                [8.05502, -9.66459],
            ],
            19.2085,
        ),
        "Parabola": ([[0, 0]], 0.0),
    }

    if objective_function not in global_optima:
        raise ValueError(
            f"Objective function '{objective_function}' is not recognized."
        )

    return global_optima[objective_function]


def expected_improvement(
    X: np.ndarray, y_max: float, gp: GaussianProcessRegressor, xi: float = 0.0
) -> np.ndarray:
    """
    Compute the Expected Improvement (EI) acquisition values for a set of input
    points.

    The Expected Improvement is used in Bayesian optimization to balance
    exploration and exploitation when searching for the maximum of an unknown
    function. It quantifies the expected amount by which sampling at a new point
    will improve over the current best observed value.

    Args:
        X (np.ndarray): 2D array of shape (n_points, n_features) representing
            the input points where EI is evaluated.
        y_max (float): The current maximum observed value of the objective
            function.
        gp (GaussianProcessRegressor): A fitted Gaussian process regressor used
            to predict mean and standard deviation.
        xi (float, optional): Exploration-exploitation trade-off hyperparameter.
            Larger values encourage exploration. Default is 0.0 (standard EI).

    Returns:
        np.ndarray: 1D array of expected improvement values at each point in X,
            shape (n_points,).
    """
    mu, sigma = gp.predict(X, return_std=True)  # type: ignore
    with np.errstate(divide="warn"):
        improvement = mu - (y_max + xi)
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        if any(sigma == 0):
            ei[sigma == 0.0] = 0.0
    return ei


def probability_of_improvement(
    x_sample: np.ndarray,
    model: GaussianProcessRegressor,
    y_max: float,
    xi: float = 0.0,
) -> np.ndarray:
    """
    Compute the Probability of Improvement (PI) acquisition function.

    The probability of improvement is used in Bayesian optimization to estimate
    the likelihood that sampling at given points will yield an improvement over
    the current maximum observed value.

    Args:
        x_sample (np.ndarray): Points at which the acquisition function should
            be evaluated, with shape (n_samples, n_features).
        model (GaussianProcessRegressor): A fitted Gaussian process model used
            to predict the mean and standard deviation at the sample points.
        y_max (float): The current maximum known value of the target function.
        xi (float, optional): Exploration-exploitation trade-off hyperparameter.
            Larger values encourage exploration. Default is 0.0 (standard PI).

    Returns:
        np.ndarray: The probability of improvement at each point in `x_sample`
            with shape (n_samples,).
    """
    mu, sigma = model.predict(x_sample, return_std=True)  # type: ignore
    with np.errstate(divide="warn"):
        Z = (mu - (y_max + xi)) / sigma
        pi = norm.cdf(Z)
        pi[sigma == 0.0] = 0.0  # Avoid division by zero
    return pi


def upper_confidence_bound(
    x_sample: np.ndarray, model: GaussianProcessRegressor, kappa: float
) -> np.ndarray:
    """
    Compute the Upper Confidence Bound (UCB) acquisition function.

    The UCB acquisition function is used in Bayesian optimization to balance
    exploration and exploitation by combining the predicted mean and uncertainty
    of a Gaussian process model.

    Args:
        x_sample (np.ndarray): Points at which to evaluate the acquisition
            function, with shape (n_samples, n_features).
        model (GaussianProcessRegressor): A fitted Gaussian process model used
            to predict the mean and standard deviation at the sample points.
        kappa (float): Controls the balance between exploration and exploitation.

    Returns:
        np.ndarray: The UCB value at each point in `x_sample`, with shape
            (n_samples,).
    """
    mu, sigma = model.predict(x_sample, return_std=True)  # type: ignore
    return mu + kappa * sigma


def predictive_variance(
    x_sample: np.ndarray, model: GaussianProcessRegressor
) -> np.ndarray:
    """
    Compute the Predictive Variance acquisition function.

    The Predictive Variance acquisition function focuses purely on exploration
    by selecting points with the highest prediction uncertainty. It helps
    explore regions where the model is most uncertain about the function values.

    Args:
        x_sample (np.ndarray): Points at which to evaluate the acquisition
            function, with shape (n_samples, n_features).
        model (GaussianProcessRegressor): A fitted Gaussian process model used
            to predict the standard deviation at the sample points.

    Returns:
        np.ndarray: The predictive variance at each point in `x_sample`, with shape
            (n_samples,).
    """
    _, sigma = model.predict(x_sample, return_std=True)  # type: ignore
    return sigma**2


ACQUISITION_FUNCTIONS = {
    "EI": expected_improvement,
    "PI": probability_of_improvement,
    "UCB": upper_confidence_bound,
    "PV": predictive_variance,
    "random": None,
}


class BayesianOptimizer:
    """
    A class providing methods for Bayesian Optimization using Gaussian Processes.

    Supports both synthetic (continuous) functions and dataset-based (discrete) optimization.
    Handles initialization, acquisition function selection, GP fitting, and iterative sampling.

    Args:
        objective_function (str): Name of the objective function or dataset.
        x_init (np.ndarray): Initial input samples.
        y_init (np.ndarray): Initial output values.
        kernel (str): Kernel type for the Gaussian Process.
        isotropic (bool): Whether to use an isotropic kernel.
        acquisition_function (str): Acquisition function to use ('EI', 'PI', 'UCB', 'PV', 'random').
        n_acquire (int): Number of optimization steps.
        seed (int, optional): Random seed for reproducibility. Default is 42.
    """

    def __init__(
        self,
        objective_function: str,
        x_init: np.ndarray,
        y_init: np.ndarray,
        normalize_y: bool = False,
        kernel: str = "matern",
        isotropic: bool = False,
        acquisition_function: str = "EI",
        n_acquire: int = 10,
        seed: int = 42,
        **acquisition_kwargs,
    ):
        self.objective_function = objective_function
        self.x_all_data = x_init
        self.y_all_data = y_init
        self.x_init = x_init
        self.y_init = y_init
        self.normalize_y = normalize_y
        self.x_acquired = np.empty((0, self.x_init.shape[1]))
        self.y_acquired = np.empty((0,))
        self.kernel = kernel
        self.isotropic = isotropic
        self.acquisition = acquisition_function
        self.n_acquire = n_acquire
        self.seed = seed
        self.acquisition_kwargs = acquisition_kwargs
        self.gp_model = None
        self.y_max_history = np.empty((0,))

    def evaluate_objective(self, x_next) -> torch.Tensor:
        """
        Evaluates the objective function at the given input.

        Args:
            x_next (np.ndarray or torch.Tensor): The input at which to evaluate
                the objective function.

        Returns:
            torch.Tensor: The output of the synthetic objective function evaluated
            at x_next.
        """
        synthetic_function = gp.load_test_function(self.objective_function)
        # Before calling the synthetic function:
        if isinstance(x_next, np.ndarray):
            x_next = torch.from_numpy(
                x_next.astype(np.float32)
            )  # or np.float64 if needed

        if x_next.ndim == 1:
            x_next = x_next.unsqueeze(0)  # make it 2D if needed

        y_next = synthetic_function(x_next)
        return y_next

    def gp_model_fit(self) -> GaussianProcessRegressor:
        """
        Fits a Gaussian Process (GP) model to the available data.

        Uses the specified kernel, normalization, and random seed to initialize
        the GaussianProcessRegressor. The model is trained on all available input
        and output data.

        Returns:
            GaussianProcessRegressor: The fitted GP model.
        """
        dim = self.x_all_data.shape[1]
        self.gp_model = GaussianProcessRegressor(
            kernel=gp.get_kernel(self.kernel, dim, self.isotropic),
            n_restarts_optimizer=10,
            random_state=self.seed,
            normalize_y=self.normalize_y,
        )
        self.gp_model.fit(self.x_all_data, self.y_all_data)
        return self.gp_model

    def propose_location(
        self, acquisition: str = "EI", n_restarts: int = 100
    ) -> np.ndarray:
        """
        Proposes the next location to evaluate using a specified acquisition function.

        This method selects the next candidate point in the search space for
        evaluation based on the given acquisition function. It supports 'EI'
        (Expected Improvement), 'PI' (Probability of Improvement), 'UCB'
        (Upper Confidence Bound), 'PV' (Predictive Variance), and 'random'. For
        non-random acquisition functions, it performs multiple restarts of optimization
        to find the best candidate.

        Args:
            acquisition (str, optional): The acquisition function to use. Must
                be one of 'EI', 'PI', 'UCB', 'PV', or 'random'. Defaults to 'EI'.
            n_restarts (int, optional): Number of random restarts for the optimizer.
                Defaults to 100.

        Returns:
            np.ndarray: The proposed next location as a 1D array.

        Raises:
            ValueError: If an invalid acquisition function is specified.
        """
        rng = np.random.RandomState(self.seed)
        synthetic_function = gp.load_test_function(self.objective_function)
        epsilon = 1e-4
        bounds_low = [b[0] for b in synthetic_function._bounds]
        bounds_high = [b[1] for b in synthetic_function._bounds]
        input_size = len(bounds_low)
        bounds = [
            (low + epsilon, high - epsilon)
            for low, high in zip(bounds_low, bounds_high)
        ]
        y_max = np.max(self.y_max_history) if len(self.y_max_history) > 0 else 0.0

        if acquisition not in ACQUISITION_FUNCTIONS:
            raise ValueError(
                "Invalid acquisition function. Choose 'EI', 'PI', 'UCB', 'PV', or 'random'."
            )

        acq_func = ACQUISITION_FUNCTIONS[acquisition]

        if acquisition == "random":
            # Just pick a random point in the domain
            return rng.uniform(bounds_low, bounds_high)

        max_val = -np.inf
        max_x = np.asarray([np.inf] * input_size)
        starting_points = rng.uniform(
            bounds_low, bounds_high, size=(n_restarts, input_size)
        )

        for x0 in starting_points:

            def acq_wrap(x):
                if acquisition == "EI":
                    xi = self.acquisition_kwargs.get("xi", 0.0)
                    return -acq_func(
                        x.reshape(1, -1), y_max, self.gp_model, xi=xi
                    ).item()
                elif acquisition == "PI":
                    xi = self.acquisition_kwargs.get("xi", 0.0)
                    return -acq_func(
                        x.reshape(1, -1), self.gp_model, y_max, xi=xi
                    ).item()
                elif acquisition == "UCB":
                    kappa = self.acquisition_kwargs.get("kappa", 2.0)
                    return -acq_func(
                        x.reshape(1, -1), self.gp_model, kappa=kappa
                    ).item()
                elif acquisition == "PV":
                    return -acq_func(x.reshape(1, -1), self.gp_model).item()
                else:
                    raise ValueError("Invalid acquisition function.")

            res = minimize(
                fun=acq_wrap,
                x0=x0,
                bounds=bounds,
                method="L-BFGS-B",
            )
            if -res.fun > max_val:
                max_val = -res.fun
                max_x = res.x

        x_next = max_x
        return x_next

    def bayes_opt(
        self, df: Optional[pd.DataFrame] = None, n_init: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Unified Bayesian Optimization method for both dataset and synthetic function.

        If data is provided (as a DataFrame), runs dataset-based ("discrete") BO.
        Otherwise, runs synthetic function ("continuous") BO using the class's
        objective_function.

        Args:
            df: pd.DataFrame or None
                If provided, DataFrame with columns x0...xn and 'y'
            n_init: int or None
                Number of initial points (dataset mode)
            n_iter: int or None
                Number of BO iterations (overrides self.n_acquire if provided)
            seed: int or None
                Random seed
        Returns:
            x_all_data, y_all_data, y_max_history: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        # Ensure reproducibility with initial points
        rng = np.random.RandomState(self.seed)

        # Dataset mode (acquiring from precollected data, "discrete" optimization
        #   on "limited" data)
        if df is not None:
            df = (df - df.min()) / (df.max() - df.min())
            x = df.iloc[:, :-1].to_numpy(dtype=float)
            y = df.iloc[:, -1].to_numpy(dtype=float)
            n_iter = self.n_acquire

            initial_indices = rng.choice(np.arange(len(df)), size=n_init, replace=False)
            x_init = x[initial_indices]
            y_init = y[initial_indices]

            self.x_all_data = x_init.copy()
            self.y_all_data = y_init.copy()
            self.x_init = x_init.copy()
            self.y_init = y_init.copy()
            self.x_acquired = np.empty((0, x_init.shape[1]))
            self.y_acquired = np.empty((0,))
            self.y_max_history = np.array([np.max(y_init)], dtype=float)

            gp_model = self.gp_model_fit()

            remaining_indices = set(range(len(df))) - set(initial_indices)
            for _ in range(n_iter):
                x_remaining = x[list(remaining_indices)]
                # Compute acquisition values
                if self.acquisition == "EI":
                    xi = self.acquisition_kwargs.get("xi", 0.0)
                    acquisition_values = expected_improvement(
                        x_remaining,
                        np.max(self.y_all_data),
                        gp_model,
                        xi=xi,
                    )
                elif self.acquisition == "PI":
                    xi = self.acquisition_kwargs.get("xi", 0.0)
                    acquisition_values = probability_of_improvement(
                        x_remaining,
                        gp_model,
                        np.max(self.y_all_data),
                        xi=xi,
                    )
                elif self.acquisition == "UCB":
                    kappa = self.acquisition_kwargs.get("kappa", 2.0)
                    acquisition_values = upper_confidence_bound(
                        x_remaining,
                        gp_model,
                        kappa=kappa,
                    )
                elif self.acquisition == "PV":
                    acquisition_values = predictive_variance(
                        x_remaining,
                        gp_model,
                    )
                elif self.acquisition == "random":
                    acquisition_values = rng.uniform(size=x_remaining.shape[0])
                else:
                    raise ValueError("Invalid acquisition function.")

                next_idx_in_remaining = np.argmax(acquisition_values)
                next_index = list(remaining_indices)[next_idx_in_remaining]

                next_point = x[next_index].reshape(1, -1)
                next_value = float(y[next_index])

                self.x_all_data = np.vstack((self.x_all_data, next_point))
                self.y_all_data = np.append(self.y_all_data, next_value)
                self.x_acquired = np.append(self.x_acquired, next_point, axis=0)
                self.y_acquired = np.append(self.y_acquired, next_value)

                gp_model = self.gp_model_fit()
                y_max = np.max(self.y_all_data)
                self.y_max_history = np.append(self.y_max_history, y_max)

                remaining_indices.remove(next_index)

        # Synthetic function mode (acquiring from synthetic function, "continuous"
        #    optimization on "full" input space)
        else:
            n_iter = self.n_acquire
            # Initialize y_max_history with the best initial value
            self.y_max_history = np.array([np.max(self.y_init)], dtype=float)
            for _ in range(n_iter):
                # Propose the next sampling point by maximizing the acquisition
                #   function
                x_next = self.propose_location(self.acquisition, n_restarts=10)
                # Evaluate the objective function at x_next
                y_next = self.evaluate_objective(x_next)
                # Add the new sample to the data
                self.x_all_data = np.vstack((self.x_all_data, x_next))
                self.y_all_data = np.append(self.y_all_data, y_next)
                self.x_acquired = np.append(self.x_acquired, [x_next], axis=0)
                self.y_acquired = np.append(self.y_acquired, y_next)
                y_max = np.max(self.y_all_data)
                # Re-fit the GP model with the updated data
                self.gp_model_fit()
                # Update y_max_history
                self.y_max_history = np.append(self.y_max_history, y_max)

        return self.x_all_data, self.y_all_data, self.y_max_history


def plot_acquisition_comparison(
    max_output_EI: np.ndarray,
    max_output_PI: np.ndarray,
    max_output_UCB: np.ndarray,
    max_output_PV: np.ndarray,
    max_output_random: np.ndarray,
    kernel: str = "rbf",
    n_iter: int = 10,
    n_init: int = 5,
    objective_data: str = "___ data",
    xi: float = 0.0,
    kappa: float = 2.0,
) -> None:
    """
    Plot the maximum observed output versus iteration for different acquisition
    functions.

    This function generates a line plot comparing the progression of the maximum
    output over optimization iterations for several acquisition strategies:
    Expected Improvement (EI), Probability of Improvement (PI), Upper Confidence
    Bound (UCB), Predictive Variance (PV), and Uniform Random sampling. The plot is
    saved as a PNG file in the './plots' directory.

    Args:
        max_output_EI (np.ndarray): Array of maximum output per iteration using
            Expected Improvement.
        max_output_PI (np.ndarray): Array of maximum output per iteration using
            Probability of Improvement.
        max_output_UCB (np.ndarray): Array of maximum output per iteration using
            Upper Confidence Bound.
        max_output_PV (np.ndarray): Array of maximum output per iteration using
            maximum Predictive Variance.
        max_output_random (np.ndarray): Array of maximum output per iteration using
            random sampling.
        kernel (str): Name of the kernel used in the optimization (for plot filename).
        n_iter (str): Number of optimization iterations (for plot filename).
        n_init (str): Number of initial samples (for plot filename).
        objective_data (str, optional): Name or description of the dataset/objective
            (for plot filename). Defaults to "___ data".

    Returns:
        None: This function is for visualization and does not return any value.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(
        max_output_EI,
        marker="o",
        c="blue",
        label=f"EI (xi = {xi})",
    )
    plt.plot(
        max_output_PI,
        marker="o",
        c="orange",
        label=f"PI (xi = {xi})",
    )
    plt.plot(
        max_output_UCB,
        marker="o",
        c="green",
        label=f"UCB (kappa = {kappa})",
    )
    plt.plot(
        max_output_PV,
        marker="o",
        c="red",
        label="PV",
    )
    plt.plot(
        max_output_random,
        marker="o",
        c="purple",
        label="Uniform Random",
    )

    plt.title("Maximum Observed Output vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Maximum Output")

    # Set y-axis limits
    y_min = min(
        max_output_EI.min(),
        max_output_PI.min(),
        max_output_UCB.min(),
        max_output_PV.min(),
        max_output_random.min(),
    )
    plt.ylim(0.9 * y_min, 1.025)

    plt.legend()
    plt.grid()

    if not os.path.exists("plots"):
        os.makedirs("plots")
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    filepath = os.path.join(
        "plots",
        f"bo_{objective_data}_{kernel}_maxit_{n_iter}_init_{n_init}_{timestamp}.png",
    )
    plt.savefig(filepath)
    print(f"Figure saved to {filepath}")
