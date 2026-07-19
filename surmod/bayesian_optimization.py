from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Union, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from botorch.acquisition.analytic import (
    LogExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    PosteriorStandardDeviation,
)
from botorch.optim import optimize_acqf

from surmod.test_functions import load_test_function
from surmod.gpytorch_gaussian_process import GPSurrogate
from surmod.space_fill_design import generate_initial_design


def sample_parabola(
    n_initial: int,
    bounds_low: Union[float, Sequence[float], np.ndarray],
    bounds_high: Union[float, Sequence[float], np.ndarray],
    input_size: int,
    radius: float = 7,
    seed: int = 1,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = []

    while len(samples) < n_initial:
        x_point = rng.uniform(bounds_low, bounds_high, size=input_size)
        if np.linalg.norm(x_point) > radius:
            samples.append(x_point)

    return np.array(samples)


def sample_data(
    objective_function: str,
    bounds_low: Union[float, Sequence[float], np.ndarray],
    bounds_high: Union[float, Sequence[float], np.ndarray],
    n_initial: int,
    input_size: int = 2,
    init_design: str = "random",
    seed: int = 1,
    **design_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate input and output samples from the specified synthetic objective.

    Args:
        objective_function: Name of the objective function.
        bounds_low: Lower bounds.
        bounds_high: Upper bounds.
        n_initial: Number of initial points.
        input_size: Input dimension.
        init_design: One of 'random', 'lhd', 'maximin_lhd'.
        seed: Random seed (default: 1).
        design_kwargs: Extra kwargs forwarded to generate_initial_design().

    Returns:
        Tuple of:
            x_sample: shape (n_initial, input_size)
            y_sample: shape (n_initial,)
    """
    test_function = load_test_function(objective_function)

    if objective_function == "Parabola" and init_design == "random":
        x_data = sample_parabola(
            n_initial, bounds_low, bounds_high, input_size, seed=seed
        )
    else:
        x_data = generate_initial_design(
            bounds_low=bounds_low,
            bounds_high=bounds_high,
            n_samples=n_initial,
            method=init_design,
            seed=seed,
            **design_kwargs,
        )

    x_tensor = torch.as_tensor(x_data, dtype=torch.float32)
    y_tensor = test_function(x_tensor)

    x_sample = x_tensor.detach().cpu().numpy()
    y_sample = y_tensor.detach().cpu().numpy().reshape(-1)

    return x_sample, y_sample


def get_synth_global_optima(
    objective_function: str,
) -> Tuple[List[List[float]], float]:
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


def select_initial_dataset_indices(
    x: np.ndarray,
    n_init: int,
    method: str = "random",
    seed: int = 42,
    **design_kwargs,
) -> np.ndarray:
    """
    Select initial dataset rows.

    For method='random', sample rows uniformly without replacement.
    For method='lhd' or 'maximin_lhd', generate a space-filling design in
    normalized [0,1]^d space and map each design point to the nearest
    available dataset row, enforcing uniqueness.

    Args:
        x: Dataset inputs, assumed already normalized to [0,1], shape (n, d)
        n_init: Number of initial points
        method: 'random', 'lhd', or 'maximin_lhd'
        seed: Random seed
        design_kwargs: Extra arguments forwarded to generate_initial_design()

    Returns:
        Array of selected row indices, shape (n_init,)
    """
    rng = np.random.default_rng(seed)
    n_rows, dim = x.shape

    if n_init > n_rows:
        raise ValueError("n_init cannot exceed number of available dataset rows.")

    method = method.lower()

    if method == "random":
        return rng.choice(n_rows, size=n_init, replace=False)

    targets = generate_initial_design(
        bounds_low=np.zeros(dim),
        bounds_high=np.ones(dim),
        n_samples=n_init,
        method=method,
        seed=seed,
        **design_kwargs,
    )

    remaining = set(range(n_rows))
    selected = []

    for target in targets:
        remaining_list = np.sort(list(remaining))
        x_remaining = x[remaining_list]
        dists = np.linalg.norm(x_remaining - target, axis=1)
        best_local_idx = np.argmin(dists)
        chosen_idx = int(remaining_list[best_local_idx])
        selected.append(chosen_idx)
        remaining.remove(chosen_idx)

    return np.array(selected, dtype=int)


class BayesianOptimizer:
    def __init__(
        self,
        objective_function: str,
        x_init: np.ndarray,
        y_init: np.ndarray,
        kernel: str = "matern",
        isotropic: bool = False,
        acquisition_function: str = "EI",
        n_acquire: int = 10,
        seed: int = 42,
        noise_bounds: Optional[Tuple[float, float]] = None,
        fixed_noise: Optional[float] = None,
        init_design: str = "random",
        init_design_kwargs: Optional[dict] = None,
        **acquisition_kwargs,
    ):
        self.objective_function = objective_function
        self.x_init = np.asarray(x_init, dtype=float)
        self.y_init = np.asarray(y_init, dtype=float).reshape(-1)

        self.x_all_data = self.x_init.copy()
        self.y_all_data = self.y_init.copy()

        self.kernel = kernel
        self.isotropic = isotropic
        self.acquisition = acquisition_function
        self.n_acquire = n_acquire
        self.seed = seed
        self.noise_bounds = noise_bounds
        self.fixed_noise = fixed_noise
        self.acquisition_kwargs = acquisition_kwargs

        self.x_acquired = np.empty((0, self.x_init.shape[1]), dtype=float)
        self.y_acquired = np.empty((0,), dtype=float)
        self.y_max_history = np.empty((0,), dtype=float)

        self.gp_model: Optional[GPSurrogate] = None
        self.init_design = init_design
        self.init_design_kwargs = init_design_kwargs or {}

    def evaluate_objective(self, x_next: np.ndarray) -> np.ndarray:
        synthetic_function = load_test_function(self.objective_function)

        bounds = self._get_objective_bounds().cpu().numpy()
        x_next = np.asarray(x_next, dtype=np.float64).reshape(-1)
        x_next = np.clip(x_next, bounds[0], bounds[1])

        x_tensor = torch.as_tensor(x_next, dtype=torch.float64).unsqueeze(0)

        try:
            y_tensor = synthetic_function(x_tensor)
        except ValueError:
            print("Objective bounds low :", bounds[0])
            print("Objective bounds high:", bounds[1])
            print("Tried x_next        :", x_next)
            raise

        return y_tensor.detach().cpu().numpy().reshape(-1)

    def gp_model_fit(self) -> GPSurrogate:
        self.gp_model = GPSurrogate(
            x_train=self.x_all_data,
            y_train=self.y_all_data,
            kernel=self.kernel,
            isotropic=self.isotropic,
            scale_inputs=True,
            scale_outputs=True,
            noise_bounds=(
                self.noise_bounds if self.noise_bounds is not None else (1e-8, 1e-3)
            ),
            fixed_noise=self.fixed_noise,
        )
        self.gp_model.fit()
        return self.gp_model

    def _get_objective_bounds(self) -> torch.Tensor:
        synthetic_function = load_test_function(self.objective_function)
        bounds_low = [b[0] for b in synthetic_function._bounds]
        bounds_high = [b[1] for b in synthetic_function._bounds]

        return torch.tensor([bounds_low, bounds_high], dtype=torch.float64)

    def _build_analytic_acquisition(self):
        if self.gp_model is None or self.gp_model.model is None:
            raise ValueError(
                "GP model must be fit before building acquisition function."
            )

        model = self.gp_model.model
        acquisition_name = self.acquisition.upper()

        if acquisition_name == "EI":
            best_f = self.y_all_data.max()
            return LogExpectedImprovement(model=model, best_f=best_f)
        elif acquisition_name == "PI":
            best_f = self.y_all_data.max()
            return ProbabilityOfImprovement(model=model, best_f=best_f)
        elif acquisition_name == "UCB":
            beta = self.acquisition_kwargs.get("beta", 2.0)
            return UpperConfidenceBound(model=model, beta=beta)
        elif acquisition_name == "PV":
            return PosteriorStandardDeviation(model=model)
        else:
            raise ValueError(
                "Invalid acquisition function. Choose 'EI', 'PI', 'UCB', 'PV', or 'random'."
            )

    def propose_location(
        self,
        num_restarts: int = 30,
        raw_samples: int = 1000,
    ) -> np.ndarray:
        rng = np.random.RandomState(self.seed)
        bounds_t = self._get_objective_bounds()
        bounds = bounds_t.cpu().numpy()

        if self.acquisition.lower() == "random":
            x_next = rng.uniform(bounds[0], bounds[1])
            return np.clip(
                np.asarray(x_next, dtype=np.float64).reshape(-1), bounds[0], bounds[1]
            )

        acq_func = self._build_analytic_acquisition()

        candidate, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds_t,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

        x_next = candidate.detach().cpu().numpy().reshape(-1)
        x_next = np.clip(np.asarray(x_next, dtype=np.float64), bounds[0], bounds[1])
        return x_next

    def _score_candidates_discrete(
        self,
        x_candidates: np.ndarray,
    ) -> np.ndarray:
        rng = np.random.RandomState(self.seed)

        if self.acquisition.lower() == "random":
            return rng.uniform(size=x_candidates.shape[0])

        acq_func = self._build_analytic_acquisition()
        x_tensor = torch.as_tensor(x_candidates, dtype=torch.float64).unsqueeze(1)

        with torch.no_grad():
            values = acq_func(x_tensor).detach().cpu().numpy().reshape(-1)

        return values

    def score_candidates(
        self,
        x_candidates: np.ndarray,
    ) -> np.ndarray:
        return self._score_candidates_discrete(x_candidates)

    def _append_observation(
        self,
        x_next: np.ndarray,
        y_next_scalar: float,
    ) -> None:
        x_next = np.asarray(x_next, dtype=float).reshape(1, -1)

        self.x_all_data = np.vstack((self.x_all_data, x_next))
        self.y_all_data = np.append(self.y_all_data, y_next_scalar)
        self.x_acquired = np.vstack((self.x_acquired, x_next))
        self.y_acquired = np.append(self.y_acquired, y_next_scalar)
        self.y_max_history = np.append(self.y_max_history, np.max(self.y_all_data))

    def step(
        self,
        df: Optional[pd.DataFrame] = None,
        remaining_indices: Optional[set[int]] = None,
        x_grid: Optional[np.ndarray] = None,
        grid_shape: Optional[tuple[int, int]] = None,
        return_diagnostics: bool = False,
    ) -> dict:
        self.gp_model_fit()
        gp = self.gp_model
        if gp is None:
            raise ValueError("GP model failed to fit.")

        snapshot: dict = {}

        if return_diagnostics and x_grid is not None:
            mu, _ = gp.predict(x_grid)

            acq_values = self.score_candidates(x_grid)

            if self.acquisition.upper() == "EI":
                acq_values = np.exp(acq_values)

            snapshot["gp_mean_max_value"] = float(np.max(mu))
            snapshot["gp_mean_max_location"] = np.asarray(x_grid[np.argmax(mu), :])
            snapshot["mu"] = mu.reshape(grid_shape) if grid_shape is not None else mu
            snapshot["acq_values"] = (
                acq_values.reshape(grid_shape) if grid_shape is not None else acq_values
            )

        if df is None:
            x_next = self.propose_location()
            y_next = self.evaluate_objective(x_next)
            y_next_scalar = float(y_next[0])

        else:
            if remaining_indices is None or len(remaining_indices) == 0:
                raise ValueError(
                    "remaining_indices must be provided and non-empty for dataset BO."
                )

            x = df.iloc[:, :-1].to_numpy(dtype=float)
            y = df.iloc[:, -1].to_numpy(dtype=float).reshape(-1)

            remaining_list = list(remaining_indices)
            x_remaining = x[remaining_list]

            acquisition_values = self.score_candidates(x_remaining)
            next_idx_in_remaining = int(np.argmax(acquisition_values))
            next_index = remaining_list[next_idx_in_remaining]

            x_next = x[next_index]
            y_next_scalar = float(y[next_index])
            snapshot["selected_index"] = next_index

        self._append_observation(x_next, y_next_scalar)

        x_best = self.x_all_data[np.argmax(self.y_all_data), :]

        snapshot.update(
            dict(
                x_next=np.asarray(x_next, dtype=float),
                y_next=y_next_scalar,
                y_max=float(np.max(self.y_all_data)),
                x_best=x_best,
                acquired_max=float(np.max(self.y_all_data)),
            )
        )

        return snapshot

    def bayes_opt(
        self,
        df: Optional[pd.DataFrame] = None,
        n_init: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if df is not None:
            df = df.copy()

            x = df.iloc[:, :-1].to_numpy(dtype=float)
            y = df.iloc[:, -1].to_numpy(dtype=float).reshape(-1)

            n_total = len(df)
            if n_init > n_total:
                raise ValueError("n_init cannot exceed the number of rows in df.")

            # Only for LHD / maximin-LHD matching, not for GP training
            x_min = x.min(axis=0)
            x_max = x.max(axis=0)
            x_range = np.where(x_max > x_min, x_max - x_min, 1.0)
            x_for_init = (x - x_min) / x_range

            initial_indices = select_initial_dataset_indices(
                x=x_for_init,
                n_init=n_init,
                method=self.init_design,
                seed=self.seed,
                **self.init_design_kwargs,
            )
            remaining_indices = set(range(n_total)) - set(initial_indices)

            self.x_init = x[initial_indices].copy()
            self.y_init = y[initial_indices].copy()
            self.x_all_data = self.x_init.copy()
            self.y_all_data = self.y_init.copy()
            self.x_acquired = np.empty((0, self.x_init.shape[1]), dtype=float)
            self.y_acquired = np.empty((0,), dtype=float)
            self.y_max_history = np.array([np.max(self.y_all_data)], dtype=float)

            for _ in range(self.n_acquire):
                if len(remaining_indices) == 0:
                    break

                snapshot = self.step(
                    df=df,
                    remaining_indices=remaining_indices,
                    return_diagnostics=False,
                )
                remaining_indices.remove(snapshot["selected_index"])

            return self.x_all_data, self.y_all_data, self.y_max_history

        self.x_all_data = self.x_init.copy()
        self.y_all_data = self.y_init.copy()
        self.x_acquired = np.empty((0, self.x_init.shape[1]), dtype=float)
        self.y_acquired = np.empty((0,), dtype=float)
        self.y_max_history = np.array([np.max(self.y_all_data)], dtype=float)

        for _ in range(self.n_acquire):
            self.step(return_diagnostics=False)

        return self.x_all_data, self.y_all_data, self.y_max_history

    def _clip_to_objective_bounds(self, x: np.ndarray) -> np.ndarray:
        bounds = self._get_objective_bounds().cpu().numpy()
        return np.clip(np.asarray(x, dtype=float), bounds[0], bounds[1])


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
    beta: float = 2.0,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(max_output_EI, marker="o", c="blue", label="EI")
    plt.plot(max_output_PI, marker="o", c="orange", label="PI")
    plt.plot(max_output_UCB, marker="o", c="green", label=f"UCB (beta = {beta})")
    plt.plot(max_output_PV, marker="o", c="red", label="PV")
    plt.plot(max_output_random, marker="o", c="purple", label="Uniform Random")

    plt.title("Best Observed Value vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Best Observed Value")

    all_outputs = [
        max_output_EI,
        max_output_PI,
        max_output_UCB,
        max_output_PV,
        max_output_random,
    ]
    y_min = np.min(all_outputs)
    y_max = np.max(all_outputs)

    if np.isfinite(y_min) and np.isfinite(y_max) and y_min != y_max:
        plt.ylim(0.95 * y_min, 1.05 * y_max)

    plt.legend()
    plt.grid()

    Path("plots").mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    filepath = (
        Path("plots")
        / f"bo_{objective_data}_{kernel}_maxit_{n_iter}_init_{n_init}_{timestamp}.png"
    )
    plt.savefig(filepath, bbox_inches="tight")
    print(f"Figure saved to {filepath}")
