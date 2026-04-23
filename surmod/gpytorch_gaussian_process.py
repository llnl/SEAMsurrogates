# wrapper for gpytoch GP fitting
import os
from datetime import datetime
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import copy
import warnings
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.exceptions.errors import ModelFittingError
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, PeriodicKernel, RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from numpy.typing import NDArray


def fit_gpytorch_mll_multistart(
    build_model_and_mll,
    n_restarts: int = 10,
    seed: int | None = None,
):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    best_model = None
    best_mll = None
    best_loss = float("inf")

    for i in range(n_restarts):
        model, mll = build_model_and_mll()
        model.train()
        mll.train()

        with torch.no_grad():
            if hasattr(model.covar_module.base_kernel, "lengthscale"):
                model.covar_module.base_kernel.lengthscale = 10 ** np.random.uniform(
                    -2, 1
                )

            if hasattr(model.covar_module, "outputscale"):
                model.covar_module.outputscale = 10 ** np.random.uniform(-1, 1)

            if hasattr(model.likelihood, "noise"):
                model.likelihood.noise = 10 ** np.random.uniform(-6, -1)

        abnormal = False
        fit_failed = False
        warning_msgs = []

        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                fit_gpytorch_mll(mll)

            for w in caught_warnings:
                msg = str(w.message)
                if "ABNORMAL" in msg or "OptimizationStatus.FAILURE" in msg:
                    abnormal = True
                    warning_msgs.append(msg)

        except ModelFittingError as e:
            fit_failed = True
            warning_msgs.append(f"ModelFittingError: {e}")
        except Exception as e:
            fit_failed = True
            warning_msgs.append(f"{type(e).__name__}: {e}")

        model.train()
        mll.train()

        try:
            with torch.no_grad():
                output = model(model.train_inputs[0])
                loss = -mll(output, model.train_targets).item()
        except Exception as e:
            print(f"restart {i + 1}/{n_restarts}, failed to evaluate loss: {e}")
            for msg in warning_msgs:
                print(f"  warning: {msg}")
            continue

        if not np.isfinite(loss):
            print(f"restart {i + 1}/{n_restarts}, loss is not finite, skipping")
            for msg in warning_msgs:
                print(f"  warning: {msg}")
            continue

        if fit_failed:
            status = "FIT_FAILED_BUT_SCORED"
        elif abnormal:
            status = "ABNORMAL"
        else:
            status = "OK"

        print(f"restart {i + 1}/{n_restarts}, loss={loss:.6f}, status={status}")

        for msg in warning_msgs:
            print(f"  warning: {msg}")

        if loss < best_loss:
            best_loss = loss
            best_model = copy.deepcopy(model)
            best_mll = copy.deepcopy(mll)

    if best_model is None or best_mll is None:
        raise RuntimeError("All multistart GP fits failed or produced invalid losses.")

    return best_model, best_mll, best_loss


class GPSurrogate:
    """
    Gaussian Process surrogate model using BoTorch SingleTaskGP.

    This class stores training and optional test data, optionally applies
    input normalization and output standardization, fits a GP model with
    BoTorch, and provides prediction, evaluation, and plotting utilities.

    Args:
        x_train: Training input array of shape (n_train, n_features).
        y_train: Training target array of shape (n_train,) or (n_train, 1).
        x_test: Optional test input array of shape (n_test, n_features).
        y_test: Optional test target array of shape (n_test,) or (n_test, 1).
        kernel: Kernel type, one of "rbf", "matern", or "periodic".
        isotropic: If True, use a shared lengthscale. If False, use ARD.
        scale_inputs: Whether to normalize inputs to the unit cube.
        scale_outputs: Whether to standardize outputs.
        lengthscale_bounds: Bounds on the lengthscale parameter(s), current option is for inputs scaled to [0,1].  Defaults to [1e-2,10]
        noise_bounds: Bounds on the nugget parameter, default is assuming output scaled to mean 0, variance 1. Defaults to [1e-16,1e-1]
    """

    def __init__(
        self,
        x_train: NDArray,
        y_train: NDArray,
        x_test: Optional[NDArray] = None,
        y_test: Optional[NDArray] = None,
        kernel: str = "rbf",
        isotropic: bool = False,
        scale_inputs: bool = True,
        scale_outputs: bool = True,
        lengthscale_bounds: tuple[float, float] = (1e-2, 100.0),
        noise_bounds: tuple[float, float] = (1e-16, 1e-1),
        optimization_restarts: int = 3,
    ) -> None:
        """
        Initialize the GP surrogate model.

        Args:
            x_train: Training inputs.
            y_train: Training outputs.
            x_test: Optional test inputs.
            y_test: Optional test outputs.
            kernel: Covariance kernel type.
            isotropic: Whether to use isotropic lengthscales.
            scale_inputs: Whether to normalize inputs.
            scale_outputs: Whether to standardize outputs.
            lengthscale_bounds: Bounds on the lengthscale parameter(s), current option is for inputs scaled to [0,1].  Defaults to [1e-2,10]
            noise_bounds: Bounds on the nugget parameter, default is assuming output scaled to mean 0, variance 1. Defaults to [1e-16,1e-1]
            optimization_restrats: Number of times to randomly initialize the hyperparamter optimization. Defaults to 5
        """
        self.x_train: torch.Tensor = torch.as_tensor(x_train, dtype=torch.float64)
        self.y_train: torch.Tensor = torch.as_tensor(
            y_train, dtype=torch.float64
        ).reshape(-1, 1)

        self.x_test: Optional[torch.Tensor] = (
            None if x_test is None else torch.as_tensor(x_test, dtype=torch.float64)
        )
        self.y_test: Optional[torch.Tensor] = (
            None
            if y_test is None
            else torch.as_tensor(y_test, dtype=torch.float64).reshape(-1, 1)
        )

        self.kernel: str = kernel
        self.isotropic: bool = isotropic
        self.scale_inputs: bool = scale_inputs
        self.scale_outputs: bool = scale_outputs
        self.optimization_restarts: int = optimization_restarts
        self.model: Optional[SingleTaskGP] = None
        self.mll: Optional[ExactMarginalLogLikelihood] = None
        self.lengthscale_bounds = lengthscale_bounds
        self.noise_bounds = noise_bounds
        self._build_model()

    def _get_covar_module(self) -> ScaleKernel:
        """
        Construct the covariance module for the GP model.

        Returns:
            A scaled kernel module.

        Raises:
            ValueError: If the kernel type is unsupported.
        """
        input_dim = self.x_train.shape[1]
        ard_num_dims = None if self.isotropic else input_dim
        lengthscale_constraint = Interval(*self.lengthscale_bounds)

        if self.kernel == "rbf":
            base_kernel = RBFKernel(
                ard_num_dims=ard_num_dims, lengthscale_constraint=lengthscale_constraint
            )
        elif self.kernel == "matern":
            base_kernel = MaternKernel(
                nu=2.5,
                ard_num_dims=ard_num_dims,
                lengthscale_constraint=lengthscale_constraint,
            )
        elif self.kernel == "periodic":
            base_kernel = PeriodicKernel(
                ard_num_dims=ard_num_dims,
                lengthscale_constraint=lengthscale_constraint,
            )
        else:
            raise ValueError("kernel must be 'rbf', 'matern', or 'periodic'")

        return ScaleKernel(base_kernel)

    def _build_fresh_model_and_mll(self):
        self._build_model()
        return self.model, self.mll

    def _build_model(self) -> None:
        """
        Build the BoTorch SingleTaskGP model.

        Applies optional input and output transforms.

        Returns:
            None
        """
        input_transform = (
            Normalize(d=self.x_train.shape[1]) if self.scale_inputs else None
        )
        outcome_transform = Standardize(m=1) if self.scale_outputs else None

        self.model = SingleTaskGP(
            train_X=self.x_train,
            train_Y=self.y_train,
            covar_module=self._get_covar_module(),
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )

        self.model.likelihood.noise_covar.register_constraint(
            "raw_noise",
            Interval(*self.noise_bounds),
        )
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

    def fit(self) -> None:
        """
        Fit the GP model by maximizing the exact marginal log likelihood.

        Returns:
            None
        """
        best_model, best_mll, best_loss = fit_gpytorch_mll_multistart(
            self._build_fresh_model_and_mll,
            n_restarts=self.optimization_restarts,
        )

        if best_model is None or best_mll is None:
            raise RuntimeError("Multistart fitting failed to produce a model.")

        self.model = best_model
        self.mll = best_mll
        print(f"best loss: {best_loss:.6f}")
        self.model.eval()
        self.mll.eval()

    def predict(
        self,
        x: Optional[NDArray | torch.Tensor] = None,
    ) -> Tuple[NDArray, NDArray]:
        """
        Predict posterior mean and standard deviation for input points.

        If x is not provided, stored test inputs are used.

        Args:
            x: Optional prediction inputs.

        Returns:
            A tuple containing:
                - mean: Posterior mean predictions.
                - std: Posterior standard deviations.

        Raises:
            ValueError: If no prediction inputs are available.
        """
        if self.model is None:
            raise ValueError("Model has not been built.")

        if x is None:
            if self.x_test is None:
                raise ValueError("No prediction data provided and x_test is not set.")
            x_tensor = self.x_test
        elif isinstance(x, torch.Tensor):
            x_tensor = x
        else:
            x_tensor = torch.as_tensor(x, dtype=torch.float64)

        self.model.eval()

        with torch.no_grad():
            posterior = self.model.posterior(x_tensor)
            mean = posterior.mean.squeeze(-1).cpu().numpy()
            std = posterior.variance.sqrt().squeeze(-1).cpu().numpy()

        return mean, std

    def evaluate(self) -> dict[str, Any]:
        """
        Evaluate the GP model on the stored test dataset.

        Computes MSE, RMSE, and 95 percent interval coverage.

        Returns:
            Dictionary of evaluation metrics and predictions.

        Raises:
            ValueError: If test data is unavailable.
        """
        if self.x_test is None or self.y_test is None:
            raise ValueError("x_test and y_test must be provided for evaluation.")

        mean, std = self.predict(self.x_test)
        y_true = self.y_test.squeeze(-1).cpu().numpy()

        mse = float(np.mean((y_true - mean) ** 2))
        rmse = float(np.sqrt(mse))
        lower = mean - 1.96 * std
        upper = mean + 1.96 * std
        coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))

        return {
            "mse": mse,
            "rmse": rmse,
            "coverage": coverage,
            "mean": mean,
            "std": std,
        }

    @staticmethod
    def compute_max_error(
        output: NDArray,
        target: NDArray,
        inputs: NDArray,
    ) -> Tuple[float, NDArray]:
        """
        Compute the maximum absolute error and the corresponding input.

        Args:
            output: Predicted values.
            target: True values.
            inputs: Inputs corresponding to predictions.

        Returns:
            Maximum absolute error and associated input row.
        """
        abs_errors = np.abs(output - target)
        max_idx = np.argmax(abs_errors.flatten())
        sample_index = max_idx % inputs.shape[0]
        return float(abs_errors.flatten()[max_idx]), inputs[sample_index]

    def sample_posterior(
        self,
        x: np.ndarray | None = None,
        n_samples: int = 1,
    ) -> np.ndarray:
        """
        Draw samples from the GP posterior at the given input points.

        Args:
            x: Optional prediction inputs of shape (n_points, n_features). If not
                provided, stored test inputs are used.
            n_samples: Number of posterior samples to draw.

        Returns:
            Posterior samples as a NumPy array of shape
            (n_samples, n_points).
        """
        if self.model is None:
            raise ValueError("Model has not been built.")

        if x is None:
            if self.x_test is None:
                raise ValueError("No input data provided and x_test is not set.")
            x_tensor = self.x_test
        else:
            x_tensor = torch.as_tensor(x, dtype=torch.float64)

        self.model.eval()

        with torch.no_grad():
            posterior = self.model.posterior(x_tensor)
            samples = posterior.rsample(torch.Size([n_samples]))
            samples = samples.squeeze(-1).cpu().numpy()

        return samples

    def posterior_gradient(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the gradient of the posterior mean with respect to the inputs.

        Args:
            x: Input array of shape (n_points, n_features).

        Returns:
            Gradient of the posterior mean with respect to x, as a NumPy array
            of shape (n_points, n_features).
        """
        if self.model is None:
            raise ValueError("Model has not been built.")

        x_tensor = torch.as_tensor(x, dtype=torch.float64, device=self.x_train.device)
        x_tensor = x_tensor.clone().detach().requires_grad_(True)

        self.model.eval()

        posterior = self.model.posterior(x_tensor)
        mean = posterior.mean.squeeze(-1)

        grad = torch.autograd.grad(
            outputs=mean.sum(),
            inputs=x_tensor,
            create_graph=False,
            retain_graph=False,
        )[0]

        return grad.detach().cpu().numpy()

    def plot_test_predictions(
        self, objective_data_name: str = "GP Test Predictions"
    ) -> None:
        """
        Plot observed versus predicted test values with 95 percent intervals.

        Args:
            objective_data_name: Plot and file label.

        Returns:
            None
        """
        if self.x_test is None or self.y_test is None:
            raise ValueError("x_test and y_test must be provided for plotting.")

        results = self.evaluate()
        prediction_mean = results["mean"]
        std_dev = results["std"]
        rmse = results["rmse"]
        coverage = results["coverage"]
        observed = self.y_test.squeeze(-1).cpu().numpy()

        plt.style.use("seaborn-v0_8-whitegrid")
        plt.figure()

        plt.errorbar(
            observed,
            prediction_mean,
            yerr=1.96 * std_dev,
            fmt="o",
            capsize=5,
            color="blue",
        )

        max_value = max(np.max(observed), np.max(prediction_mean + 1.96 * std_dev))
        min_value = min(np.min(observed), np.min(prediction_mean - 1.96 * std_dev))
        plt.plot([min_value, max_value], [min_value, max_value], "k-", linewidth=2)

        plt.ylabel("Predicted", fontsize=14)
        plt.xlabel("Observed", fontsize=14)
        plt.title(objective_data_name)
        plt.text(
            0.3,
            0.95,
            f"RMSE: {rmse:.5f}, Coverage: {coverage:.2%}",
            ha="center",
            fontsize=14,
            transform=plt.gca().transAxes,
        )
        plt.tight_layout()

        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        if not os.path.exists("plots"):
            os.makedirs("plots")
        path_to_plot = os.path.join(
            "plots", f"{objective_data_name}_test_predictions_{timestamp}.png"
        )
        plt.savefig(path_to_plot, bbox_inches="tight")
        print(f"Figure saved to {path_to_plot}")
