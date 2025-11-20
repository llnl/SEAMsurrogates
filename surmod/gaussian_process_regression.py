"""
Functions for Gaussian process surrogates.
"""

import os
import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

from botorch.test_functions.synthetic import (
    Ackley,
    Branin,
    Griewank,
    HolderTable,
)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    Matern,
    WhiteKernel,
    Kernel,
    DotProduct,
)

from surmod import parabola


def get_kernel(
    kernel: str,
    dim: int,
    isotropic: bool = False,
) -> Kernel:
    """
    Function to generate kernel for GPregressor in sklearn.

    Args:
        kernel (str): Choice of kernel, accepts "rbf", "matern", and "matern_dot".
        dim (int): Dimension of inputs for the kernel.
        isotropic (bool, optional): If True, sets the model to an isotropic kernel with a single lengthscale for all inputs. Defaults to False.

    Raises:
        ValueError: If kernel is not one of the approved options.

    Returns:
        Kernel: An sklearn kernel object.
    """
    constant_kernel = ConstantKernel(
        constant_value=1.0,
        constant_value_bounds=(1e-5, 1e5),
    )
    length_scale = 1.0 if isotropic else [1.0] * dim
    bounds = (1e-2, 1e2)
    if kernel == "rbf":
        return constant_kernel * RBF(
            length_scale=length_scale, length_scale_bounds=bounds
        ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-16, 1e1))
    elif kernel == "matern":
        return constant_kernel * Matern(
            nu=2.5, length_scale=length_scale, length_scale_bounds=bounds
        ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-16, 1e1))
    elif kernel == "matern_dot":
        return constant_kernel * (
            Matern(
                nu=2.5,
                length_scale=length_scale,
                length_scale_bounds=bounds,
            )
            * DotProduct()
        ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-16, 1e1))
    else:
        raise ValueError(
            f"Kernel '{kernel}' not found. Choose from 'rbf', 'matern',"
            " or 'matern_dot'."
        )


def load_test_function(objective_function: str):
    """
    Loads a test function instance for simulating data based on the given
    objective function name.

    Args:
        objective_function (str): The name of the objective function to load.
            Supported values are "Parabola", "Ackley", "Griewank", "Branin",
            and "HolderTable".

    Returns:
        object: An instance of the requested test function, initialized with
        standard parameters.

    Raises:
        ValueError: If the specified objective function name is not recognized.
    """
    if objective_function == "Parabola":
        test_function = parabola.Parabola(
            dim=2, negate=True, bounds=[(-25, 25), (-25, 25)]
        )
    elif objective_function == "Ackley":
        test_function = Ackley(
            dim=2, negate=True, bounds=[(-32.768, 32.768), (-32.768, 32.768)]
        )
    elif objective_function == "Griewank":
        test_function = Griewank(dim=2, negate=True, bounds=[(-100, 45), (-100, 45)])
    elif objective_function == "Branin":
        test_function = Branin(negate=True)
    elif objective_function == "HolderTable":
        test_function = HolderTable(negate=True)
    else:
        raise ValueError(
            f"Test function '{objective_function}' not found. "
            "Choose from 'Parabola', 'Ackley', 'Griewank', 'Branin',"
            " or 'HolderTable'."
        )
    return test_function


def simulate_data(objective_function: str, num_train: int, num_test: int):
    """
    Simulates training and testing data from a specified test function.

    Args:
        objective_function (str): The name of the objective function to simulate
            data from. Supported values are "Parabola", "Ackley", "Griewank",
            "Branin", and "HolderTable".
        num_train (int): Number of training samples to generate.
        num_test (int): Number of testing samples to generate.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            A tuple containing:
                - x_train (np.ndarray): Training input data of shape (num_train, 2).
                - x_test (np.ndarray): Testing input data of shape (num_test, 2).
                - y_train (np.ndarray): Training target data of shape (num_train,).
                - y_test (np.ndarray): Testing target data of shape (num_test,).

    Raises:
        ValueError: If the specified objective function name is not recognized.
    """
    # Set-up simulation
    num_total = num_train + num_test
    test_function = load_test_function(objective_function)
    bounds_low = [b[0] for b in test_function._bounds]
    bounds_high = [b[1] for b in test_function._bounds]

    # Sample random data from test function
    np.random.seed(1)
    x_data = np.random.uniform(bounds_low, bounds_high, size=(num_total, 2))
    y_data = np.array(test_function(torch.tensor(x_data)))

    # Split data into training and testing sets
    x_train = x_data.copy()[:num_train]
    y_train = y_data.copy()[:num_train]

    x_test = x_data.copy()[num_train:]
    y_test = y_data.copy()[num_train:]

    return x_train, x_test, y_train, y_test


def compute_max_error(output: np.ndarray, target: np.ndarray, inputs: np.ndarray):
    """
    Computes the maximum absolute error between prediction and target values.

    Args:
        output (np.ndarray): Predicted values, shape (n_samples,) or (n_samples, n_outputs).
        target (np.ndarray): True target values, same shape as output.
        inputs (np.ndarray): Input data corresponding to each prediction, shape (n_samples, n_features).

    Returns:
        Tuple[float, np.ndarray]:
            max_error_value: Maximum absolute error value.
            max_error_inputs: Input(s) corresponding to the maximum error(s).
    """
    abs_errors = np.abs(output - target)
    abs_errors_flat = abs_errors.flatten()
    max_error_index_flat = np.argmax(abs_errors_flat)
    # Map back to sample index (row in inputs)
    sample_index = max_error_index_flat % inputs.shape[0]
    max_error_value = abs_errors_flat[max_error_index_flat]
    max_error_inputs = inputs[sample_index]
    return float(max_error_value), max_error_inputs


def log_results(message: str, path_to_log: str):
    """
    Writes a message to a log file at the specified path.

    Creates the directory "./output_log" if it does not exist.
    Appends the message to the log file.
    Prints a confirmation message to the console.

    Args:
        message (str): The message to be written to the log file.
        path_to_log (str): The path to the log file where the message will be appended.
    """
    if not os.path.exists("output_log"):
        os.makedirs("output_log")
    with open(path_to_log, "a") as f:
        f.write("\n--------------------\n\n" + message)
    print(f"Output log saved to end of {path_to_log}.")


def plot_gp_mean_prediction(
    x_train: np.ndarray,
    y_train: np.ndarray,
    gp_model: GaussianProcessRegressor,
    test_mse: float,
    kernel: Kernel,
    objective_data_name: str,
    alpha: float,
    scale_x: bool,
    normalize_y: bool,
    input_scaler=None,
) -> None:
    """
    Plots the mean prediction surface of a Gaussian Process (GP) model along
    with training data and the true test function.

    Creates a 3D plot visualizing the GP mean prediction, the true test
    function, and training points. Saves the plot as a PNG file in the
    './plots' directory, creating the directory if it does not exist.
    Prints the path to the saved figure.

    Args:
       x_train (np.ndarray): Training input data, shape (n_samples, 2).
        y_train (np.ndarray): Training target values, shape (n_samples,).
       gp_model: Trained GP model object with a predict method.
        test_mse (float): Mean squared error of the GP model on test data.
       kernel: Kernel used in the GP model (for display in the plot title).
        objective_data_name (str): Name of the test function or dataset.
       alpha (float): Alpha parameter used in the GP model (for display in the plot title).
       scale_x (bool): Whether or not the input features were scaled, for display.
       normalize_y (bool): Whether or not the output targets were normalized, for display.
       input_scaler (Optional[object]): Scaler object for input normalization, with transform and inverse_transform methods. Default is None.

    """
    # Calculate test rmse
    test_rmse = np.sqrt(test_mse)

    # Set-up test function and bounds
    test_function = load_test_function(objective_data_name)
    bounds_low = [b[0] for b in test_function._bounds]
    bounds_high = [b[1] for b in test_function._bounds]

    # Generate a grid for contour plotting
    x1 = np.linspace(bounds_low[0], bounds_high[0], 100)
    x2 = np.linspace(bounds_low[1], bounds_high[1], 100)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    x_grid = np.vstack([x1_grid.ravel(), x2_grid.ravel()]).T

    # Evaluate the test function on the original scale
    y_grid = np.array([test_function(torch.tensor(x)) for x in x_grid]).reshape(
        x1_grid.shape
    )

    # Evaluate GP model on test grid
    if input_scaler is not None:
        x_grid = input_scaler.transform(x_grid)

    mu = gp_model.predict(x_grid, return_std=False)
    if isinstance(mu, tuple):
        mu = mu[0]  # take the mean
    mu = mu.reshape(x1_grid.shape)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the surface and contour
    ax.plot_surface(x1_grid, x2_grid, mu, cmap="viridis", alpha=0.6)  # type: ignore
    ax.contour(
        x1_grid,
        x2_grid,
        y_grid,
        levels=20,
        cmap="inferno",
        linestyles="solid",
    )

    # Plot the training points
    if input_scaler is not None:
        x_train = input_scaler.inverse_transform(x_train)

    ax.scatter(
        x_train[:, 0],
        x_train[:, 1],
        y_train,
        color="red",
        s=20,  # type: ignore
        label="Training Points",
        marker="o",
    )

    title_lines = [
        f"{objective_data_name} Test Function and GP Mean",
        f"Training samples: {len(x_train)}",
        f"Alpha: {alpha}",
        f"kernel: {kernel}",
        f"Scale_x: {scale_x}",
        f"Normalize_y: {normalize_y}",
        f"Test RMSE: {test_rmse:.5f}",
    ]

    # Set plot labels
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("Value")  # type: ignore
    ax.set_title("\n".join(title_lines))
    ax.legend()

    # Specify where to save plot of GP fit, create directory if it doesn't exist
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    if not os.path.exists("plots"):
        os.makedirs("plots")
    path_to_plot = os.path.join(
        "plots", f"{objective_data_name}_gp_mean_{timestamp}.png"
    )
    plt.tight_layout()
    plt.savefig(path_to_plot)
    print(f"Figure saved to {path_to_plot}")


def plot_gp_std_dev_prediction(
    x_train: np.ndarray,
    gp_model: GaussianProcessRegressor,
    test_mse: float,
    kernel: Kernel,
    objective_data_name: str,
    alpha: float,
    scale_x: bool,
    normalize_y: bool,
    input_scaler=None,
) -> None:
    """
    Plots the predictive standard deviation (uncertainty) of a trained Gaussian
    Process (GP) model over a 2D input space, using a heatmap. Also displays the
    training points and relevant model information.

    The plot is saved as a PNG file.

    Args:
        x_train (np.ndarray): Training input features used to fit the GP model.
        gp_model (GaussianProcessRegressor): Trained Gaussian Process model for prediction.
        test_mse (float): Mean squared error on the test set, used for RMSE calculation.
        kernel (Kernel): Kernel object used in the GP model, for display in the plot title.
        objective_data_name (str): Name of the objective or dataset, used for labeling and saving the plot.
        alpha (float): Noise level or regularization parameter used in the GP model, for display.
        scale_x (bool): Whether or not the input features were scaled, for display.
        normalize_y (bool): Whether or not the outputs were normalized, for display.
        input_scaler (optional): Fitted scaler or transformer for input normalization, if used.

    """
    # Calculate test rmse
    test_rmse = np.sqrt(test_mse)

    # Set up test function and bounds
    test_function = load_test_function(objective_data_name)
    bounds_low = [b[0] for b in test_function._bounds]
    bounds_high = [b[1] for b in test_function._bounds]

    # Generate a grid for plotting
    x1 = np.linspace(bounds_low[0], bounds_high[0], 100)
    x2 = np.linspace(bounds_low[1], bounds_high[1], 100)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    x_grid = np.vstack([x1_grid.ravel(), x2_grid.ravel()]).T

    # Evaluate GP model variance on test grid
    if input_scaler is not None:
        x_grid = input_scaler.transform(x_grid)

    __, std = gp_model.predict(x_grid, return_std=True)  # type: ignore
    std_grid = std.reshape(x1_grid.shape)

    # Create a 2D heatmap plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.grid(False)
    c = ax.imshow(
        std_grid,
        extent=(bounds_low[0], bounds_high[0], bounds_low[1], bounds_high[1]),
        origin="lower",
        cmap="viridis",
        aspect="auto",
    )

    # Plot the training points
    if input_scaler is not None:
        x_train = input_scaler.inverse_transform(x_train)

    ax.scatter(
        x_train[:, 0],
        x_train[:, 1],
        color="red",
        s=20,
        label="Training Points",
        marker="o",
        edgecolor="black",
    )

    title_lines = [
        f"{objective_data_name} GP Predictive Standard Deviation",
        f"Training samples: {len(x_train)}",
        f"Alpha: {alpha}",
        f"kernel: {kernel}",
        f"Scale_x: {scale_x}",
        f"Normalize_y: {normalize_y}",
        f"Test RMSE: {test_rmse:.5f}",
    ]

    # Set plot labels
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("\n".join(title_lines))
    legend = ax.legend(loc="upper right")
    for text in legend.get_texts():
        text.set_color("white")

    # Add color bar for variance
    fig.colorbar(c, ax=ax, label="Predictive Standard Deviation")

    # Save plot
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    if not os.path.exists("plots"):
        os.makedirs("plots")
    path_to_plot = os.path.join(
        "plots", f"{objective_data_name}_gp_std_dev_{timestamp}.png"
    )
    plt.tight_layout()
    plt.savefig(path_to_plot)
    print(f"Figure saved to {path_to_plot}")


def plot_test_predictions(
    x_test: np.ndarray,
    observed: np.ndarray,
    gp_model: GaussianProcessRegressor,
    objective_data_name: str,
) -> None:
    """
    Plots the predicted vs. observed values for test data using a Gaussian
    Process model, including error bars for the 95% prediction interval.

    Calculates and displays RMSE and coverage of the prediction interval. Saves
    the plot as a PNG file.

    Args:
        x_test (np.ndarray): Test input features for prediction.
        observed (np.ndarray): Observed target values corresponding to x_test.
        gp_model (GaussianProcessRegressor): Trained Gaussian Process model for prediction.
        objective_data_name (str): Name of the objective or dataset, used for labeling and saving the plot.

    """
    prediction_mean, std_dev = gp_model.predict(x_test, return_std=True)  # type: ignore

    # Calculate 95% prediction interval coverage
    lower_bounds = prediction_mean.flatten() - 1.96 * std_dev.flatten()
    upper_bounds = prediction_mean.flatten() + 1.96 * std_dev.flatten()
    coverage = np.mean(
        (observed.flatten() >= lower_bounds) & (observed.flatten() <= upper_bounds)
    )

    # Calculate MSE
    mse = np.mean((observed.flatten() - prediction_mean.flatten()) ** 2)
    # Calculate RMSE
    rmse = np.sqrt(mse)

    # Set Seaborn style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Create a plot
    plt.figure()

    # Plot truth vs prediction mean with error bars
    plt.errorbar(
        observed,
        prediction_mean,
        yerr=1.96 * std_dev,
        fmt="o",
        capsize=5,
        color="blue",
    )

    # Add a line for y = x
    max_value = max(np.max(observed), max(upper_bounds))
    min_value = min(np.min(observed), min(lower_bounds))
    plt.plot([min_value, max_value], [min_value, max_value], "k-", linewidth=2)

    # Add labels and title
    plt.ylabel("Predicted", fontsize=14)
    plt.xlabel("Observed", fontsize=14)
    plt.title(f"{objective_data_name} \n {gp_model.kernel_}")
    plt.text(
        0.3,
        0.95,
        f"RMSE: {rmse:.5f}, Coverage: {coverage:.2%}",
        ha="center",
        fontsize=14,
        transform=plt.gca().transAxes,
    )
    plt.tight_layout()

    # Save the plot
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    if not os.path.exists("plots"):
        os.makedirs("plots")
    path_to_plot = os.path.join(
        "plots", f"{objective_data_name}_test_predictions_{timestamp}.png"
    )
    plt.savefig(path_to_plot, bbox_inches="tight")
    print(f"Figure saved to {path_to_plot}")
