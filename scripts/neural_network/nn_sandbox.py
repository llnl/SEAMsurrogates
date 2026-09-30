#!/usr/bin/env python3

"""
Train neural network surrogate models on synthetic test functions.

This script trains feedforward neural networks on 2D test functions, with options
for customizing network architecture, learning rate, batch size, and data scaling.
Supports single-run and multi-configuration training modes. Results are saved as
plots and loss histories.

Usage examples:

./nn_sandbox.py --help
./nn_sandbox.py
./nn_sandbox.py --test-function griewank --epochs 200 --learning-rate 0.001
./nn_sandbox.py --hidden-sizes 16 8 --batch-size 20 --epochs 250
./nn_sandbox.py --test-function branin --hidden-sizes 64 32 16 --n-test 500
./nn_sandbox.py --test-function ackley --activation tanh --epochs 300
./nn_sandbox.py --test-function parabola --activation sigmoid --learning-rate 0.0001
./nn_sandbox.py --multi-train --multi-hidden-sizes 8 16 --multi-learning-rates 0.001 0.0001
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from surmod import neural_network as nn
from surmod.test_functions import load_test_function
from surmod.utils import log_results


def parse_arguments():
    """Get command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train neural network surrogate models on synthetic test functions.",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    experiment = parser.add_argument_group("experiment options")
    nn_options = parser.add_argument_group("neural network options")

    experiment.add_argument(
        "-f",
        "--test-function",
        type=str,
        default="ackley",
        help="Test function to use. Supported: parabola, ackley, branin, holder_table, griewank, six_hump_camel.",
    )

    experiment.add_argument(
        "--n-train",
        type=int,
        default=90,
        help="Number of training points.",
    )

    experiment.add_argument(
        "--n-test",
        type=int,
        default=10,
        help="Number of testing points.",
    )

    nn_options.add_argument(
        "-hs",
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[12, 12],
        help="Sizes of hidden layers.",
    )

    nn_options.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )

    nn_options.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=5,
        help="Batch size for training.",
    )

    nn_options.add_argument(
        "-l",
        "--learning-rate",
        type=float,
        default=0.00001,
        help="Learning rate for SGD optimization.",
    )

    nn_options.add_argument(
        "-a",
        "--activation",
        type=str,
        choices=["relu", "sigmoid", "tanh"],
        default="relu",
        help="Activation function to use between layers.",
    )

    nn_options.add_argument(
        "-nx",
        "--normalize-x",
        action="store_true",
        default=False,
        help="Whether or not to normalize the input values by removing the "
        "mean and scaling to unit-variance per dimension.",
    )

    nn_options.add_argument(
        "-sx",
        "--scale-x",
        action="store_true",
        default=False,
        help="Whether or not to scale the input values to [0,1] using min-max "
        "scaling per dimension.",
    )

    nn_options.add_argument(
        "-ny",
        "--normalize-y",
        action="store_true",
        default=False,
        help="Whether or not to normalize the output values by removing the "
        "mean and scaling to unit-variance.",
    )

    nn_options.add_argument(
        "-sy",
        "--scale-y",
        action="store_true",
        default=False,
        help="Whether or not to scale the output values to [0,1] using min-max"
        " scaling.",
    )

    nn_options.add_argument(
        "-mt",
        "--multi-train",
        action="store_true",
        default=False,
        help="If set, trains across multiple hidden dims and learning rates.",
    )

    nn_options.add_argument(
        "-mh",
        "--multi-hidden-sizes",
        type=int,
        nargs="+",
        default=[8, 12, 16],
        help="List of sizes to apply to both (two) hidden layers.",
    )

    nn_options.add_argument(
        "-ml",
        "--multi-learning-rates",
        type=float,
        nargs="+",
        default=[1e-3, 1e-4, 1e-5],
        help="List of learning rates to try.",
    )

    nn_options.add_argument(
        "-sp",
        "--surface-plot",
        action="store_true",
        default=False,
        help="If set, generates a surface plot of surrogate and test function "
        "Only works when -mt is NOT flagged.",
    )

    nn_options.add_argument(
        "-vp",
        "--verbose-plot",
        action="store_true",
        default=False,
        help="If set, includes (hyper)parameter values in loss plot title "
        "Only works when -mt is NOT flagged.",
    )

    args = parser.parse_args()

    return args


def plot_surface_3d(
    synthetic_function: SyntheticTestFunction,
    model,
    title: str,
    plots_dir: Path,
    resolution: int = 50,
    angle: tuple[float, float] = (30, 120),
    input_scaler=None,
    output_scaler=None,
):
    """
    Plot the true surface of a synthetic function and model predictions in 3D.

    This function generates a grid of input points within the bounds of the
    synthetic function, computes the true values and model predictions, and
    visualizes both surfaces in a 3D plot.

    Args:
        synthetic_function: Callable synthetic test function. It must expose
            bounds through ``_bounds`` and accept a ``torch.Tensor`` input.
        model: The PyTorch neural net model to make predictions from.
        title: Title for the plot and output file, usually the test-function
            name.
        plots_dir: Directory where the plot is saved.
        resolution: Number of points per dimension in the surface grid.
        angle: ``(elevation, azimuth)`` viewing angles for the 3D plot.
        input_scaler: Optional scaler with a ``transform`` method applied to
            the input grid before prediction.
        output_scaler: Optional scaler with an ``inverse_transform`` method
            applied to model predictions.
    """
    # Generate a grid of points within the bounds of the test function
    bounds_low = [b[0] for b in synthetic_function._bounds]
    bounds_high = [b[1] for b in synthetic_function._bounds]
    margin = 1e-6  # Small margin to avoid floating-point precision issues
    x1 = np.linspace(bounds_low[0] + margin, bounds_high[0] - margin, resolution)
    x2 = np.linspace(bounds_low[1] + margin, bounds_high[1] - margin, resolution)
    X1, X2 = np.meshgrid(x1, x2)
    grid_points = np.stack([X1.ravel(), X2.ravel()], axis=1)

    grid_points_tensor = torch.Tensor(grid_points).float()

    # Compute true surface values
    true_surface = (
        synthetic_function(grid_points_tensor)
        .detach()
        .numpy()
        .reshape(resolution, resolution)
    )

    if input_scaler is not None:
        # Convert tensor to numpy if needed
        grid_points_np = grid_points_tensor.numpy()
        # Apply the scaler
        grid_points_scaled_np = input_scaler.transform(grid_points_np)
        # Convert back to tensor if you need to use it as a tensor
        grid_points_tensor = torch.from_numpy(grid_points_scaled_np).float()

    # Compute model predictions
    with torch.no_grad():
        predicted_surface = (
            model(grid_points_tensor).detach().numpy().reshape(resolution, resolution)
        )

    if output_scaler is not None:
        # Inverse transform to get predictions back to original scale
        predicted_surface = output_scaler.inverse_transform(predicted_surface)
        predicted_surface = predicted_surface.reshape(resolution, resolution)

    # Create a new figure and 3D axis
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Plot true surface in 3D
    ax.plot_surface(  # type: ignore
        X1,
        X2,
        true_surface,
        cmap="viridis",
        alpha=0.7,
        edgecolor="none",
    )
    # Overlay model predictions in 3D
    ax.plot_surface(  # type: ignore
        X1,
        X2,
        predicted_surface,
        cmap="coolwarm",
        alpha=0.5,
        edgecolor="none",
    )

    ax.set_title(f"{title} - True & Model Surfaces")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Value")  # type: ignore

    # Set the viewing angle
    ax.view_init(angle[0], angle[1])  # type: ignore

    viridis_cmap = matplotlib.colormaps["viridis"]
    coolwarm_cmap = matplotlib.colormaps["coolwarm"]

    true_patch = mpatches.Patch(
        color=viridis_cmap(0.6), label="True Surface", alpha=0.7
    )
    model_patch = mpatches.Patch(
        color=coolwarm_cmap(0.6),
        label="Model Prediction",
        alpha=0.5,
    )

    ax.legend(handles=[true_patch, model_patch], loc="upper left")

    # Create plots directory if it doesn't exist and save plot
    plots_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    filepath = plots_dir / f"surface_plot_{title}_{timestamp}.png"
    plt.savefig(filepath)
    print(f"Figure saved to {filepath}")


def main():
    """
    Train neural-network surrogates on synthetic test functions and save plots.
    """
    args = parse_arguments()
    test_function = args.test_function
    normalize_x = args.normalize_x
    scale_x = args.scale_x
    normalize_y = args.normalize_y
    scale_y = args.scale_y
    seed = args.seed
    epochs = args.epochs
    batch_size = args.batch_size
    hidden_sizes = args.hidden_sizes
    learning_rate = args.learning_rate
    activation = args.activation
    multi_train = args.multi_train
    multi_hidden_sizes = args.multi_hidden_sizes
    multi_learning_rates = args.multi_learning_rates
    surface_plot = args.surface_plot
    verbose_plot = args.verbose_plot
    n_train = args.n_train
    n_test = args.n_test

    # Weight initialization (default PyTorch)
    initialize_weights_normal = False

    # Set output directory relative to this script
    script_dir = Path(__file__).parent
    plots_dir = script_dir / "plots"

    # Generate random data from test function
    synthetic_function = load_test_function(test_function)
    input_size = synthetic_function.dim
    torch.manual_seed(seed)

    bounds_low = torch.tensor([b[0] for b in synthetic_function._bounds])
    bounds_high = torch.tensor([b[1] for b in synthetic_function._bounds])

    # Generate data based on command line arguments
    n_total = n_train + n_test
    x_data = torch.rand(n_total, input_size) * (bounds_high - bounds_low) + bounds_low
    y_data = synthetic_function(x_data)

    # Split data into training and testing sets based on n_train parameter
    x_train = x_data[:n_train]
    x_test = x_data[n_train:]
    y_train = y_data[:n_train]
    y_test = y_data[n_train:]

    scaler_x_train = None
    scaler_y_train = None

    if normalize_x and scale_x:
        raise ValueError("Choose either normalize_x or scale_x, not both.")

    if normalize_x or scale_x:
        # Create the scaler and fit it on training data
        if normalize_x:
            print(
                "Input data is being normalized to have mean 0, variance 1, in "
                "each dimension based on training data.\n"
            )
            scaler_x_train = StandardScaler()

        if scale_x:
            print(
                "Input data is being scaled using min max scaling in each "
                "dimension based on training data.\n"
            )
            scaler_x_train = MinMaxScaler()

        scaler_x_train.fit(x_train)  # type: ignore

        # Transform both train and test sets
        x_train = scaler_x_train.transform(x_train)  # type: ignore
        x_test = scaler_x_train.transform(x_test)  # type: ignore

        # Convert back to torch tensors
        x_train = torch.as_tensor(x_train, dtype=torch.float32)
        x_test = torch.as_tensor(x_test, dtype=torch.float32)

    if normalize_y and scale_y:
        raise ValueError("Choose either normalize_y or scale_y, not both.")

    if normalize_y or scale_y:
        # Note: if y is normalized or scaled, all losses and metrics during
        # training and testing are computed in this transformed space, not in
        # the original output units of the test function.

        # Create the scaler and fit it on training data
        if normalize_y:
            print(
                "Output data is being normalized to have mean 0, variance 1 "
                "based on training data.\n"
                "Note: training and testing losses will be in normalized units, "
                "not in the original test function units.\n"
            )
            scaler_y_train = StandardScaler()

        if scale_y:
            print(
                "Output data is being scaled using min-max scaling based on "
                "training data.\n"
                "Note: training and testing losses will be in scaled units, "
                "not in the original test function units.\n"
            )
            scaler_y_train = MinMaxScaler()

        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        scaler_y_train.fit(y_train)  # type: ignore

        # Transform both train and test sets
        y_train = scaler_y_train.transform(y_train)  # type: ignore
        y_test = scaler_y_train.transform(y_test)  # type: ignore

        # Convert back to torch tensors
        y_train = torch.as_tensor(y_train, dtype=torch.float32)
        y_test = torch.as_tensor(y_test, dtype=torch.float32)

    # Do multiple train/test runs with various learning rates & hidden layers
    #   size and plot loss over epochs results
    if multi_train:
        # Create subplots for each learning rate and hidden layers size
        fig, axs = plt.subplots(
            len(multi_hidden_sizes),
            len(multi_learning_rates),
            figsize=(15, 15),
        )
        fig.suptitle(f"Training and Testing Losses - {test_function}", fontsize=16)

        # Train and test FFNN
        results: nn.LossSweepResults = {}

        for hid_sz in multi_hidden_sizes:
            for lr in multi_learning_rates:
                hidden_sizes = [hid_sz, hid_sz]
                model, train_losses, test_losses = nn.train(
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    hidden_sizes,
                    epochs,
                    lr,
                    batch_size,
                    seed,
                    initialize_weights_normal,
                    activation,
                )

                # Store losses by the actual hyperparameter values.
                results[(hid_sz, lr)] = nn.TrainingRunHistory(
                    train_losses=train_losses,
                    test_losses=test_losses,
                )

        print("All training finished!\n")

        # Plot train and test loss over epochs
        nn.plot_losses_multiplot(
            results,
            multi_learning_rates,
            multi_hidden_sizes,
            axs,
            test_function,
            plots_dir,
        )

    # Default: Do one train/test run and plot loss over epochs results
    else:
        # Train and test FFNN
        start_time = time.time()
        model, train_losses, test_losses = nn.train(
            x_train,
            y_train,
            x_test,
            y_test,
            hidden_sizes,
            epochs,
            learning_rate,
            batch_size,
            seed,
            initialize_weights_normal,
            activation,
        )
        elapsed_time = time.time() - start_time

        # Log results
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        log_lines = [
            f"Run timestamp (%m%d_%H%M%S): {timestamp}",
            f"Test Function: {test_function}",
            f"Number of training points: {n_train}",
            f"Number of testing points: {n_test}",
            f"Hidden layer sizes: {hidden_sizes}",
            f"Activation function: {activation}",
            f"Learning rate: {learning_rate}",
            f"Batch size: {batch_size}",
            f"Epochs: {epochs}",
            f"Normalize x: {normalize_x}",
            f"Scale x: {scale_x}",
            f"Normalize y: {normalize_y}",
            f"Scale y: {scale_y}",
            f"Final train loss: {train_losses[-1]:.5e}",
            f"Final test loss: {test_losses[-1]:.5e}",
            f"Elapsed time for training NN: {elapsed_time:.3f} seconds\n",
        ]
        log_message = "\n".join(log_lines)
        print(log_message)

        results_dir = Path(__file__).parent / "results"
        log_results(
            log_message,
            path_to_log=results_dir / f"{test_function}_nn.txt",
        )

        if verbose_plot:
            # Plot train and test loss over epochs with (hyper)parameters
            #   included
            nn.plot_losses_verbose(
                train_losses,
                test_losses,
                learning_rate,
                batch_size,
                hidden_sizes,
                normalize_x,
                scale_x,
                normalize_y,
                scale_y,
                n_train,
                n_test,
                test_function,
                plots_dir,
            )

        else:
            # Plot train and test loss over epochs
            nn.plot_losses(train_losses, test_losses, test_function, plots_dir)

        if surface_plot:
            plot_surface_3d(
                synthetic_function,
                model,
                title=test_function,
                plots_dir=plots_dir,
                resolution=50,
                angle=(30, 120),
                input_scaler=scaler_x_train,
                output_scaler=scaler_y_train,
            )

        # Get neural network predictions
        model.eval()
        with torch.no_grad():
            predictions = model(x_test)

        # Back-transform predictions and test outputs for plotting (if scaling was applied)
        y_test_plot = y_test
        predictions_plot = predictions
        if scaler_y_train is not None:
            # Convert to numpy and inverse transform
            y_test_np = y_test.numpy().reshape(-1, 1)
            predictions_np = predictions.numpy().reshape(-1, 1)
            y_test_plot = torch.tensor(scaler_y_train.inverse_transform(y_test_np))
            predictions_plot = torch.tensor(
                scaler_y_train.inverse_transform(predictions_np)
            )

        nn.plot_predictions(
            y_test_plot,
            predictions_plot,
            test_losses[-1],
            test_function,
            plots_dir,
        )


if __name__ == "__main__":
    main()
