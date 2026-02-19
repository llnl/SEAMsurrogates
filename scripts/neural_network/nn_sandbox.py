#!/usr/bin/env python3

"""
This script trains a feedforward neural network (FFNN) surrogate model on
synthetic test functions (Ackley, SixHumpCamel, or Griewank). It provides
options for customizing the network architecture, learning rate, batch size,
and for running multiple training configurations. Results are plotted and saved
within working directory.

Usage:

# Make script executable
chmod +x ./nn_sandbox.py

# See help.
./nn_sandbox.py -h

# Train a NN on the Ackley function with default settings.
./nn_sandbox.py

# Train a NN on the Griewank function with 200 epochs and a custom learning rate.
./nn_sandbox.py --objective_function=Griewank -n 200 -l 0.001

# Train a NN with custom hidden layer sizes and batch size.
./nn_sandbox.py --hidden_sizes 16 8 -b 10

# Train with custom number of training and test points
./nn_sandbox.py --n_train 200 --n_test 50

# Train and compare multiple NNs with different hidden layer sizes and learning rates.
./nn_sandbox.py --multi_train --multi_hidden_sizes 8 16 --multi_learning_rates 0.001 0.0001
"""

import argparse
from datetime import datetime
import os
from typing import Tuple

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from surmod import neural_network as nn


def parse_arguments():
    """Get command line arguments."""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="A script to train NN surrogate models on synthetic test "
        "functions.",
    )

    parser.add_argument(
        "-f",
        "--objective_function",
        type=str,
        choices=["Ackley", "SixHumpCamel", "Griewank"],
        default="Ackley",
        help="Choose objective function (Ackley, SixHumpCamel, or Griewank).",
    )

    parser.add_argument(
        "-nx",
        "--normalize_x",
        action="store_true",
        default=False,
        help="Whether or not to normalize the input values by removing the "
        "mean and scaling to unit-variance per dimension.",
    )

    parser.add_argument(
        "-sx",
        "--scale_x",
        action="store_true",
        default=False,
        help="Whether or not to scale the input values to [0,1] using min-max "
        "scaling per dimension.",
    )

    parser.add_argument(
        "-ny",
        "--normalize_y",
        action="store_true",
        default=False,
        help="Whether or not to normalize the output values by removing the "
        "mean and scaling to unit-variance.",
    )

    parser.add_argument(
        "-sy",
        "--scale_y",
        action="store_true",
        default=False,
        help="Whether or not to scale the output values to [0,1] using min-max"
        " scaling.",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=1,
        help="Random number generator seed.",
    )

    parser.add_argument(
        "-n",
        "--num_epochs",
        type=int,
        default=100,
        help="Number of epochs for training.",
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=5,
        help="Batch size for training.",
    )

    parser.add_argument(
        "--n_train",
        type=int,
        default=90,
        help="Number of training points.",
    )

    parser.add_argument(
        "--n_test",
        type=int,
        default=10,
        help="Number of testing points.",
    )

    parser.add_argument(
        "-hs",
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=[12, 12],
        help="Sizes of hidden layers.",
    )

    parser.add_argument(
        "-l",
        "--learning_rate",
        type=float,
        default=0.00001,
        help="Learning rate for SGD optimization.",
    )

    parser.add_argument(
        "-mt",
        "--multi_train",
        action="store_true",
        default=False,
        help="If set, trains across multiple hidden dims and learning rates.",
    )

    parser.add_argument(
        "-mh",
        "--multi_hidden_sizes",
        type=int,
        nargs="+",
        default=[8, 12, 16],
        help="List of sizes to apply to both (two) hidden layers. Must have -mt flagged.",
    )

    parser.add_argument(
        "-ml",
        "--multi_learning_rates",
        type=float,
        nargs="+",
        default=[1e-3, 1e-4, 1e-5],
        help="List of learning rates to try. Must have -mt flagged.",
    )

    parser.add_argument(
        "-sp",
        "--surface_plot",
        action="store_true",
        default=False,
        help="If set, generates a surface plot of surrogate and test function "
        "Only works when -mt is NOT flagged.",
    )

    parser.add_argument(
        "-vp",
        "--verbose_plot",
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
    resolution: int = 50,
    angle: Tuple[float, float] = (30, 120),
    input_scaler=None,
    output_scaler=None,
):
    """
    Plots the true surface of a synthetic function and the
    predictions of a model in 3D.

    This function generates a grid of input points within the bounds
    of the synthetic function, computes the true values and model
    predictions (optionally applying input/output scalers), and
    visualizes both surfaces in a 3D plot. The plot is saved to the
    'plots' directory.

    Args:
        synthetic_function (Any): A callable object representing the
            test function. Must have a 'bounds'
            attribute(tuple of (low, high) for each input dimension)
            and be callable on a torch.Tensor.
        model: The PyTorch neural net model to make predictions from.
        title (str): Title for the plot and output file, usually
            the objective data/function name.
        resolution (int, optional): Number of points per dimension
            for the surface grid. Default is 50.
        angle (Tuple[float, float], optional):
            The (elevation, azimuth) viewing angles for the 3D plot.
            Default is (30, 120).
        input_scaler: Optional sklearn.preprocessing scaler with a
            .transform method to apply to input grid points.
        output_scaler: Optional sklearn.preprocessing scaler with an
            .inverse_transform method to apply to model predictions.
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

    # Create proxy artists for the legend
    # Get the colormap object
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
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    filename = f"surface_plot_{title}_{timestamp}.png"
    filepath = os.path.join(plots_dir, filename)
    plt.savefig(filepath)
    print(f"Figure saved to {filepath}")


def main():
    """
    Parses command-line arguments, generates synthetic data, trains a
    neural network surrogate model, and plots training/testing loss curves.
    Supports single or multiple training runs with varying hyperparameters.
    """

    # Parse command line arguments
    args = parse_arguments()
    objective_function = args.objective_function
    normalize_x = args.normalize_x
    scale_x = args.scale_x
    normalize_y = args.normalize_y
    scale_y = args.scale_y
    seed = args.seed
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    hidden_sizes = args.hidden_sizes
    learning_rate = args.learning_rate
    multi_train = args.multi_train
    multi_hidden_sizes = args.multi_hidden_sizes
    multi_learning_rates = args.multi_learning_rates
    surface_plot = args.surface_plot
    verbose_plot = args.verbose_plot
    n_train = args.n_train
    n_test = args.n_test

    # Weight initialization (default PyTorch)
    initialize_weights_normal = False

    # Generate random data from test function
    synthetic_function = nn.load_test_function(objective_function)
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
        x_train = torch.from_numpy(x_train).float()
        x_test = torch.from_numpy(x_test).float()

    if normalize_y and scale_y:
        raise ValueError("Choose either normalize_y or scale_y, not both.")

    if normalize_y or scale_y:
        # Note: if y is normalized or scaled, all losses and metrics during
        # training and testing are computed in this transformed space, not in
        # the original output units of the objective function.

        # Create the scaler and fit it on training data
        if normalize_y:
            print(
                "Output data is being normalized to have mean 0, variance 1 "
                "based on training data.\n"
                "Note: training and testing losses will be in normalized units, "
                "not in the original objective function units.\n"
            )
            scaler_y_train = StandardScaler()

        if scale_y:
            print(
                "Output data is being scaled using min-max scaling based on "
                "training data.\n"
                "Note: training and testing losses will be in scaled units, "
                "not in the original objective function units.\n"
            )
            scaler_y_train = MinMaxScaler()

        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        scaler_y_train.fit(y_train)  # type: ignore

        # Transform both train and test sets
        y_train = scaler_y_train.transform(y_train)  # type: ignore
        y_test = scaler_y_train.transform(y_test)  # type: ignore

        # Convert back to torch tensors
        y_train = torch.from_numpy(y_train).float()
        y_test = torch.from_numpy(y_test).float()

    # Do multiple train/test runs with various learning rates & hidden layers
    #   size and plot loss over epochs results
    if multi_train:
        # Create subplots for each learning rate and hidden layers size
        fig, axs = plt.subplots(
            len(multi_hidden_sizes),
            len(multi_learning_rates),
            figsize=(15, 15),
        )
        fig.suptitle(f"Training and Testing Losses - {objective_function}", fontsize=16)

        # Train and test FFNN
        n = len(multi_hidden_sizes)
        m = len(multi_learning_rates)
        train_losses_grid = [[[] for _ in range(m)] for _ in range(n)]
        test_losses_grid = [[[] for _ in range(m)] for _ in range(n)]

        for i, hid_sz in enumerate(multi_hidden_sizes):
            for j, lr in enumerate(multi_learning_rates):
                hidden_sizes = [hid_sz, hid_sz]
                model, train_losses, test_losses = nn.train_neural_net(
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    hidden_sizes,
                    num_epochs,
                    lr,
                    batch_size,
                    seed,
                    initialize_weights_normal,
                )

                # Store train and test loss results over epochs
                train_losses_grid[i][j] = train_losses
                test_losses_grid[i][j] = test_losses

        print("All training finished!\n")

        # Plot train and test loss over epochs
        nn.plot_losses_multiplot(
            train_losses_grid,
            test_losses_grid,
            multi_learning_rates,
            multi_hidden_sizes,
            axs,
            objective_function,
        )

    # Default: Do one train/test run and plot loss over epochs results
    else:
        # Train and test FFNN
        model, train_losses, test_losses = nn.train_neural_net(
            x_train,
            y_train,
            x_test,
            y_test,
            hidden_sizes,
            num_epochs,
            learning_rate,
            batch_size,
            seed,
            initialize_weights_normal,
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
                objective_function,
            )

        else:
            # Plot train and test loss over epochs
            nn.plot_losses(train_losses, test_losses, objective_function)

        if surface_plot:
            plot_surface_3d(
                synthetic_function,
                model,
                title=objective_function,
                resolution=50,
                angle=(30, 120),
                input_scaler=scaler_x_train,
                output_scaler=scaler_y_train,
            )


if __name__ == "__main__":
    main()
