"""
Functions for neural network surrogates.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class NeuralNet(nn.Module):
    """
    A customizable feedforward neural network for regression tasks.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        initialize_weights_normal: bool,
    ):
        """
        Initialize the NeuralNet.

        Args:
            input_size (int): Number of input features.
            hidden_sizes (list of int): Sizes of hidden layers.
            output_size (int): Number of output features.
            initialize_weights_normal (bool): Whether to initialize weights
            with a normal distribution.
        """
        super(NeuralNet, self).__init__()
        self.layers = nn.ModuleList()

        # Create the first hidden layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))

        # Create hidden layers based on the hidden_sizes list
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))

        # Add the final output layer
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # Initialize weights
        if initialize_weights_normal:
            self._normal_weight_init()

    def _normal_weight_init(self) -> None:
        """
        Initialize all weights of the neural network with normal distribution
        (mean=0.0, std=0.1) and biases to zero.
        """
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Initialize weights with normal distribution
                nn.init.normal_(layer.weight, mean=0.0, std=0.1)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Initialize biases to zero

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        for layer in self.layers:
            x = layer(x)
        return x


def train(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    hidden_sizes: List[int],
    n_epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    initialize_weights_normal: bool,
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    Train a feedforward neural network and evaluate its performance.

    Args:
        x_train (torch.Tensor): Training input features of shape (n_samples, n_features).
        y_train (torch.Tensor): Training target values of shape (n_samples,) or (n_samples, 1).
        x_test (torch.Tensor): Test input features of shape (n_test_samples, n_features).
        y_test (torch.Tensor): Test target values of shape (n_test_samples,) or (n_test_samples, 1).
        hidden_sizes (List[int]): List specifying the number of units in each hidden layer.
        n_epochs (int): Number of epochs to train the network.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Number of samples per training batch.
        seed (int): Random seed for reproducibility.
        initialize_weights_normal (bool): If True, initialize weights with a normal distribution.
    Returns:
        Tuple[nn.Module, List[float], List[float]]: Trained neural network model, list of training losses per epoch, and list of test losses per epoch.
    """
    # Specify fixed output and input sizes
    input_size = x_train.shape[1]
    output_size = 1

    accumulation_steps = 4

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the neural network
    model = NeuralNet(input_size, hidden_sizes, output_size, initialize_weights_normal)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Lists to store losses
    train_losses = []
    test_losses = []

    # Set a random number generator seed for reproducibility
    torch.manual_seed(seed)

    # Training loop
    for epoch in range(n_epochs):
        model.train()  # Set the model to training mode
        epoch_loss = 0.0

        for i, (inputs, targets) in enumerate(train_loader):
            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, targets.view(-1, 1))

            # Backward pass
            loss.backward()

            # Accumulate gradients and update parameters only after
            #   accumulation_steps batches
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()  # Update model parameters
                optimizer.zero_grad()  # Reset gradients for the next cycle

            epoch_loss += loss.item()

        # Average loss for the epoch
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluate on the test set
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            test_outputs = model(x_test)
            test_loss = criterion(test_outputs, y_test.view(-1, 1))
            test_losses.append(test_loss.item())

        # Print the loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{n_epochs}], "
                f"Training Loss (MSE): {avg_train_loss:.5f}, "
                f"Testing Loss (MSE): {test_loss.item():.5f}"
            )

    print("Training finished!\n")

    return model, train_losses, test_losses


def plot_losses(
    train_losses: List[float],
    test_losses: List[float],
    dataset: str,
    plots_dir: Path,
) -> None:
    """
    Plot and save the training and testing loss curves across epochs.

    Args:
        train_losses (List[float]): List of training loss values (MSE) for each
            epoch.
        test_losses (List[float]): List of testing loss values (MSE) for each
            epoch.
        dataset (str): Name of the dataset. Used in the plot title and filename.
        plots_dir (Path): Directory where plots will be saved.
    """
    plots_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    filepath = plots_dir / f"loss_vs_epoch_{dataset}_{timestamp}.png"

    final_test_rmse = np.sqrt(test_losses[-1])

    n_epochs = len(train_losses)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_epochs + 1), train_losses, label="Training Loss (MSE)")
    plt.plot(range(1, n_epochs + 1), test_losses, label="Testing Loss (MSE)")
    plt.yscale("log")
    plt.title(
        f"Training and Testing Losses - {dataset}\n"
        f"Final Test Loss (RMSE): {final_test_rmse:.5f}"
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss (log scale)")
    plt.legend()
    plt.grid()
    plt.savefig(filepath)
    print(f"Figure saved to {filepath}")


def plot_losses_verbose(
    train_losses: List[float],
    test_losses: List[float],
    learning_rate: float,
    batch_size: int,
    hidden_sizes: List[int],
    normalize_x: bool,
    scale_x: bool,
    normalize_y: bool,
    scale_y: bool,
    train_data_size: int,
    test_data_size: int,
    dataset: str,
    plots_dir: Path,
) -> None:
    """
    Plot and save training and testing loss curves across epochs, with
    hyperparameter values in the plot title.

    Args:
        train_losses (List[float]): List of training loss values (MSE) for each
            epoch.
        test_losses (List[float]): List of testing loss values (MSE) for each
            epoch.
        learning_rate (float): Learning rate used during training.
        batch_size (int): Batch size used during training.
        hidden_sizes (List[int]): List of hidden layer sizes in the model.
        normalize_x (bool): Whether input features (x) were normalized.
        scale_x (bool): Whether input features (x) were scaled.
        normalize_y (bool): Whether target values (y) were normalized.
        scale_y (bool): Whether target values (y) were scaled.
        train_data_size (int): Number of samples in the training set.
        test_data_size (int): Number of samples in the testing set.
        dataset (str): Name of the dataset. Used in the plot title and filename.
        plots_dir (Path): Directory where plots will be saved.
    """
    plots_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    filepath = plots_dir / f"loss_vs_epoch_{dataset}_verbose_{timestamp}.png"

    final_test_rmse = np.sqrt(test_losses[-1])

    n_epochs = len(train_losses)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_epochs + 1), train_losses, label="Training Loss (MSE)")
    plt.plot(range(1, n_epochs + 1), test_losses, label="Testing Loss (MSE)")
    plt.yscale("log")
    title = (
        f"{dataset} \n "
        f"Train size: {train_data_size} | Test size: {test_data_size} | "
        f"LR: {learning_rate:.2e} | "
        f"Batch: {batch_size} | "
        f"HS: {hidden_sizes} | "
    )
    if normalize_x:
        title += f"Norm. x: {normalize_x} | "
    if scale_x:
        title += f"Scale. x: {scale_x} | "
    if normalize_y:
        title += f"Norm. y: {normalize_y} | "
    if scale_y:
        title += f"Scale. y: {scale_y} | "
    title += f"Final Test Loss (RMSE): {final_test_rmse:.5f}"
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss (log scale)")
    plt.legend()
    plt.grid()
    plt.savefig(filepath)
    print(f"Figure saved to {filepath}")


def plot_losses_multiplot(
    train_losses_grid: List[List[List[float]]],
    test_losses_grid: List[List[List[float]]],
    learning_rates: List[float],
    hid_dims: List[int],
    axs: Sequence[Sequence[matplotlib.axes.Axes]],
    dataset: str,
    plots_dir: Path,
) -> None:
    """
    Plots training and test losses for multiple runs on a grid of subplots.

    Each subplot corresponds to a specific combination of hidden dimension and
    learning rate, displaying the training and test loss curves over epochs.
    The final test loss (RMSE) is shown in each subplot title. The resulting
    multiplot figure is saved to the specified plots directory with a filename
    that includes the dataset name and a timestamp.

    Args:
        train_losses_grid (Sequence[Sequence[List[float]]]):
            2D grid where each element is a list of training losses per epoch
            for a specific (hidden_dim, learning_rate) pair.
        test_losses_grid (Sequence[Sequence[List[float]]]):
            2D grid where each element is a list of test losses per epoch for a
            specific (hidden_dim, learning_rate) pair.
        learning_rates (List[float]):
            List of learning rates corresponding to the columns of the subplot
            grid.
        hid_dims (List[int]):
            List of hidden dimensions corresponding to the rows of the subplot
            grid.
        axs (Sequence[Sequence[matplotlib.axes.Axes]]):
            2D grid of matplotlib Axes objects for plotting.
        dataset (str):
            Name of the dataset, used in the saved filename.
        plots_dir (Path):
            Directory where plots will be saved.
    """
    for i, hid_sz in enumerate(hid_dims):
        for j, lr in enumerate(learning_rates):
            ax = axs[i][j]
            train_losses = train_losses_grid[i][j]
            test_losses = test_losses_grid[i][j]
            n_epochs = len(train_losses)

            # Calculate final test RMSE
            final_test_rmse = np.sqrt(test_losses[-1])

            ax.plot(range(1, n_epochs + 1), train_losses, label="Train Loss")
            ax.plot(range(1, n_epochs + 1), test_losses, label="Test Loss")
            ax.set_yscale("log")
            ax.set_title(
                f"hid_dim={hid_sz}, lr={lr}\nFinal Test Loss (RMSE): "
                f"{final_test_rmse:.5f}"
            )
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss (log scale)")
            ax.legend()
            ax.grid()

    # Save the multiplot figure
    plots_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    filepath = plots_dir / f"multi_loss_vs_epoch_{dataset}_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(filepath)
    print(f"Figure saved to {filepath}")


def plot_predictions(
    y_test: torch.Tensor,
    predictions: torch.Tensor,
    final_test_mse: float,
    dataset: str,
    plots_dir: Path,
) -> None:
    """
    Plots the actual test values against the predicted values.

    This function creates a parity plot comparing the true test values to the
    model's predictions. A reference line for perfect prediction is included.
    The final test loss (RMSE) is displayed in the plot title. The plot is saved
    in the specified plots directory, with a filename that includes the dataset
    name and a timestamp.

    Args:
        y_test (torch.Tensor):
            The true target values for the test set.
        predictions (torch.Tensor):
            The predicted values from the model for the test set.
        final_test_mse (float):
            The final mean squared error on the test set.
        dataset (str):
            Name of the dataset, used in the filename.
        plots_dir (Path):
            Directory where plots will be saved.
    """
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test.numpy(), predictions.numpy(), alpha=0.5)
    plt.plot(
        [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--"
    )  # Line for perfect prediction

    final_test_rmse = np.sqrt(final_test_mse)

    plt.title(
        f"Test Output vs Predicted Output | Final Test Loss (RMSE): "
        f"{final_test_rmse:.5f}"
    )
    plt.xlabel("Test Output")
    plt.ylabel("Predicted Output")
    plt.grid()

    # Set equal limits for x and y axes
    y_test_np = y_test.cpu().numpy()
    predictions_np = predictions.cpu().numpy()
    limits = [
        min(y_test_np.min(), predictions_np.min()),
        max(y_test_np.max(), predictions_np.max()),
    ]
    plt.xlim(limits)
    plt.ylim(limits)
    plt.axis("square")

    plots_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    filepath = plots_dir / f"prediction_vs_test_{dataset}_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(filepath)
    print(f"Figure saved to {filepath}")
