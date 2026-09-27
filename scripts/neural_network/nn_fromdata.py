#!/usr/bin/env python3

"""
Train neural network surrogate models on datasets from data/.

This script trains feedforward neural networks on real datasets, with options
for customizing network architecture, learning rate, batch size, and data scaling.
Results are saved as plots and loss histories.

Usage examples:

./nn_fromdata.py --help
./nn_fromdata.py
./nn_fromdata.py -d JAG --hidden-sizes 10 20
./nn_fromdata.py -d JAG --hidden-sizes 15 15 --batch-size 20 --epochs 400
./nn_fromdata.py -d borehole --hidden-sizes 60 60 --batch-size 40 --epochs 600 --learning-rate 0.02
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from surmod import data_processing
from surmod import neural_network as nn
from surmod.utils import log_results


def parse_arguments() -> argparse.Namespace:
    """Get command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train neural network surrogate models on datasets from data/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    data_options = parser.add_argument_group("data options")
    nn_options = parser.add_argument_group("neural network options")

    data_options.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=list(data_processing.DATASET_CONFIG.keys()),
        default="JAG",
        help="Which dataset to use (default: JAG).",
    )

    data_options.add_argument(
        "-tr",
        "--n-train",
        type=int,
        default=400,
        help="Number of train samples (default: 400).",
    )

    data_options.add_argument(
        "-te",
        "--n-test",
        type=int,
        default=100,
        help="Number of test samples (default: 100).",
    )

    data_options.add_argument(
        "--LHD",
        action="store_true",
        help="Use an LHD design.",
    )

    nn_options.add_argument(
        "-hs",
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[15, 15],
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
        default=0.001,
        help="Learning rate for SGD optimization.",
    )

    nn_options.add_argument(
        "-vp",
        "--verbose-plot",
        action="store_true",
        default=False,
        help="If set, includes (hyper)parameter values in loss plot title.",
    )

    args = parser.parse_args()

    return args


def main() -> None:
    # Parse command line arguments
    args = parse_arguments()
    dataset = args.dataset
    n_train = args.n_train
    n_test = args.n_test
    seed = args.seed
    LHD = args.LHD
    epochs = args.epochs
    batch_size = args.batch_size
    hidden_sizes = args.hidden_sizes
    learning_rate = args.learning_rate
    verbose_plot = args.verbose_plot

    # Set output directory relative to this script
    script_dir = Path(__file__).parent
    plots_dir = script_dir / "plots"

    # Check data availability
    n_samples = n_test + n_train
    if n_samples > 10000:
        raise ValueError(
            f"Requested samples ({n_samples}) exceed existing dataset(s) size "
            "limit (10000)."
        )

    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Weight initialization (normal with mean = 0, sd = 0.1)
    initialize_weights_normal = True

    # Load data into data frame and split into train and test sets
    df = data_processing.load_data(
        dataset=dataset, n_samples=n_samples, random=True, seed=seed
    )
    print("Data subset shape:", df.shape)
    x_train, x_test, y_train, y_test = data_processing.split_data(
        df, LHD=LHD, n_train=n_train, seed=seed
    )

    # Normalize data (critical for numerical stability with different feature scales)
    x_train, x_test, y_train, y_test = data_processing.normalize_data(
        x_train, x_test, y_train, y_test
    )
    print("Data normalized (zero mean, unit variance)\n")

    # Convert training and test data to float32 tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Train the neural net
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
    )
    elapsed_time = time.time() - start_time

    # Log results
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    log_lines = [
        f"Run timestamp (%m%d_%H%M%S): {timestamp}",
        f"Dataset: {dataset}",
        f"Number of training points: {n_train}",
        f"Number of testing points: {x_test.shape[0]}",
        f"Hidden layer sizes: {hidden_sizes}",
        f"Learning rate: {learning_rate}",
        f"Batch size: {batch_size}",
        f"Epochs: {epochs}",
        f"Final train loss: {train_losses[-1]:.5e}",
        f"Final test loss: {test_losses[-1]:.5e}",
        f"Elapsed time for training NN: {elapsed_time:.3f} seconds\n",
    ]
    log_message = "\n".join(log_lines)
    print(log_message)

    results_dir = Path(__file__).parent / "results"
    log_results(
        log_message,
        path_to_log=results_dir / f"{dataset}_nn.txt",
    )

    if verbose_plot:
        # Plot train and test loss over epochs with hyperparameters included
        nn.plot_losses_verbose(
            train_losses,
            test_losses,
            learning_rate,
            batch_size,
            hidden_sizes,
            normalize_x=False,
            scale_x=False,
            normalize_y=False,
            scale_y=False,
            train_data_size=n_train,
            test_data_size=x_test.shape[0],
            dataset=dataset,
            plots_dir=plots_dir,
        )

    else:
        # Plot train and test loss over epochs
        nn.plot_losses(train_losses, test_losses, dataset, plots_dir)

    # Get neural network predictions
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        predictions = model(x_test)
    nn.plot_predictions(y_test, predictions, test_losses[-1], dataset, plots_dir)


if __name__ == "__main__":
    main()
