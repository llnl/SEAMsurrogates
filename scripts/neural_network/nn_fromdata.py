#!/usr/bin/env python3

"""
This script trains a neural network on a chosen dataset. It provides options
for specifying the number of epochs, batch size, sizes of hidden layers, and
learning rate. It saves two metric plots to the directory containing this script.

Usage:

# Make script executable
chmod +x ./nn_fromdata.py

# See help
./nn_fromdata.py -h

# Train a neural net with hidden layers of sizes 10 and 20
./nn_fromdata.py --hidden_sizes 10 20

# Train a neural net with hidden layers of sizes 5 and 10, a batch size 20,
#   and 200 epochs
./nn_fromdata.py --hidden_sizes 5 10 -b 20 -n 200

# Train a neural net with layers of size 60 and 60, a learning rate of 0.02,
#   and a batch size of 40
./nn_fromdata.py --hidden_sizes 60 60 -n 600 -l 0.02 -b 40
"""

import argparse

import torch

from surmod import neural_network as nn, data_processing


def parse_arguments():
    """Get command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train neural network on JAG ICF data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-d",
        "--data",
        type=str,
        choices=["JAG", "borehole"],
        default = "JAG",
        help="Which dataset to use (defualt: JAG)."
    )

    parser.add_argument(
        "-tr",
        "--num_train",
        type=int,
        default=400,
        help="Number of train samples (default: 400).",
    )

    parser.add_argument(
        "-te",
        "--num_test",
        type=int,
        default=100,
        help="Number of test samples (default: 100).",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
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
        "-hs",
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=[5, 5],
        help="Sizes of hidden layers.",
    )

    parser.add_argument(
        "-l",
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for SGD optimization.",
    )

    parser.add_argument(
        "-vp",
        "--verbose_plot",
        action="store_true",
        default=False,
        help="If set, includes (hyper)parameter values in loss plot title.",
    )

    args = parser.parse_args()

    return args


def main():
    # Parse command line arguments
    args = parse_arguments()
    data = args.data
    num_train = args.num_train
    num_test = args.num_test
    seed = args.seed
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    hidden_sizes = args.hidden_sizes
    learning_rate = args.learning_rate
    verbose_plot = args.verbose_plot

    # Check data availability
    num_samples = num_test + num_train
    if num_samples > 10000:
        raise ValueError(
            f"Requested samples ({num_samples}) exceed existing dataset(s) size "
            "limit (10000)."
        )

    # Weight initialization (normal with mean = 0, sd = 0.1)
    initialize_weights_normal = True

    # Load data into data frame and split into train and test sets
    df = data_processing.load_data(dataset= data, n_samples=num_samples, random=False)
    print("Data subset shape:", df.shape)
    x_train, x_test, y_train, y_test = data_processing.split_data(
        df, LHD=False, n_train=num_train, seed=seed
    )

    # Convert training and test data to float32 tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Train the neural net
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
        # Plot train and test loss over epochs with (hyper)parameters included
        #   scaling for JAG data (not currently implemented; not needed)
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
            train_data_size=num_train,
            test_data_size=x_test.shape[0],
            objective_data=data,
        )

    else:
        # Plot train and test loss over epochs
        nn.plot_losses(train_losses, test_losses, data)

    # Get neural network predictions
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        predictions = model(x_test)
    nn.plot_predictions(y_test, predictions, test_losses[-1], data)


if __name__ == "__main__":
    main()
