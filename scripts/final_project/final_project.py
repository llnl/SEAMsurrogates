#!/usr/bin/env python3

"""
This script trains a neural network on the LEO drag dataset.  It provides options
for specifying the number of epochs, batch size, sizes of hidden layers, and
learning rate.  It saves two images to the directory containing this script.

Usage:

# Make script executable
chmod +x ./nn_jag.py

# See help
./nn_jag.py -h

# Train a neural net with hidden layers of sizes 10 and 20
./nn_jag.py --hidden_sizes 10 20

# Train a neural net with hidden layers of sizes 5 and 10, a batch size 20,
#   and 200 epochs
./nn_jag.py --hidden_sizes 5 10 -b 20 -n 200

# Train a neural net with layers of size 60 and 60, a learning rate of 0.02,
#   and a batch size of 40
./nn_jag.py --hidden_sizes 60 60 -n 600 -l 0.02 -b 40
"""

import argparse

import torch

from surmod import jag, neural_network as nn


def parse_arguments():
    """Get command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train neural network on LEO drag data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-ns",
        "--num_samples",
        type=int,
        default=10000,
        help="Number of data samples pulled from the jag_10k dataset.",
    )

    parser.add_argument(
        "-tr",
        "--num_train",
        type=int,
        default=None,
        help="Number of test samples (default: 75%% of num_samples).",
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
    num_samples = args.num_samples
    if args.num_train is None:
        num_train = int(0.75 * args.num_samples)
    else:
        num_train = args.num_train
    seed = args.seed
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    hidden_sizes = args.hidden_sizes
    learning_rate = args.learning_rate
    verbose_plot = args.verbose_plot

    # Weight initialization (normal with mean = 0, sd = 0.1)
    initialize_weights_normal = True

    # Load data into data frame and split into train and test sets
    df = jag.load_data(n_samples=num_samples, random=False)
    print("LEO drag data subset shape:", df.shape)
    print(df.columns)
    
    x_train, x_test, y_train, y_test = jag.split_data(
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
        #   no scaling needed for JAG data (not currently implemented)
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
            objective_data="JAG",
        )

    else:
        # Plot train and test loss over epochs
        nn.plot_losses(train_losses, test_losses, "JAG")

    # Get neural network predictions
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        predictions = model(x_test)
    nn.plot_predictions(y_test, predictions, test_losses[-1], "JAG")


if __name__ == "__main__":
    main()
