#!/usr/bin/env python3

"""
This script trains a neural network on the LEO drag dataset.  It provides options
for specifying the number of epochs, batch size, sizes of hidden layers, and
learning rate.  It saves two images to the directory containing this script.
"""

import argparse

import torch
import seaborn as sns
import matplotlib.pyplot as plt

from surmod import leo_drag, neural_network as nn


def main():
    # control parameters here rather than through args
    num_samples = 10000
    num_train = int(0.75 * num_samples)
    seed = 42
    num_epochs = 100
    batch_size = 5
    hidden_sizes = [12, 12]
    learning_rate = .001
    verbose_plot = False

    # Weight initialization (normal with mean = 0, sd = 0.1)
    initialize_weights_normal = True

    # Load data into data frame and split into train and test sets
    df = leo_drag.load_data(n_samples=num_samples, random=False)
    print("LEO drag data subset shape:", df.shape)
    
    sns.pairplot(df)
    plt.savefig('pairplot.png')
    
    x_train, x_test, y_train, y_test = leo_drag.split_data(
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
        #   no scaling needed for drag data (not currently implemented)
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
            objective_data="LEO Drag",
        )

    else:
        # Plot train and test loss over epochs
        nn.plot_losses(train_losses, test_losses, "LEO Drag")

    # Get neural network predictions
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        predictions = model(x_test)
    nn.plot_predictions(y_test, predictions, test_losses[-1], "LEO Drag")

if __name__ == "__main__":
    main()
