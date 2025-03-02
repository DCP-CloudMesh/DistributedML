# This is the main script for training a SimpleCNN model on CIFAR10 dataset.

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time
import shutil
import matplotlib.pyplot as plt
import numpy as np
from networks.simpleCNN import SimpleCNN
from dataloader.cifar10_dataset import CIFAR10Dataset_RandomShuffle, CIFAR10Classes
from dataloader.dataloader import get_data_loaders
from train.train import train
from train.val import val
from test.test import test


# Hyperparameters (can use CLI)
batch_size = 64
learning_rate = 0.001
epochs = 10

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = "mps"
print(f"Using device: {device}")


def get_class_distribution(dataloader):
    class_distribution = {i: 0 for i in range(len(CIFAR10Classes))}
    for _, labels in dataloader:
        for label in labels:
            item = label.item()
            class_distribution[item] += 1
    return class_distribution


def plot_class_distribution(
    train_class_distribution,
    test_class_distribution,
    save_path="",
):
    _, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(CIFAR10Classes))
    y_train = [train_class_distribution[cls] for cls in train_class_distribution.keys()]
    y_test = [test_class_distribution[cls] for cls in test_class_distribution.keys()]

    # Set width of bars and positions of the bars
    width = 0.35
    ax.bar(x - width / 2, y_train, width, label="Train")
    ax.bar(x + width / 2, y_test, width, label="Test")

    ax.set_xticks(x)
    ax.set_xticklabels(CIFAR10Classes, rotation=45)
    ax.set_ylabel("Number of Images")
    ax.set_title("Class Distribution in Train and Test Sets")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path + "class_distribution.png")
    plt.close()


def main():
    start_time = time.time()

    # Define transforms
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Create the datasets
    data_path = "CIFAR10/"
    if not os.path.exists(os.path.join(data_path, "output")):
        os.mkdir(os.path.join(data_path, "output"))
    dataset = CIFAR10Dataset_RandomShuffle(data_path, transform=transform)

    # we want to split the data set into 90% train and 10% test and we want different partitions and do 10 experiments
    num_experiments = 10
    for i in range(num_experiments):
        print(f"Experiment {i+1} of {num_experiments}")
        data_path = f"CIFAR10/random_shuffle/experiment_{i+1}/"
        # delete and create the directory
        if os.path.exists(data_path):
            shutil.rmtree(data_path)
        os.makedirs(data_path, exist_ok=True)

        # create new dataset and dataloader
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
        train_loader, val_loader, test_loader = get_data_loaders(
            train_dataset, test_dataset, batch_size
        )
        print("dataset and dataloader objects created")

        # determine the distribution of classes in train and test dataloader and plot them out
        train_class_distribution = get_class_distribution(train_loader)
        test_class_distribution = get_class_distribution(test_loader)
        print(train_class_distribution, test_class_distribution)
        plot_class_distribution(
            train_class_distribution, test_class_distribution, data_path
        )

        # define the model and loss and optimizer
        model = SimpleCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        print("model and loss and optimizer objects created")

        # train the model
        for epoch in range(epochs):
            print("training")
            train(model, device, train_loader, optimizer, criterion, epoch)
            print("validating")
            val(model, device, val_loader, criterion, epoch, data_path)

        # Test the model
        test(model, device, test_loader, criterion, data_path)

        # Save the model checkpoint
        torch.save(model.state_dict(), f"{data_path}output/model.pth")
        print("Finished Training. Model saved as model.pth.")

    end_time = time.time()
    print("Total Time: ", end_time - start_time)
    print("Start Time: ", start_time)
    print("End Time: ", end_time)


if __name__ == "__main__":
    main()

    # # test plot
    # train_class_distribution = {
    #     0: 500,
    #     1: 500,
    #     2: 500,
    #     3: 500,
    #     4: 500,
    #     5: 500,
    #     6: 500,
    #     7: 500,
    #     8: 500,
    #     9: 500,
    # }
    # test_class_distribution = {
    #     0: 100,
    #     1: 100,
    #     2: 100,
    #     3: 100,
    #     4: 100,
    #     5: 100,
    #     6: 100,
    #     7: 100,
    #     8: 100,
    #     9: 100,
    # }
    # plot_class_distribution(train_class_distribution, test_class_distribution)
