# This is the main script for training a SimpleCNN model on CIFAR10 dataset.

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time
import sys

from networks.simpleCNN import SimpleCNN
from networks.resnet18 import Resnet18
from dataloader.cifar10_dataset import CIFAR10Dataset
from dataloader.dataloader import get_data_loaders
from train.train import train_quantized
from train.val import val
from test.test import test


# Hyperparameters (can use CLI)
batch_size = 64
learning_rate = 0.01
epochs = 10

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = "mps"
print(f"Using device: {device}")


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
    train_dataset = CIFAR10Dataset(
        os.path.join(data_path, "train"), transform=transform
    )
    test_dataset = CIFAR10Dataset(os.path.join(data_path, "test"), transform=transform)

    # Create data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        train_dataset, test_dataset, batch_size
    )

    # Model & loss & optimizer
    # model = SimpleCNN().to(device)
    model = Resnet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # initalize model with gaussian distribution with mean 0 and std 10
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = torch.randn(param.data.shape) * 1000
            param.data = param.data.to(torch.float32).to(device)

    # Train the model
    for epoch in range(epochs):
        print("training")
        train_quantized(model, device, train_loader, optimizer, criterion, epoch)
        print("validating")
        val(model, device, val_loader, criterion, epoch, data_path)

        # # print model params
        # for name, param in model.named_parameters():
        #     print(name, param.data)

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
