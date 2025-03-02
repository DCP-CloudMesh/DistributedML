# This is the main script for training a SimpleCNN model on CIFAR10 dataset.

import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time
import shutil
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from multiprocessing import Process, Pool, Queue
from multiprocessing.managers import BaseManager
from torch.utils.data import DataLoader, Subset
import copy
from tqdm import tqdm

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
num_partitions = 5
num_experiments = 10


# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = "cpu"
print(f"Using device: {device}")


# model publisher class
class ModelPublisher:
    def __init__(self, num_partitions, model):
        self.model = model
        self.state_dicts = [None for _ in range(num_partitions)]
        # self.gradients = [{} for _ in range(num_partitions)]
        self.avg_check = True

    def get_model(self):
        return self.model

    def get_state_dict(self, model: nn.Module):
        return model.state_dict()

    def average_state_dicts(self):
        # print([type(i) for i in self.state_dicts])
        if self.avg_check:
            for state_dict in self.state_dicts:
                if state_dict is None:
                    return None
            self.avg_check = False
        avg_state_dict = {}
        for key in self.state_dicts[0]:
            tensors = [
                state_dict[key].float()
                for state_dict in self.state_dicts
                if key in state_dict.keys()
            ]
            avg_state_dict[key] = torch.stack(tensors).mean(dim=0)
        return avg_state_dict

    def publish(self, model_index, model):
        self.state_dicts[model_index] = self.get_state_dict(model)
        avg_state_dicts = self.average_state_dicts()

        # apply averaged state dicts to model
        if avg_state_dicts is not None:
            self.model.load_state_dict(avg_state_dicts)


class MyManager(BaseManager):
    pass


MyManager.register("ModelPublisher", ModelPublisher)


def Manager():
    m = MyManager()
    m.start()
    return m


def train_model_iteration(
    model_index: int,
    model_publisher,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    data_path,
    device,
):
    # model
    model = copy.deepcopy(model_publisher.get_model())

    # Model & loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("training")
    model.train()

    iteration = 0

    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch+1}/{num_epochs}")
        running_loss = 0.0

        # Wrap your data loader with tqdm for a progress bar
        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc="Training Epoch {}".format(epoch + 1),
        )
        for batch_idx, (data, target) in progress_bar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            progress_bar.set_postfix(
                loss=running_loss / (batch_idx + 1),
                current_batch=batch_idx,
                refresh=True,
            )

            #  update the global network, only update every 10 iterations to make program faster
            if iteration % 10 == 0:
                model_publisher.publish(model_index, model)
                pub_state_dict = model_publisher.get_model().state_dict()
                model.load_state_dict(pub_state_dict)

            iteration += 1

        print("[Epoch %d] loss: %.3f" % (epoch + 1, running_loss / len(train_loader)))
    return


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


def run_experiment(
    experiment_num, dataset, device, batch_size, num_partitions, epochs, learning_rate
):
    print(f"Starting Experiment {experiment_num+1} of {num_experiments}")
    data_path = f"CIFAR10/random_shuffle_distributed/experiment_{experiment_num+1}/"
    # delete and create the directory
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.makedirs(data_path, exist_ok=True)

    # create new dataset and dataloader
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader, val_loader, test_loader = get_data_loaders(
        train_dataset, test_dataset, batch_size
    )
    print(f"Dataset and dataloader objects created for experiment {experiment_num+1}")

    # determine the distribution of classes
    train_class_distribution = get_class_distribution(train_loader)
    test_class_distribution = get_class_distribution(test_loader)
    plot_class_distribution(
        train_class_distribution, test_class_distribution, data_path
    )

    # define the base model
    base_model = SimpleCNN().to(device)

    # data partitions
    partition_size = math.ceil(len(train_dataset) / num_partitions)
    partitions = []
    for i in range(num_partitions):
        start_idx = i * partition_size
        end_idx = min(start_idx + partition_size, len(train_dataset))
        partitions.append(Subset(train_dataset, range(start_idx, end_idx)))

    # Create a DataLoader for each partition
    train_loaders = [
        DataLoader(partition, batch_size=batch_size, shuffle=True)
        for partition in partitions
    ]

    manager = Manager()
    model_publisher = manager.ModelPublisher(num_partitions, base_model)

    # starting processes for partitions
    processes = []
    for i in range(num_partitions):
        p = Process(
            target=train_model_iteration,
            args=(
                i,
                model_publisher,
                train_loaders[i],
                val_loader,
                epochs,
                learning_rate,
                data_path,
                device,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    base_model = model_publisher.get_model()

    # Test the model
    criterion = nn.CrossEntropyLoss()
    test(base_model, device, test_loader, criterion, data_path)

    # Save the model checkpoint
    torch.save(base_model.state_dict(), f"{data_path}output/model.pth")
    print(f"Finished Training Experiment {experiment_num+1}. Model saved as model.pth.")


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

    # Run experiments in parallel
    experiment_processes = []
    for i in range(num_experiments):
        p = Process(
            target=run_experiment,
            args=(
                i,
                dataset,
                device,
                batch_size,
                num_partitions,
                epochs,
                learning_rate,
            ),
        )
        p.start()
        experiment_processes.append(p)

    for p in experiment_processes:
        p.join()

    end_time = time.time()
    print("Total Time: ", end_time - start_time)
    print("Start Time: ", start_time)
    print("End Time: ", end_time)


if __name__ == "__main__":
    main()
