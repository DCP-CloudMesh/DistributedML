import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import multiprocessing
from multiprocessing import Process, Pool, Queue
import math
import copy
import time
import yaml

import torch.optim as optim
from networks.efficientNetB0 import EfficientNetB0
from networks.simpleCNN import SimpleCNN
from networks.resnet50 import Resnet50
from networks.resnet18 import Resnet18
from dataloader.cifar10_dataset import CIFAR10Dataset
from dataloader.dataloader import get_data_loaders
from train.val import val
from test.test import test
from train.train import train

from utils import LOG, LOG_AND_PRINT, train_model, get_model_parameters, average_model_parameters, apply_model_parameters, average_model_gradients, apply_averaged_parameters_and_gradients



def main():
    start_time = time.time()

    # Read from config file
    with open("config.yml", "r") as stream:
        try:
            configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    batch_size = configs.get('batch_size')
    learning_rate = configs.get('learning_rate')
    num_epochs = configs.get('num_epochs')
    num_partitions = configs.get('num_partitions')
    model_name = configs.get('model_name')
    device_name = configs.get('device_name')
    data_path = configs.get('data_path')
    logging_path = configs.get('logging_path')

    print(f"Using device: {device_name}")

    # Get user input
    while True:
        id = int(input("Enter process id: "))
        if id < 0 or id > num_partitions-1:
            print(f"Enter an id between 0 and {num_partitions-1}")
        else:
            break

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create the datasets
    if not os.path.exists(os.path.join(data_path, 'output')):
        os.mkdir(os.path.join(data_path, 'output'))
    train_dataset = CIFAR10Dataset(os.path.join(data_path, 'train'), transform=transform)
    test_dataset = CIFAR10Dataset(os.path.join(data_path, 'test'), transform=transform)

    # data loaders
    _, val_loader, test_loader = get_data_loaders(train_dataset, test_dataset, batch_size)

    # num_partitions & base model
    if model_name == 'SimpleCNN':
        model = SimpleCNN().to(device_name)
    elif model_name == "Resnet50":
        model = Resnet50().to(device_name)
    elif model_name == "Resnet18":
        model = Resnet18().to(device_name)
    elif model_name == "enet0":
        model = EfficientNetB0().to(device_name)
    else:
        print("Model not supported")
        exit()

    # data partitions
    partition_size = math.ceil(len(train_dataset) / num_partitions)
    partitions = []
    for i in range(num_partitions):
        start_idx = i * partition_size
        end_idx = min(start_idx + partition_size, len(train_dataset))
        partitions.append(Subset(train_dataset, range(start_idx, end_idx)))

    # Verify if the partitions are non-overlapping and cover the entire dataset
    assert sum([len(partition) for partition in partitions]) == len(train_dataset)

    # Create a DataLoader for each partition
    train_loaders = [DataLoader(partition, batch_size=batch_size, shuffle=True) for partition in partitions]
    train_loader = train_loaders[id-1]

    # Print separator before starting
    if id == 0: 
        LOG_AND_PRINT(text=f"========================================", file_path=logging_path)

    # Model & loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        LOG_AND_PRINT(f'epoch {epoch} started for process {id}', file_path=logging_path)
        LOG_AND_PRINT(f'training started for process {id}', file_path=logging_path)
        train(model, device_name, train_loader, optimizer, criterion, epoch)

        LOG_AND_PRINT(f'validating started for process {id}', file_path=logging_path)
        val(model, device_name, val_loader, criterion, epoch, data_path)

        LOG_AND_PRINT(text=f"Saving model_{id}_{epoch}.pth", file_path=logging_path)
        torch.save(model.state_dict(), f'{data_path}output/model_{id}_{epoch}.pth')

        # The leader and must combine the model
        if id == 0:
            follower_models = [] 
            for i in range(0, num_partitions):
                LOG_AND_PRINT(text=f"Leader is waiting for model_{i}_{epoch}.pth", file_path=logging_path)

                cur_model_path = f'{data_path}output/model_{i}_{epoch}.pth'

                # Busy wait until file is available
                while not os.path.exists(cur_model_path):
                    time.sleep(1)
                LOG_AND_PRINT(text=f"Leader read model_{i}_{epoch}.pth", file_path=logging_path)

                # Read and save the model
                temp_model = SimpleCNN().to(device_name)
                temp_model.load_state_dict(torch.load(cur_model_path))
                follower_models.append(temp_model)

                # Delete the model after reading it
                os.remove(cur_model_path)

            # Apply the averaged parameters to the leader model
            all_model_parameters = [get_model_parameters(model) for model in follower_models]
            avg_parameters = average_model_parameters(all_model_parameters)
            # model = SimpleCNN().to(device_name)
            apply_model_parameters(model, avg_parameters)
            # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Save the model checkpoint
            LOG_AND_PRINT(text=f"Saving model_combined_{epoch}.pth", file_path=logging_path)
            torch.save(model.state_dict(), f'{data_path}output/model_combined_{epoch}.pth')

            LOG_AND_PRINT(text=f"========================================", file_path=logging_path)
        # The followers must wait for the combined model from the leader before proceeding
        else: 
            LOG_AND_PRINT(text=f"Follower {id} is waiting for model_combined_{epoch}.pth", file_path=logging_path)

            cur_model_path = f"{data_path}output/model_combined_{epoch}.pth"

            # Busy wait until file is available
            while not os.path.exists(cur_model_path):
                time.sleep(1)

            LOG_AND_PRINT(text=f"Follower {id} read model_combined_{epoch}.pth", file_path=logging_path)

            # Read and load the model
            # model = SimpleCNN().to(device_name)
            model.load_state_dict(torch.load(cur_model_path))
            # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # for epoch in range(num_epochs):
    #     processes = []
    #     queues = []

    #     # model copys
    #     # models = [copy.deepcopy(base_model) for _ in range(num_partitions)]


    #     print(f"validation {epoch}")
    #     criterion = nn.CrossEntropyLoss()
    #     val(base_model, device_name, val_loader, criterion, epoch, data_path)

    #     # starting processes
    #     print(f"training {epoch}")
    #     train_model(base_model, train_loaders[i], epoch, learning_rate, device_name)

        # # print(gradients)
        # avg_gradients = average_model_gradients(gradients)

        # all_model_parameters = [get_model_parameters(model) for model in models]
        # avg_parameters = average_model_parameters(all_model_parameters)

        # apply_averaged_parameters_and_gradients(base_model, avg_parameters, avg_gradients)

        # optimizer = optim.Adam(base_model.parameters(), lr=learning_rate)
        # optimizer.step()

    # Test the model
    criterion = nn.CrossEntropyLoss()
    test(model, device_name, test_loader, criterion, data_path)

    # Save the model checkpoint
    torch.save(model.state_dict(), f'{data_path}/output/model_{id}_final.pth')
    print('Finished Training. Model saved as model_{id}_final.pth.')

    end_time = time.time()
    print("Total Time: ", end_time-start_time)
    print("Start Time: ", start_time)
    print("End Time: ", end_time)


if __name__ == '__main__':
    main()
