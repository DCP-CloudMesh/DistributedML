import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import multiprocessing
from multiprocessing import Process, Pool, Queue
import math
import copy
import time

import socket
import pickle
from io import BytesIO

from networks.efficientNetB0 import EfficientNetB0
from networks.simpleCNN import SimpleCNN
from networks.resnet50 import Resnet50
from networks.resnet18 import Resnet18
from dataloader.cifar10_dataset import CIFAR10Dataset
from dataloader.dataloader import get_data_loaders
from train.train import train_distributed
from train.val import val
from test.test import test

# Pytorch DDP
# https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

# Hyperparameters (can use CLI)
batch_size = 64
learning_rate = 0.001
epochs = 10
model_name = 'Resnet18'
num_partitions = 5
device = 'cpu'
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

wandb.init(
    project="cloudmesh-iteration-based-training",
    config={
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "architecture": model_name,
        "dataset": "CIFAR10",
        "num_partitions": num_partitions,
        "device": device
    }
)


# each process will have their own
def train_model(model, train_loader, queue, epoch):
    # print("Number of CPUs being used", multiprocessing.cpu_count())
    # Model & loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print('training')
    gradients = train_distributed(model, device, train_loader, optimizer, criterion, epoch)
    queue.put(gradients)
    

def get_model_parameters(model):
    """ Extract parameters from a single model. """
    parameters = {name: param.clone().detach() for name, param in model.named_parameters()}
    return parameters


def average_model_parameters(model_parameters_list):
    """ Average the parameters of models in a list. """
    avg_parameters = {}
    for key in model_parameters_list[0].keys():
        # Stack the same parameter from each model and then take the mean
        avg_parameters[key] = torch.stack([params[key] for params in model_parameters_list]).mean(dim=0)
    return avg_parameters


def average_model_gradients(gradient_list):
    """ Average the gradients of models in a list. """
    avg_gradients = {}
    for key in gradient_list[0].keys():
        # Stack the same gradient from each model and then take the mean
        avg_gradients[key] = torch.stack([grads[key] for grads in gradient_list]).mean(dim=0)
    return avg_gradients


def apply_averaged_parameters_and_gradients(model, avg_parameters, avg_gradients):
    """ Apply averaged parameters and gradients to a model. """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in avg_parameters:
                param.copy_(avg_parameters[name])
            if param.grad is not None and name in avg_gradients:
                param.grad.copy_(avg_gradients[name])
                

def send_data_bounds(client_socket, partition_boundaries):
    id = client_socket.recv(1024).decode('utf-8')
    if not id: return
    print(f"Client id {id} connecting")
    
    id = int(id)
    partition = partition_boundaries[id]
    
    buffer = BytesIO()
    pickle.dump(partition, buffer)
    buffer.seek(0)
    
    # Send the serialized list to the client
    data = buffer.read()
    client_socket.sendall(data)
    client_socket.close()
    return

def send_model_info(client_socket, base_model):
    id = client_socket.recv(1024).decode('utf-8')
    if not id: return
    print(f"Client id {id} connecting")
    
    buffer = BytesIO()
    torch.save(base_model.state_dict(), buffer)
    buffer.seek(0) 
    
    data = buffer.read()
    client_socket.sendall(data)
    client_socket.close()
    return


def main():
    start_time = time.time()

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create the datasets
    data_path = 'CIFAR10/'
    if not os.path.exists(os.path.join(data_path, 'output')):
        os.mkdir(os.path.join(data_path, 'output'))
    train_dataset = CIFAR10Dataset(os.path.join(data_path, 'train'), transform=transform)
    test_dataset = CIFAR10Dataset(os.path.join(data_path, 'test'), transform=transform)

    # data loaders
    _, val_loader, test_loader = get_data_loaders(train_dataset, test_dataset, batch_size)

    # num_partitions & base model
    base_model = None
    if model_name == 'SimpleCNN':
        base_model = SimpleCNN().to(device)
    elif model_name == "Resnet50":
        base_model = Resnet50().to(device)
    elif model_name == "Resnet18":
        base_model = Resnet18().to(device)
    elif model_name == "enet0":
        base_model = EfficientNetB0().to(device)
    else:
        print("Model not supported")
        exit()

    # data partitions
    partition_size = math.ceil(len(train_dataset) / num_partitions)
    partitions = []
    partition_boundaries = []
    for i in range(num_partitions):
        start_idx = i * partition_size
        end_idx = min(start_idx + partition_size, len(train_dataset))
        partition_boundaries.append((start_idx, end_idx))
        partitions.append(Subset(train_dataset, range(start_idx, end_idx)))

    # Verify if the partitions are non-overlapping and cover the entire dataset
    assert sum([len(partition) for partition in partitions]) == len(train_dataset)

    # # Create a DataLoader for each partition
    # train_loaders = [DataLoader(partition, batch_size=batch_size, shuffle=True) for partition in partitions]
    # print(train_loaders)
    
    # first we send data to client
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', 9999))
    server_socket.listen(5)
    print("Listening on port 9999...")
    
    # data set connections
    for i in range(num_partitions):
        client, addr = server_socket.accept()
        print(f"Accepted connection from {addr[0]}:{addr[1]}")
        client_handler = threading.Thread(target=send_data_bounds, args=(client, partition_boundaries,))
        client_handler.start()
        
    server_socket.listen(5)
    print("Listening on port 9999...")

    # setting up the model
    for i in range(num_partitions):
        client, addr = server_socket.accept()
        print(f"Accepted connection from {addr[0]}:{addr[1]}")
        client_handler = threading.Thread(target=send_model_info, args=(client, base_model))
        client_handler.start()
        
        
        
        
        
        
        
        


    for epoch in range(epochs):
        processes = []
        queues = []

        # model copys
        models = [copy.deepcopy(base_model) for _ in range(num_partitions)]

        # print(f"validation {epoch}")
        # criterion = nn.CrossEntropyLoss()
        # p = Process(target=val, args=(base_model, device, val_loader, criterion, epoch, data_path))
        # p.start()
        # processes.append(p)

        # starting processes
        print(f"training {epoch}")
        for i in range(num_partitions):
            queue = Queue()
            queues.append(queue)
            p = Process(target=train_model, args=(models[i], train_loaders[i], queue, epoch))
            p.start()
            processes.append(p)

        # join processes
        for p in processes:
            p.join()

        # getting result from queues
        gradients = []
        for q in queues:
            gradients.append(q.get())

        # print(gradients)
        avg_gradients = average_model_gradients(gradients)

        all_model_parameters = [get_model_parameters(model) for model in models]
        avg_parameters = average_model_parameters(all_model_parameters)

        apply_averaged_parameters_and_gradients(base_model, avg_parameters, avg_gradients)

        # optimizer = optim.Adam(base_model.parameters(), lr=learning_rate)
        # optimizer.step()

    # Test the model
    criterion = nn.CrossEntropyLoss()
    test(base_model, device, test_loader, criterion, data_path)

    # Save the model checkpoint
    torch.save(base_model.state_dict(), f'{data_path}output/model.pth')
    print('Finished Training. Model saved as model.pth.')

    wandb.finish()
    end_time = time.time()
    print("Total Time: ", end_time-start_time)
    print("Start Time: ", start_time)
    print("End Time: ", end_time)


if __name__ == '__main__':
    main()
