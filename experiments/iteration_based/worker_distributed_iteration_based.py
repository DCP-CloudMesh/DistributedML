import numpy as np
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
from train.val import val
from test.test import test


# Hyperparameters (can use CLI)
batch_size = 64
learning_rate = 0.001
epochs = 10
model_name = 'Resnet18'
num_partitions = 5
device = 'cpu'
print(f"Using device: {device}")


def main(id):
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
    model = None
    if model_name == 'SimpleCNN':
        model = SimpleCNN().to(device)
    elif model_name == "Resnet50":
        model = Resnet50().to(device)
    elif model_name == "Resnet18":
        model = Resnet18().to(device)
    elif model_name == "enet0":
        model = EfficientNetB0().to(device)
    else:
        print("Model not supported")
        return

    ############################################################
    # LOADING DATA
    buffer = BytesIO()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect(('localhost', 9999))
        try:
            # Sending data to server
            message = f"{id}"
            client_socket.send(message.encode('utf-8'))

            # Receiving data from server
            while True:
                part = client_socket.recv(1024)
                if not part:
                    break 
                buffer.write(part)

        except Exception as e:
            print(f"Error: {e}")
            return
    
    buffer.seek(0)
    partition_bounds = pickle.load(buffer)
    partition = Subset(train_dataset, range(partition_bounds[0], partition_bounds[1]))
    train_loader = DataLoader(partition, batch_size=batch_size, shuffle=True)
    
    ############################################################
    # barrier socket (recieve a message)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as barrier_socker:
        client_socket.connect(('localhost', 7777))
        
    
    
    ############################################################
    # LOADING MODEL
    buffer = BytesIO()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect(('localhost', 9999))
        try:
            # Sending data to server
            message = f"{id}"
            client_socket.send(message.encode('utf-8'))

            # Receiving data from server
            while True:
                part = client_socket.recv(1024)
                if not part:
                    break 
                buffer.write(part)

        except Exception as e:
            print(f"Error: {e}")
            return
    
    buffer.seek(0)
    try:
        model_state_dict = torch.load(buffer, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        return
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.load_state_dict(model_state_dict)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        # Wrap your data loader with tqdm for a progress bar
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Training Epoch {}'.format(epoch+1))
        for batch_idx, (data, target) in progress_bar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss/(batch_idx+1), current_batch=batch_idx, refresh=True)
            
            logging_data = [
                id,
                epoch + 1,
                batch_idx,
                running_loss / (batch_idx + 1),
            ]
        
    end_time = time.time()
    print("Total Time: ", end_time-start_time)
    print("Start Time: ", start_time)
    print("End Time: ", end_time)


if __name__ == "__main__":
    id = int(input('id: '))
    main(id)
