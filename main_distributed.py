import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from multiprocessing import Process, Pool, Queue
import math
import copy

from networks.simpleCNN import SimpleCNN
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

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = 'cpu'
print(f"Using device: {device}")

# each process will have their own
def train_model(model, train_loader, queue, epoch):
    # Model & loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print('training')
    gradients = train_distributed(model, device, train_loader, optimizer, criterion, epoch)
    # print('validating')
    # val(model, device, val_loader, criterion, epoch, data_path)
    queue.put(gradients)
    

def main():
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
    _, _, test_loader = get_data_loaders(train_dataset, test_dataset, batch_size)

    # num_partitions & base model
    num_partitions = 5
    base_model = SimpleCNN().to(device)

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
    print(train_loaders)

    for epoch in range(epochs):
        processes = []
        queues = []

        # model copys
        models = [copy.deepcopy(base_model) for _ in range(num_partitions)]

        # starting processes
        for i in range(num_partitions):
            queue = Queue()
            queues.append(queue)
            p = Process(target=train_model, args=(models[i], train_loaders[i], queue, epoch))
            p.start()
            processes.append(p)

        # getting result from queues
        gradients = []
        for q in queues:
            gradients.append(q.get())

        # join processes
        for p in processes:
            p.join()

        print(gradients)

        # averaging the gradients
        avg_gradients = {k: torch.stack([grad[k] for grad in gradients]).mean(dim=0) 
                    for k in gradients[0]}

        for name, parameter in base_model.named_parameters():
            if name in avg_gradients:
                parameter.grad = avg_gradients[name]

        optimizer = optim.Adam(base_model.parameters(), lr=learning_rate)
        optimizer.step()

    # Test the model
    criterion = nn.CrossEntropyLoss()
    test(base_model, device, test_loader, criterion, data_path)

    exit(0)
        
    # Average out the gradients
    



    

    # # Train the model
    # for epoch in range(epochs):
    #     print('training')
    #     train(model, device, train_loader, optimizer, criterion, epoch)
    #     print('validating')
    #     val(model, device, val_loader, criterion, epoch, data_path)

    # # Test the model
    # test(model, device, test_loader, criterion, data_path)

    # # Save the model checkpoint
    # torch.save(model.state_dict(), f'{data_path}output/model.pth')
    # print('Finished Training. Model saved as model.pth.')


if __name__ == '__main__':
    main()
