import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from networks.simpleCNN import SimpleCNN
from dataloader.cifar10_dataset import CIFAR10Dataset
from dataloader.dataloader import get_data_loaders
from train.train import train
from train.val import val
from test.test import test


# Hyperparameters (can use CLI)
batch_size = 64
learning_rate = 0.001
epochs = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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

    # Create data loaders
    train_loader, val_loader, test_loader = get_data_loaders(train_dataset, test_dataset, batch_size)

    # Model & loss & optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(epochs):
        print('training')
        train(model, device, train_loader, optimizer, criterion, epoch)
        print('validating')
        val(model, device, val_loader, criterion, epoch, data_path)

    # Test the model
    test(model, device, test_loader, criterion, data_path)

    # Save the model checkpoint
    torch.save(model.state_dict(), f'{data_path}output/simple_cnn.pth')
    print('Finished Training. Model saved as simple_cnn.pth.')


if __name__ == '__main__':
    main()
