import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time

from networks.vision_transformer import VisionTransformer
from dataloader.cifar10_dataset import CIFAR10Dataset
from dataloader.dataloader import get_data_loaders
from train.train import train
from train.val import val
from test.test import test


# Hyperparameters (can use CLI)
batch_size = 64
learning_rate = 0.00001
epochs = 10

# transformer params
img_size=32
patch_size=4
in_channels=3      # CIFAR-10 images are RGB, so 3 input channels
embed_size=128     # Smaller embedding size for a smaller dataset
num_layers=6       # Fewer layers might be sufficient for CIFAR-10
num_heads=8        # Adjusted number of heads
num_classes=10     # CIFAR-10 has 10 classes
dropout=0.01

# device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# device = 'cpu'
print(f"Using device: {device}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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

    # Create data loaders
    train_loader, val_loader, test_loader = get_data_loaders(train_dataset, test_dataset, batch_size)

    # Model & loss & optimizer
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_size=embed_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(model)
    print(count_parameters(model))

    # Train the model
    for epoch in range(epochs):
        print('training')
        train(model, device, train_loader, optimizer, criterion, epoch)
        print('validating')
        val(model, device, val_loader, criterion, epoch, data_path)

    # Test the model
    test(model, device, test_loader, criterion, data_path)

    # Save the model checkpoint
    torch.save(model.state_dict(), f'{data_path}output/model.pth')
    print('Finished Training. Model saved as model.pth.')

    end_time = time.time()
    print("Total Time: ", end_time-start_time)
    print("Start Time: ", start_time)
    print("End Time: ", end_time)


if __name__ == '__main__':
    main()
