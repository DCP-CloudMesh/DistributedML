import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import multiprocessing
from multiprocessing import Process, Pool, Queue
from multiprocessing.managers import BaseManager
import math
import copy
import time
import yaml

import torch.optim as optim
from tqdm import tqdm
import torch.nn.init as init

from networks.efficientNetB0 import EfficientNetB0
from networks.simpleCNN import SimpleCNN
from networks.resnet50 import Resnet50
from networks.resnet18 import Resnet18
from dataloader.cifar10_dataset import CIFAR10Dataset
from dataloader.dataloader import get_data_loaders
from train.val import val
from test.test import test

from utils import get_model_parameters, average_model_parameters, average_model_gradients, apply_averaged_parameters_and_gradients, apply_averaged_parameters

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
            tensors = [state_dict[key].float() for state_dict in self.state_dicts if key in state_dict.keys()]
            avg_state_dict[key] = torch.stack(tensors).mean(dim=0)
        return avg_state_dict
        
 
    def publish(self, model_index, model):
        '''
        gradients = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                gradients[name] = parameter.grad.clone().to('cpu')
        self.gradients[model_index] = gradients
        avg_gradients = average_model_gradients(self.gradients)
        
        self.params[model_index] = get_model_parameters(model)
        avg_parameters = average_model_parameters(self.params)
        
        # self.model = apply_averaged_parameters_and_gradients(self.model, avg_parameters, avg_gradients)
        self.model = apply_averaged_parameters(self.model, avg_parameters)
        '''
        self.state_dicts[model_index] = self.get_state_dict(model)
        avg_state_dicts = self.average_state_dicts()
        
        # apply averaged state dicts to model
        if avg_state_dicts is not None:
            self.model.load_state_dict(avg_state_dicts)
            

class MyManager(BaseManager):
    pass


MyManager.register('ModelPublisher', ModelPublisher)


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
        device
    ):
    # model
    model = copy.deepcopy(model_publisher.get_model())

    # Model & loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print('training')
    model.train()
    
    iteration = 0

    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch+1}/{num_epochs}")
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
            
            #  update the global network
            model_publisher.publish(model_index, model)
            pub_state_dict = model_publisher.get_model().state_dict()
            model.load_state_dict(pub_state_dict)
            
            # pub_params = get_model_parameters(model_publisher.get_model())
            # model = apply_averaged_parameters(model, pub_params)
            # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # model = copy.deepcopy(model_publisher.get_model())
            # optimizer.step()
            iteration += 1

        print('[Epoch %d] loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))
        
        # if model_index == 0:
        #     print(f"Validation {epoch}")
        #     val(model, device, val_loader, criterion, epoch, data_path)

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

    print(f"Using device: {device_name}")

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
        base_model = SimpleCNN().to(device_name)
    elif model_name == "Resnet50":
        base_model = Resnet50().to(device_name)
    elif model_name == "Resnet18":
        base_model = Resnet18().to(device_name)
    elif model_name == "enet0":
        base_model = EfficientNetB0().to(device_name)
    else:
        print("Model not supported")
        exit()
    base_model.train()
    
    # # apply gaussian distribution on model params
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             init.normal_(m.weight, mean=0.0, std=0.1)
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             init.normal_(m.weight, mean=0.0, std=0.1)
    #             init.constant_(m.bias, 0)
    # base_model.apply(_initialize_weights)

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
    
    # # starting models
    # models = [copy.deepcopy(base_model) for _ in range(num_partitions)]
    # global model_publisher
    # model_publisher = ModelPublisher(num_partitions, base_model)
    manager = Manager()
    model_publisher = manager.ModelPublisher(num_partitions, base_model)
    
    # starting processes
    processes = []

    for i in range(num_partitions):
        p = Process(target=train_model_iteration, args=(i, model_publisher, train_loaders[i], val_loader, num_epochs, learning_rate, data_path, device_name))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
    
    base_model = model_publisher.get_model()

    # Test the model
    criterion = nn.CrossEntropyLoss()
    test(base_model, device_name, test_loader, criterion, data_path)

    # Save the model checkpoint
    torch.save(base_model.state_dict(), f'{data_path}output/model.pth')
    print('Finished Training. Model saved as model.pth.')

    end_time = time.time()
    print("Total Time: ", end_time-start_time)
    print("Start Time: ", start_time)
    print("End Time: ", end_time)


if __name__ == '__main__':
    main()
