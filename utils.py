# This file contains utility functions for distributed training.

import torch
import torch.nn as nn
import torch.optim as optim
from train.train import train_distributed


# each process will have their own
def train_model(model, train_loader, queue, epoch, learning_rate, device):
    # print("Number of CPUs being used", multiprocessing.cpu_count())
    # Model & loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("training")
    gradients = train_distributed(
        model, device, train_loader, optimizer, criterion, epoch
    )
    queue.put(gradients)


def get_model_parameters(model):
    """Extract parameters from a single model."""
    parameters = {
        name: param.clone().detach() for name, param in model.named_parameters()
    }
    return parameters


def average_model_parameters(model_parameters_list):
    """Average the parameters of models in a list."""
    avg_parameters = {}
    for key in model_parameters_list[0].keys():
        # Stack the same parameter from each model and then take the mean
        avg_parameters[key] = torch.stack(
            [params[key] for params in model_parameters_list if key in params.keys()]
        ).mean(dim=0)
    return avg_parameters


def average_model_gradients(gradient_list):
    """Average the gradients of models in a list."""
    avg_gradients = {}
    for key in gradient_list[0].keys():
        # Stack the same gradient from each model and then take the mean
        avg_gradients[key] = torch.stack(
            [grads[key] for grads in gradient_list if key in grads.keys()]
        ).mean(dim=0)
    return avg_gradients


def apply_averaged_parameters_and_gradients(model, avg_parameters, avg_gradients):
    """Apply averaged parameters and gradients to a model."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in avg_parameters:
                param.copy_(avg_parameters[name])
            if param.grad is not None and name in avg_gradients:
                param.grad.copy_(avg_gradients[name])
    return model


def apply_averaged_parameters(model, avg_parameters):
    """Apply averaged parameters and gradients to a model."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in avg_parameters:
                param.copy_(avg_parameters[name])
    return model
