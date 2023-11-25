import torch
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights
import torch.nn as nn
import torch.optim as optim


class Resnet50(nn.Module):
    def __init__(self, num_classes=10):
        super(Resnet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)

        # freezing all layers on backbone
        for param in self.resnet50.parameters():
            param.requires_grad = False

        # only training fully connected layer and last conv layer
        for param in self.resnet50.layer4.parameters():
            param.requires_grad = True

        for param in self.resnet50.fc.parameters():
            param.requires_grad = True

        for name, param in self.resnet50.named_parameters():
            print(name, param.requires_grad)


    def forward(self, x):
        return self.resnet50(x)
