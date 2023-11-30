import torch
import torchvision.models as models
import torch.nn as nn

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=10):
        super(EfficientNetB0, self).__init__()
        self.efficientnet_b0 = models.efficientnet_b0(pretrained=True)
        self.efficientnet_b0.classifier[1] = nn.Linear(self.efficientnet_b0.classifier[1].in_features, num_classes)

        # freezing parameters in model
        # for param in self.efficientnet_b0.parameters():
        #     param.requires_grad = False
        # for param in self.efficientnet_b0.features[-1].parameters():
        #     param.requires_grad = True
        # for param in self.efficientnet_b0.classifier.parameters():
        #     param.requires_grad = True

        # Printing which layers are trainable
        for name, param in self.efficientnet_b0.named_parameters():
            print(name, param.requires_grad)

    def forward(self, x):
        return self.efficientnet_b0(x)
