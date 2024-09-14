import torch
from torch import nn
import numpy as np

class MatrixMultiplyModule(nn.Module):
    def __init__(self, size=5):
        super(MatrixMultiplyModule, self).__init__()
        self.rand = torch.rand((size,size)).type(torch.float32)
        self.rand = nn.Parameter(self.rand)
        
    def forward(self, tensor):
        return torch.mm(tensor, self.rand)

if __name__ == '__main__':
    model = MatrixMultiplyModule(5)
    tensor = torch.eye(5).type(torch.float32)
    out = model(tensor)
    print(out.shape, out)
