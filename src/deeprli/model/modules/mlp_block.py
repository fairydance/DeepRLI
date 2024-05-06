import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBlock(nn.Module):
  def __init__(self, input_dim, output_dim, n=3):
    super().__init__()
    layers = [nn.Linear(input_dim // 2**i, input_dim // 2**(i + 1), bias=True) for i in range(n - 1)]
    layers.append(nn.Linear(input_dim // 2**(n - 1), output_dim, bias=True))
    self.layers = nn.ModuleList(layers)
    self.n = n
    
  def forward(self, x):
    for i in range(self.n - 1):
      x = self.layers[i](x)
      x = F.relu(x)
    x = self.layers[self.n - 1](x)
    return x