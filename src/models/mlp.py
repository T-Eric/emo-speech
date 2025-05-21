import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """ 中间层都一样宽的MLP, 替换原本要学习的线性谱滤波器"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.is_linear = True
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("num_layers must be a positive integer.")
        elif num_layers == 1:
            # Linear transformation
            self.layer = nn.Linear(input_dim, output_dim)
        else:  # num_layers>=2
            # MLP with num_layers layers
            self.is_linear = False
            self.layers = nn.ModuleList()
            self.batch_norms = nn.ModuleList()

            self.layers.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers-2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers-1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.is_linear:
            return self.layer(x)
        else:
            for layer in range(self.num_layers-1):
                x = F.relu(self.batch_norms[layer](self.layers[layer](x)))
            return self.layers[-1](x)  # self.num_layers-1
