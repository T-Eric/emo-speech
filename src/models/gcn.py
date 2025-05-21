import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import fractional_matrix_power

from .mlp import MLP

def calc_degree(A):
    # calculate the degree matrix
    D = torch.diag(torch.sum(A, dim=1)).to(A.device)
    return D


class GraphConv(nn.Module):
    # A single graph convolution layer
    def __init__(self, in_dim, out_dim, num_layers=2, hidden_dim=128):
        '''
        These dims are for MLPs
        '''
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.MLP = MLP(num_layers, in_dim, hidden_dim, out_dim)

        # init weights and biases
        for i in range(num_layers):
            nn.init.xavier_uniform_(self.MLP.layers[i].weight)
            nn.init.constant_(self.MLP.layers[i].bias, 0.0)

    def forward(self, features, A):
        '''
        features: (batch_size,num_nodes,in_features)
        A: Adjacency matrix
        '''
        b, t, d = features.shape
        assert d == self.in_dim

        if (len(A.shape) == 2):
            # all the graphs share the same A
            Deg = calc_degree(A)
            sqrt_Deg = torch.FloatTensor(fractional_matrix_power(
                Deg.detach().cpu(), -0.5)).cuda()  # 神秘操作

            Lap = Deg-A
            Lap_norm = sqrt_Deg.matmul(Lap.matmul(sqrt_Deg))

            eigvals, U=torch.linalg.eigh(Lap_norm)

            repeated_U_t = U.t().repeat(b, 1, 1)
            repeated_U = U.repeat(b, 1, 1)

        else:
            # each graph has its own A
            repeated_U_t = []
            repeated_U = []
            for i in range(b):
                Deg = calc_degree(A[i])
                sqrt_Deg = torch.FloatTensor(fractional_matrix_power(
                    Deg.detach().cpu(), -0.5)).cuda()

                Lap = Deg-A[i]
                Lap_norm = sqrt_Deg.matmul(Lap.matmul(sqrt_Deg))

                eigvals, U=torch.linalg.eigh(Lap_norm)

                repeated_U_t.append(U.t().view(1, U.shape[0], U.shape[1]))
                repeated_U.append(U.view(1, U.shape[0], U.shape[1]))

            repeated_U_t = torch.cat(repeated_U_t)
            repeated_U = torch.cat(repeated_U)
            
        assert not repeated_U_t.is_complex(), "repeated_U_t is complex tensor"
        assert not repeated_U.is_complex(), "repeated_U is complex tensor"
        assert not features.is_complex(), "features is complex tensor"


        agg_feats = torch.bmm(repeated_U_t, features)

        # reshape to (batch_size*num_nodes,out_features) to apply MLP
        ret = self.MLP(agg_feats.view(-1, d)).view(b, -1, self.out_dim)
        ret = torch.bmm(repeated_U, ret)

        return ret


class GCN(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, final_dropout, graph_pooling_type, device, adj):
        super(GCN, self).__init__()

        self.final_dropout = final_dropout
        self.graph_pooling_type = graph_pooling_type
        self.device = device
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.adj = adj

        self.gcs = torch.nn.ModuleList()
        self.gcs.append(GraphConv(in_dim, hidden_dim))
        for i in range(num_layers-1):
            self.gcs.append(GraphConv(hidden_dim, hidden_dim))

        self.classifier = nn.Sequential(nn.Linear(self.hidden_dim, 128), nn.Dropout(
            p=self.final_dropout), nn.PReLU(128), nn.Linear(128, out_dim))

    def forward(self, X_concat):
        # X_concat: (batch_size,num_nodes,in_features)
        h = X_concat
        A = F.relu(self.adj)

        for layer in self.gcs:
            h = F.relu(layer(h, A))

        if self.graph_pooling_type == 'mean':
            h = h.mean(dim=1)
        elif self.graph_pooling_type == 'sum':
            h = h.sum(dim=1)
        elif self.graph_pooling_type == 'max':
            h = h.max(dim=1)[0]
        else:
            raise ValueError('Invalid graph pooling type')

        ret = self.classifier(h)
        return ret

class SkipGCN(nn.Module):
    # 在每次变换后保留原特征，追加一层独立MLP
    pass 