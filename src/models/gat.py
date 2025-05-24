import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import fractional_matrix_power
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Batch, Data

from .mlp import MLP
from .gcn import GraphConv, calc_degree


class SkipGraphConv(nn.Module):
    '''
    暂时让所有的连接处都使用hidden_dim
    更新：去掉所有的跳跃连接，采用hidden_dim
    '''

    def __init__(self, in_dim, out_dim, num_layers=2, hidden_dim=128):
        super(SkipGraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # assert in_dim == out_dim, "in_dim must be equal to out_dim"
        self.graph_conv = GraphConv(in_dim, hidden_dim, num_layers, hidden_dim)
        self.MLP = MLP(num_layers,hidden_dim, hidden_dim, out_dim)

        # init weights and biases
        for i in range(num_layers):
            nn.init.xavier_uniform_(self.MLP.layers[i].weight)
            nn.init.constant_(self.MLP.layers[i].bias, 0.0)

    def forward(self, features, A):
        b, t, d = features.shape
        h_gcn = self.graph_conv(features, A)
        # h_skip = h_gcn+features
        h_mlp = self.MLP(h_gcn.view(-1, h_gcn.shape[-1])).view(b, -1, self.out_dim)
        return h_mlp


class GAT(nn.Module):
    def __init__(self, in_dim, out_dim, heads=1, dropout=0.5, concat=True):
        super(GAT, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        
        self.conv = GATConv(
            in_channels=in_dim,
            out_channels=out_dim if not concat else out_dim // heads,
            heads=heads,
            dropout=dropout,
            concat=concat
        )

    def forward(self, features, A):
        '''
        features: (batch_size, num_nodes, in_features)
        A: 邻接矩阵 (batch_size, num_nodes, num_nodes) 或 (num_nodes, num_nodes)

        返回:
        output: (batch_size, num_nodes, out_features)
        '''
        b, n, _ = features.shape

        batch_x = features.view(-1, self.in_dim)

        if A.dim() == 2:
            edge_index, _ = dense_to_sparse(A)
            edge_indices = []
            for i in range(b):
                offset = i * n
                batch_edge_index = edge_index.clone()
                batch_edge_index[0] += offset
                batch_edge_index[1] += offset
                edge_indices.append(batch_edge_index)
            batch_edge_index = torch.cat(edge_indices, dim=1)
        else:
            edge_indices = []
            for i in range(b):
                edge_index, _ = dense_to_sparse(A[i])
                edge_index[0] += i * n
                edge_index[1] += i * n
                edge_indices.append(edge_index)
            batch_edge_index = torch.cat(edge_indices, dim=1)

        batch_out = self.conv(batch_x, batch_edge_index)

        # out_dim = self.out_dim if not self.concat else self.out_dim * self.heads
        return batch_out.view(b, n, -1)


# class SkipGCNGAT(nn.Module):
#     def __init__(self, num_layers, in_dim, hidden_dim, out_dim, gat_heads, final_dropout, graph_pooling_type, device, adj):
#         super(SkipGCNGAT, self).__init__()

#         self.final_dropout = final_dropout
#         self.graph_pooling_type = graph_pooling_type
#         self.device = device
#         self.num_layers = num_layers
#         self.in_dim = in_dim
#         self.hidden_dim = hidden_dim
#         self.adj = adj

#         self.gcs1 = torch.nn.ModuleList()
#         self.gcs1.append(SkipGraphConv(in_dim, hidden_dim,hidden_dim=hidden_dim))
#         for i in range(num_layers-2):
#             self.gcs1.append(SkipGraphConv(hidden_dim, hidden_dim,hidden_dim=hidden_dim))
#         self.gcs1.append(SkipGraphConv(hidden_dim, in_dim,hidden_dim=hidden_dim))

#         # first layer use multi-head, last layer use single-head
#         self.gats = torch.nn.ModuleList()
#         self.gats.append(GAT(hidden_dim, hidden_dim*2, gat_heads, concat=True))
#         self.gats.append(GAT(hidden_dim*2, hidden_dim, heads=1, concat=False))

#         self.gcs2 = torch.nn.ModuleList()
#         self.gcs2.append(SkipGraphConv(hidden_dim, hidden_dim,hidden_dim=hidden_dim))
#         for i in range(num_layers-1):
#             self.gcs2.append(SkipGraphConv(hidden_dim, hidden_dim,hidden_dim=hidden_dim))

#         self.classifier = nn.Sequential(nn.Linear(self.hidden_dim, 128), nn.Dropout(
#             p=self.final_dropout), nn.PReLU(128), nn.Linear(128, out_dim))

#     def forward(self, X_concat):
#         # x: (batch_size, num_nodes, in_dim)
#         # A: (batch_size, num_nodes, num_nodes)
#         h = X_concat
#         A = F.relu(self.adj)

#         for layers in self.gcs1:
#             h = F.relu(layers(h, A))
#         h+=X_concat

#         for layers in self.gats:
#             h = F.relu(layers(h, A))
#         h1=h

#         for layers in self.gcs2:
#             h = F.relu(layers(h, A))
#         h+=h1

#         if self.graph_pooling_type == 'mean':
#             h = h.mean(dim=1)
#         elif self.graph_pooling_type == 'sum':
#             h = h.sum(dim=1)
#         elif self.graph_pooling_type == 'max':
#             h = h.max(dim=1)[0]
#         else:
#             raise ValueError('Invalid graph pooling type')

#         ret = self.classifier(h)
#         return ret

class SkipGCNGAT(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, gat_heads, final_dropout, graph_pooling_type, device, adj):
        super(SkipGCNGAT, self).__init__()

        self.final_dropout = final_dropout
        self.graph_pooling_type = graph_pooling_type
        self.device = device
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.adj = adj

        # 输入维度投影层 - 将输入转换为hidden_dim
        self.input_projection = nn.Linear(in_dim, hidden_dim)

        # 添加层归一化以提高训练稳定性
        self.norm_input = nn.LayerNorm(hidden_dim)
        self.norm_gat = nn.LayerNorm(hidden_dim)
        self.norm_final = nn.LayerNorm(hidden_dim)

        # 第一段GCN层
        self.gcs1 = torch.nn.ModuleList()
        self.gcs1.append(SkipGraphConv(
            hidden_dim, hidden_dim, hidden_dim=hidden_dim))
        for i in range(num_layers-1):
            self.gcs1.append(SkipGraphConv(
                hidden_dim, hidden_dim, hidden_dim=hidden_dim))

        # GAT层 - 使用更一致的维度设计
        self.gats = torch.nn.ModuleList()
        self.gats.append(GAT(hidden_dim, hidden_dim*2, gat_heads, concat=True))
        self.gats.append(GAT(hidden_dim*2, hidden_dim, heads=1, concat=False))

        # 第二段GCN层
        self.gcs2 = torch.nn.ModuleList()
        self.gcs2.append(SkipGraphConv(
            hidden_dim, hidden_dim, hidden_dim=hidden_dim))
        for i in range(num_layers-1):
            self.gcs2.append(SkipGraphConv(
                hidden_dim, hidden_dim, hidden_dim=hidden_dim))

        # 跳跃连接比例参数 - 可学习的权重来平衡不同信息流
        self.alpha1 = nn.Parameter(torch.tensor(0.5))
        self.alpha2 = nn.Parameter(torch.tensor(0.5))

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.Dropout(p=self.final_dropout),
            nn.PReLU(128),
            nn.Linear(128, out_dim)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重以改善收敛性"""
        # 初始化投影层
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)

        # 初始化分类器
        nn.init.xavier_uniform_(self.classifier[0].weight)
        nn.init.zeros_(self.classifier[0].bias)
        nn.init.xavier_uniform_(self.classifier[3].weight)
        nn.init.zeros_(self.classifier[3].bias)

    def forward(self, X_concat):
        """优化的前向传播函数"""
        # 处理输入和邻接矩阵
        A = F.relu(self.adj)

        # 将输入投影到隐藏维度
        h = self.input_projection(X_concat)
        h_input = h  # 保存投影后的输入用于跳跃连接

        # 第一段GCN处理
        for layer in self.gcs1:
            h = F.relu(layer(h, A))

        # 第一个跳跃连接 - 带层归一化和可学习权重
        h = self.alpha1 * h + (1 - self.alpha1) * self.norm_input(h_input)

        # GAT处理
        h_pre_gat = h  # 保存GAT前的特征
        for i, layer in enumerate(self.gats):
            if i < len(self.gats) - 1:
                # 除了最后一层外都使用ReLU
                h = F.relu(layer(h, A))
            else:
                # 最后一层不使用激活函数
                h = layer(h, A)

        # 保存GAT后的特征用于第二个跳跃连接
        h_gat = h

        # 第二段GCN处理
        for layer in self.gcs2:
            h = F.relu(layer(h, A))

        # 第二个跳跃连接 - 带层归一化和可学习权重
        h = self.alpha2 * h + (1 - self.alpha2) * self.norm_gat(h_gat)

        # 最终归一化
        h = self.norm_final(h)

        # 图池化
        if self.graph_pooling_type == 'mean':
            h = h.mean(dim=1)
        elif self.graph_pooling_type == 'sum':
            h = h.sum(dim=1)
        elif self.graph_pooling_type == 'max':
            h = h.max(dim=1)[0]
        else:
            raise ValueError('Invalid graph pooling type')

        # 分类
        ret = self.classifier(h)
        return ret
