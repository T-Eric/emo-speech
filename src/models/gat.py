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
        self.MLP = MLP(num_layers, hidden_dim, hidden_dim, out_dim)

        # init weights and biases
        for i in range(num_layers):
            nn.init.xavier_uniform_(self.MLP.layers[i].weight)
            nn.init.constant_(self.MLP.layers[i].bias, 0.0)

    def forward(self, features, A):
        b, t, d = features.shape
        h_gcn = self.graph_conv(features, A)
        # h_skip = h_gcn+features
        h_mlp = self.MLP(
            h_gcn.view(-1, h_gcn.shape[-1])).view(b, -1, self.out_dim)
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
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
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

        # 层归一化和可学习权重
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

        h_gat = h

        for layer in self.gcs2:
            h = F.relu(layer(h, A))

        h = self.alpha2 * h + (1 - self.alpha2) * self.norm_gat(h_gat)

        h = self.norm_final(h)

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


class CustomPReLU(nn.Module):
    """原本会用第二维作为输入通道数，但是在批处理张量中应当改成3维"""

    def __init__(self, hidden_dim):
        super(CustomPReLU, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim) * 0.25)

    def forward(self, x):
        # x形状: [batch_size, num_nodes, hidden_dim]
        return torch.where(x > 0, x, self.weight.unsqueeze(0).unsqueeze(0) * x)


class EnhancedSkipGCNGAT(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, gat_heads, final_dropout, graph_pooling_type, device, adj, layer_dropout=0.1, use_multiscale_fusion=True, use_special_pooling=True):
        super(EnhancedSkipGCNGAT, self).__init__()

        self.final_dropout = final_dropout
        self.graph_pooling_type = graph_pooling_type
        self.device = device
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.adj = adj
        self.training_mode = True
        self.layer_dropout = layer_dropout
        self.use_multiscale_fusion = use_multiscale_fusion
        self.use_special_pooling = use_special_pooling

        # 输入投影
        self.input_projection = nn.Linear(in_dim, hidden_dim)

        # 层归一化
        self.norm_input = nn.LayerNorm(hidden_dim)
        self.norm_gat = nn.LayerNorm(hidden_dim)
        self.norm_final = nn.LayerNorm(hidden_dim)

        if self.use_multiscale_fusion:
            # 多尺度特征收集
            self.multiscale_projection = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim//4) for _ in range(3)
            ])
            self.feature_fusion = nn.Linear(
                hidden_dim + hidden_dim//4 * 3, hidden_dim)

        # GCN层
        self.gcs1 = torch.nn.ModuleList()
        for i in range(num_layers):
            self.gcs1.append(SkipGraphConv(
                hidden_dim, hidden_dim, hidden_dim=hidden_dim))

        # 使用修复的自定义PReLU激活函数
        self.activations = nn.ModuleList([
            CustomPReLU(hidden_dim) for _ in range(num_layers * 2 + 2)
        ])

        # GAT层
        self.gats = torch.nn.ModuleList()
        self.gats.append(GAT(hidden_dim, hidden_dim*2, gat_heads, concat=True))
        self.gats.append(GAT(hidden_dim*2, hidden_dim, heads=1, concat=False))

        # 第二段GCN
        self.gcs2 = torch.nn.ModuleList()
        for i in range(num_layers):
            self.gcs2.append(SkipGraphConv(
                hidden_dim, hidden_dim, hidden_dim=hidden_dim))

        # 跳跃连接参数
        self.alpha1 = nn.Parameter(torch.tensor(0.5))
        self.alpha2 = nn.Parameter(torch.tensor(0.5))

        if use_special_pooling:
            # 集成池化
            self.pool_attention = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )

            self.pool_fusion = nn.Linear(hidden_dim*3, hidden_dim)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),  # 改用简单的ReLU避免分类器中的维度问题
            nn.Dropout(p=self.final_dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def attention_pooling(self, x):
        # 计算每个节点的注意力分数
        scores = self.pool_attention(x)  # [batch, nodes, 1]
        scores = F.softmax(scores, dim=1)
        return torch.sum(x * scores, dim=1)

    def pooling(self, h):
        """集成多种池化方法"""
        mean_pooled = h.mean(dim=1)
        max_pooled = h.max(dim=1)[0]
        attention_pooled = self.attention_pooling(h)

        pooled = torch.cat([mean_pooled, max_pooled, attention_pooled], dim=-1)
        pooled = self.pool_fusion(pooled)

        return pooled

    def drop_edges(self, adj, dropout_prob=0.1):
        """随机丢弃一些边来增强模型鲁棒性"""
        if not self.training_mode:
            return adj

        n = adj.size(0)
        mask = torch.ones_like(adj, device=adj.device)
        indices = torch.triu_indices(n, n, 1, device=adj.device)

        vals = torch.rand(indices.size(1), device=adj.device)
        mask_vals = (vals > dropout_prob).float()

        mask[indices[0], indices[1]] = mask_vals
        mask[indices[1], indices[0]] = mask_vals

        return adj * mask

    def forward(self, X_concat, return_features=False):
        # 处理邻接矩阵，添加随机边丢弃
        A = F.relu(self.adj)
        if self.training_mode:
            A = self.drop_edges(A, dropout_prob=self.layer_dropout)

        # 投影到隐藏维度
        h = self.input_projection(X_concat)
        h_input = h

        # 收集多尺度特征
        multi_features = []

        # 第一段GCN
        act_idx = 0
        for i, layer in enumerate(self.gcs1):
            h = layer(h, A)
            h = self.activations[act_idx](h)
            act_idx += 1

            if i == len(self.gcs1) // 3:
                multi_features.append(self.multiscale_projection[0](h))

        # 第一个跳跃连接
        h = self.alpha1 * h + (1 - self.alpha1) * self.norm_input(h_input)

        # GAT处理
        for i, layer in enumerate(self.gats):
            if i < len(self.gats) - 1:
                h = layer(h, A)
                h = F.relu(h)
            else:
                h = layer(h, A)

        if len(self.gats) > 0:
            multi_features.append(self.multiscale_projection[1](h))

        h_gat = h

        # 第二段GCN
        for i, layer in enumerate(self.gcs2):
            h = layer(h, A)
            h = self.activations[act_idx](h)
            act_idx += 1

            if i == len(self.gcs2) // 2:
                multi_features.append(self.multiscale_projection[2](h))

        # 第二个跳跃连接
        h = self.alpha2 * h + (1 - self.alpha2) * self.norm_gat(h_gat)

        # 融合多尺度特征
        if len(multi_features) > 0:
            multi_concat = torch.cat([h] + multi_features, dim=-1)
            h = self.feature_fusion(multi_concat)

        # 最终归一化
        h = self.norm_final(h)

        # 保存节点特征用于返回
        node_features = h

        # 使用集成池化
        if not self.use_special_pooling:
            if self.graph_pooling_type == 'mean':
                h = h.mean(dim=1)
            elif self.graph_pooling_type == 'sum':
                h = h.sum(dim=1)
            elif self.graph_pooling_type == 'max':
                h = h.max(dim=1)[0]
            else:
                raise ValueError('Invalid graph pooling type')
        else:
            h = self.pooling(h)

        # 分类
        ret = self.classifier(h)

        if return_features:
            return ret, node_features
        return ret

    def train(self, mode=True):
        """覆盖train方法以控制训练模式"""
        super(EnhancedSkipGCNGAT, self).train(mode)
        self.training_mode = mode
        return self
