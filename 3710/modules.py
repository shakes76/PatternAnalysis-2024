import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm

class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super(GNN, self).__init__()
        
        self.conv1 = GCNConv(in_channels, 64)
        self.batch_norm1 = BatchNorm(64)
        
        self.conv2 = GCNConv(64, 32)
        self.batch_norm2 = BatchNorm(32)
        
        self.conv3 = GCNConv(32, 16)
        self.batch_norm3 = BatchNorm(16)
        
        self.conv4 = GCNConv(16, out_channels)
        
        self.dropout_rate = dropout_rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # 第一层卷积 + 批量归一化 + ReLU 激活 + Dropout
        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # 第二层卷积 + 批量归一化 + ReLU 激活 + Dropout
        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # 第三层卷积 + 批量归一化 + ReLU 激活 + Dropout
        x = self.conv3(x, edge_index)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # 输出层，使用 log_softmax
        x = self.conv4(x, edge_index)
        return F.log_softmax(x, dim=1)

