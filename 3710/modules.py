import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNN, self).__init__()
        # 第一层：将输入特征维度从 128 增加到 256
        self.conv1 = GCNConv(in_channels, 256)
        # 第二层：将特征维度从 256 降到 128
        self.conv2 = GCNConv(256, 128)
        # 第三层：将特征维度从 128 降到 64
        self.conv3 = GCNConv(128, 64)
        # 输出层：将特征维度从 64 降到类别数 out_channels
        self.conv4 = GCNConv(64, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # 第一层卷积 + ReLU 激活
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # 第二层卷积 + ReLU 激活
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # 第三层卷积 + ReLU 激活
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        # 输出层，使用 log_softmax
        x = self.conv4(x, edge_index)
        return F.log_softmax(x, dim=1)

