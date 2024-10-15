import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

def create_train_mask(labels, num_per_class=200):
    num_nodes = len(labels)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    unique_labels = torch.unique(labels)
    for label in unique_labels:
        label_indices = (labels == label).nonzero(as_tuple=True)[0]
        selected_indices = np.random.choice(label_indices.numpy(), min(num_per_class, len(label_indices)), replace=False)
        train_mask[selected_indices] = True

    return train_mask

def load_data(npz_file_path):
    # 第一步：加载 .npz 数据文件
    data = np.load(npz_file_path)

    # 第二步：从数据中提取特征、边和标签
    features = data['features']
    edges = data['edges']
    labels = data['target']

    # 第三步：将特征、边和标签转换为 PyTorch Geometric 的 Data 对象
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    # 添加自循环
    edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

    # 为半监督学习生成训练掩码
    train_mask = create_train_mask(y)

    # 创建 Data 对象
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)

    # 打印一些调试信息
    print("特征形状:", features.shape)
    print("边的形状:", edges.shape)
    print("标签的形状:", y.shape)
    print("训练节点数量:", train_mask.sum().item())
    print("Data 对象:", data)

    return data

# 调用函数加载数据
load_data('/Users/zhangxiangxu/Downloads/3710_report/facebook.npz')
