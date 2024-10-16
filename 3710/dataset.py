import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

def create_masks(labels, train_ratio=0.7, val_ratio=0.15):
    num_nodes = len(labels)
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)

    train_split = int(train_ratio * num_nodes)
    val_split = int(val_ratio * num_nodes)

    train_indices = indices[:train_split]
    val_indices = indices[train_split:train_split + val_split]
    test_indices = indices[train_split + val_split:]

    # 创建训练、验证和测试集的掩码
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    return train_mask, val_mask, test_mask

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

    # 添加自环
    edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

    # 对所有节点随机划分训练、验证和测试集
    train_mask, val_mask, test_mask = create_masks(y)

    # 创建 Data 对象
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    # 打印一些调试信息
    print("Features shape:", features.shape)
    print("Edges shape:", edges.shape)
    print("Labels shape:", y.shape)
    print("Number of nodes:", x.size(0))
    print("Number of training nodes:", train_mask.sum().item())
    print("Number of validation nodes:", val_mask.sum().item())
    print("Number of test nodes:", test_mask.sum().item())
    print("Data object:", data)

    return data
