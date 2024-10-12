import numpy as np
import torch
from torch_geometric.data import Data

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
    # Step 1: Load the .npz data file
    data = np.load(npz_file_path)

    # Step 2: Extract features, edges, and labels from the data
    features = data['features']
    edges = data['edges']
    labels = data['target']

    # Step 3: Convert features, edges, and labels into a PyTorch Geometric Data object
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    # Generate training mask for semi-supervised learning
    train_mask = create_train_mask(y)  # 使用按类别选择的方法创建训练掩码

    # Create Data object
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)

    # Print some information for debugging
    print("Features shape:", features.shape)
    print("Edges shape:", edges.shape)
    print("Labels shape:", y.shape)
    print("Number of training nodes:", train_mask.sum().item())
    print("Data object:", data)

    return data


# Call the function to load data
load_data('/Users/zhangxiangxu/Downloads/3710_report/facebook.npz')
