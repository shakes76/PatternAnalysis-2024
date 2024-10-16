import numpy as np
from torch_geometric.data import Data
import torch
from modules import *
from dataset import *
from train import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    data, masks = load_data()
    train_mask = masks[0]
    test_mask = masks[1]
    validation_mask = masks[2]

    label_validation = data.y[validation_mask]
    label_test = data.y[test_mask]
    label_train = data.y[train_mask]

    try:
        model = torch.load("GCN.pth")
        model = model.to(device)
    except FileNotFoundError:
        print(f"GCN.pth not found.")
    else:
        with torch.no_grad():
            model.eval()
            output = model(data.x, data.edge_index)
            accuracy = ((output.argmax(dim=1)[test_mask] == label_test).float()).mean()
        print(f"Test Accuracy: {100 * accuracy:.2f}%")

